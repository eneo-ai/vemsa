import asyncio
import contextlib
import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import respx
from httpx import Response
from prometheus_client import REGISTRY

from conftest import FailingEngine, FakeEngine, GateEngine, wait_for_status
from test_whisper_api import (
    API_BASE,
    ENDPOINT,
    SEGMENT_ONLY_PAYLOAD,
    SILENT_PAYLOAD,
    FakeDiarizer,
)
from vemsa.config import Settings
from vemsa.jobs import queue as queue_module
from vemsa.jobs.models import JobRequest, JobStatus, new_job
from vemsa.jobs.postgres_store import PostgresJobStore
from vemsa.jobs.queue import JobQueue
from vemsa.jobs.store import JobStore
from vemsa.pipeline.hybrid import HybridEngine
from vemsa.pipeline.whisper_api import OpenAIWhisperEngine


@contextlib.asynccontextmanager
async def running_queue(store: JobStore, engine, settings: Settings):
    queue = JobQueue(store, engine, settings)
    await queue.start()
    try:
        yield queue
    finally:
        await queue.stop()


async def wait_for_row(store: JobStore, query: str, timeout: float = 5.0):
    async with asyncio.timeout(timeout):
        while True:  # noqa: ASYNC110 — polling the database, no event to wait on
            if row := await store.pool.fetchrow(query):
                return row
            await asyncio.sleep(0.05)


def upload_job(tmp_path: Path, **request_kwargs):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake audio bytes")
    return new_job(JobRequest(**request_kwargs), audio_path=str(audio)), audio


@respx.mock
@pytest.mark.parametrize("slow_control_start", [False, True])
async def test_lease_heartbeat_precedes_and_covers_source_fetch(
    store: JobStore, settings: Settings, monkeypatch, slow_control_start: bool
):
    settings.job_lease_s = 0.3
    settings.lease_heartbeat_s = 0.05
    settings.queue_poll_interval_s = 0.01
    if slow_control_start:
        start_control = JobQueue._start_control_thread

        async def delayed_control_start(self):
            await asyncio.sleep(0.1)
            await start_control(self)

        monkeypatch.setattr(JobQueue, "_start_control_thread", delayed_control_start)
    first_renewal = threading.Event()
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()
    heartbeat_before_fetch = False
    renew_lease = PostgresJobStore.renew_lease

    async def observe_renewal(self, *args, **kwargs):
        renewed = await renew_lease(self, *args, **kwargs)
        if renewed:
            first_renewal.set()
        return renewed

    async def fetch(request):
        nonlocal heartbeat_before_fetch
        heartbeat_before_fetch = first_renewal.is_set()
        fetch_started.set()
        await release_fetch.wait()
        return Response(200, content=b"audio")

    monkeypatch.setattr(PostgresJobStore, "renew_lease", observe_renewal)
    respx.get("https://example.org/slow.mp3").mock(side_effect=fetch)
    job = new_job(JobRequest(source_url="https://example.org/slow.mp3"))
    await store.create(job)
    async with running_queue(store, FakeEngine(), settings):
        try:
            current = await store.get(job.id)
            assert current is not None and current.status != JobStatus.FAILED
            async with asyncio.timeout(5):
                await fetch_started.wait()
            assert heartbeat_before_fetch
            await asyncio.sleep(0.5)
            assert await store.claim_next_queued(worker_id="other-worker") is None
        finally:
            release_fetch.set()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)


@pytest.mark.parametrize("include_review", [False, True])
async def test_upload_job_completes_and_audio_deleted(
    store: JobStore, settings: Settings, tmp_path: Path, include_review: bool
):
    job, audio = upload_job(tmp_path, language="sv", include_speaker_review=include_review)
    await store.create(job)
    engine = FakeEngine()

    async with running_queue(store, engine, settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)

    assert not audio.exists()
    result = await store.get_result(job.id)
    assert result is not None and result.language == "sv"
    assert engine.calls == [
        {
            "audio_path": audio,
            "language": "sv",
            "model": settings.default_model,
            "diarize": True,
            "speakers": None,
            "include_speaker_review": include_review,
            "vocabulary": None,
        }
    ]


async def test_alignment_floor_fails_a_degraded_job(
    store: JobStore, settings: Settings, tmp_path: Path
):
    # FakeEngine reports no alignment rung, which ranks below every floor
    settings.min_alignment = "forced"
    job, audio = upload_job(tmp_path, language="sv")
    await store.create(job)

    async with running_queue(store, FakeEngine(), settings) as queue:
        queue.notify()
        failed = await wait_for_status(store, job.id, JobStatus.FAILED)

    assert failed.error is not None and "VEMSA_MIN_ALIGNMENT" in failed.error
    assert not audio.exists()


async def test_alignment_floor_passes_a_forced_result(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings.min_alignment = "forced"

    class ForcedEngine(FakeEngine):
        def transcribe(self, *args, **kwargs):
            return super().transcribe(*args, **kwargs).model_copy(update={"alignment": "forced"})

    job, _ = upload_job(tmp_path, language="sv")
    await store.create(job)

    async with running_queue(store, ForcedEngine(), settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)


@respx.mock
@pytest.mark.parametrize("engine_class", [HybridEngine, OpenAIWhisperEngine])
@pytest.mark.parametrize("floor", [None, "forced"])
async def test_silent_audio_completes_with_an_empty_transcript(
    store: JobStore, settings: Settings, tmp_path: Path, engine_class, floor
):
    # a chunk without speech has no timestamps to grade, so no floor fails it
    settings.whisper_api_base = API_BASE
    settings.min_alignment = floor
    respx.post(ENDPOINT).mock(return_value=Response(200, json=SILENT_PAYLOAD))
    diarizer = FakeDiarizer()
    job, audio = upload_job(tmp_path, language="sv", diarize=True)
    await store.create(job)

    async with running_queue(store, engine_class(settings, diarizer=diarizer), settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)

    result = await store.get_result(job.id)
    assert result is not None
    assert (result.text, result.segments, result.alignment) == ("", [], None)
    assert (result.duration_seconds, result.language) == (0.74, "sv")
    assert diarizer.calls == []
    assert not audio.exists()


@respx.mock
async def test_alignment_floor_still_fails_segment_text_without_words(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings.whisper_api_base = API_BASE
    settings.min_alignment = "forced"
    respx.post(ENDPOINT).mock(return_value=Response(200, json=SEGMENT_ONLY_PAYLOAD))
    job, _ = upload_job(tmp_path, language="sv", diarize=True)
    await store.create(job)

    engine = OpenAIWhisperEngine(settings, diarizer=FakeDiarizer())
    async with running_queue(store, engine, settings) as queue:
        queue.notify()
        failed = await wait_for_status(store, job.id, JobStatus.FAILED)

    assert failed.error is not None
    assert "segment_only" in failed.error and "VEMSA_MIN_ALIGNMENT" in failed.error


def align_job(tmp_path: Path):
    return upload_job(
        tmp_path,
        task="align",
        language="sv",
        segments=[{"start": 0.0, "end": 2.0, "speaker": "Anna", "text": "hej och välkomna"}],
    )


class InterpolatingEngine(FakeEngine):
    """A forced-rung result where two of three words carry easyaligner's
    fallback score (0.0): their timestamps were interpolated, not aligned."""

    def align_transcript(self, *args, **kwargs):
        result = super().align_transcript(*args, **kwargs)
        (segment,) = result.segments
        words = [
            word.model_copy(update={"probability": 0.0 if index < 2 else 0.9})
            for index, word in enumerate(segment.words)
        ]
        return result.model_copy(update={"segments": [segment.model_copy(update={"words": words})]})


async def test_align_job_completes_with_speakers_verbatim(
    store: JobStore, settings: Settings, tmp_path: Path
):
    job, audio = align_job(tmp_path)
    await store.create(job)
    engine = FakeEngine()
    async with running_queue(store, engine, settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)
    assert not audio.exists()
    result = await store.get_result(job.id)
    assert result is not None and result.alignment == "forced"
    assert [(s.speaker, s.text) for s in result.segments] == [("Anna", "hej och välkomna")]
    assert engine.calls[-1]["task"] == "align" and engine.calls[-1]["model"] == "external"


async def test_interpolated_share_floor_fails_the_job(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings.align_max_interpolated_share = 0.5
    job, _ = align_job(tmp_path)
    await store.create(job)
    async with running_queue(store, InterpolatingEngine(), settings) as queue:
        queue.notify()
        failed = await wait_for_status(store, job.id, JobStatus.FAILED)
    assert failed.error is not None and "VEMSA_ALIGN_MAX_INTERPOLATED_SHARE" in failed.error
    assert "2 of 3 words" in failed.error


async def test_interpolated_words_are_tolerated_by_default(
    store: JobStore, settings: Settings, tmp_path: Path
):
    job, _ = align_job(tmp_path)
    await store.create(job)
    async with running_queue(store, InterpolatingEngine(), settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)


async def test_failing_engine_marks_job_failed(store: JobStore, settings: Settings, tmp_path: Path):
    job, audio = upload_job(tmp_path)
    await store.create(job)

    async with running_queue(store, FailingEngine(), settings) as queue:
        queue.notify()
        failed = await wait_for_status(store, job.id, JobStatus.FAILED)

    assert failed.error is not None and "RuntimeError" in failed.error
    assert not audio.exists()


@respx.mock
async def test_source_url_is_fetched_to_work_dir(store: JobStore, settings: Settings):
    respx.get("https://example.org/meeting.mp3").mock(
        return_value=Response(200, content=b"downloaded audio")
    )
    job = new_job(JobRequest(source_url="https://example.org/meeting.mp3"))
    await store.create(job)
    engine = FakeEngine()

    async with running_queue(store, engine, settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)

    downloaded = engine.calls[0]["audio_path"]
    assert downloaded.parent == settings.work_dir
    assert not downloaded.exists()


@respx.mock
async def test_webhook_delivered_on_completion(store: JobStore, settings: Settings, tmp_path: Path):
    route = respx.post("https://hooks.example.org/done").mock(return_value=Response(200))
    job, _ = upload_job(tmp_path, webhook_url="https://hooks.example.org/done")
    await store.create(job)

    async with running_queue(store, FakeEngine(), settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)

    assert route.called
    import json

    payload = json.loads(route.calls.last.request.content)
    assert payload["job_id"] == job.id
    assert payload["status"] == "completed"
    assert payload["result"]["segments"][0]["speaker"] == "SPEAKER_00"


@respx.mock
async def test_webhook_is_signed_when_secret_is_configured(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings.webhook_signing_secret = "signing-secret"
    route = respx.post("https://hooks.example.org/done").mock(return_value=Response(200))
    job, _ = upload_job(tmp_path, webhook_url="https://hooks.example.org/done")
    await store.create(job)

    async with running_queue(store, FakeEngine(), settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)

    request = route.calls.last.request
    assert request.headers["X-Vemsa-Timestamp"]
    assert request.headers["X-Vemsa-Signature-256"].startswith("sha256=")


@respx.mock
async def test_webhook_failure_does_not_fail_job(
    store: JobStore, settings: Settings, tmp_path: Path
):
    route = respx.post("https://hooks.example.org/done").mock(return_value=Response(500))
    job, _ = upload_job(tmp_path, webhook_url="https://hooks.example.org/done")
    await store.create(job)

    async with running_queue(store, FakeEngine(), settings) as queue:
        queue.notify()
        completed = await wait_for_status(store, job.id, JobStatus.COMPLETED)
        row = await wait_for_row(store, "SELECT * FROM webhook_outbox WHERE attempt = 1")
        assert queue.healthy

    assert route.called
    assert completed.status == JobStatus.COMPLETED
    assert row["last_error"] == "HTTP 500" and row["next_attempt_at"] is not None
    assert row["lease_owner"] is None and row["lease_expires_at"] is None


async def cancelled_with_webhook(store: JobStore, webhook_url: str) -> str:
    """A cancelled job whose cancellation webhook is due now; returns the job id.

    Created before the queue starts, so no worker claims the job itself."""
    job = new_job(JobRequest(source_url="https://example.org/a.mp3"))
    await store.create(job)
    assert await store.cancel(job.id, client_id=job.client_id, webhook_url=webhook_url)
    return job.id


def webhook_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name.startswith("vemsa")]


# the text log format set up by vemsa.observability.configure_logging
TEXT_LOG = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")


def store_failures(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == "webhook.store_failed"]


def webhook_deliveries() -> dict[str, float | None]:
    return {
        outcome: REGISTRY.get_sample_value("vemsa_webhook_deliveries_total", {"outcome": outcome})
        for outcome in ("success", "retry", "exhausted", "rejected", "lost")
    }


@respx.mock
async def test_webhook_retries_an_unexpected_error_and_keeps_delivering(
    store: JobStore, settings: Settings
):
    settings.webhook_max_attempts = 2
    settings.webhook_poll_interval_s = 0.05
    failing = respx.post("https://hooks.example.org/failing").mock(
        side_effect=RuntimeError("unexpected")
    )
    working = respx.post("https://hooks.example.org/working").mock(return_value=Response(200))
    failing_job = await cancelled_with_webhook(store, "https://hooks.example.org/failing")
    await cancelled_with_webhook(store, "https://hooks.example.org/working")

    async with running_queue(store, FakeEngine(), settings):
        await wait_until(lambda: working.called)
        exhausted = await wait_for_row(
            store, "SELECT * FROM webhook_outbox WHERE next_attempt_at IS NULL"
        )

    assert failing.call_count == 2
    assert exhausted["job_id"] == failing_job and exhausted["attempt"] == 2
    assert exhausted["last_error"] == "RuntimeError"
    remaining = await store.pool.fetch("SELECT job_id FROM webhook_outbox")
    assert [row["job_id"] for row in remaining] == [failing_job]


@respx.mock
async def test_webhook_loop_survives_a_store_error(
    store: JobStore, settings: Settings, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings.webhook_poll_interval_s = 0.05
    settings.queue_poll_interval_s = 0.05
    route = respx.post("https://hooks.example.org/done").mock(return_value=Response(200))
    claim_webhook = PostgresJobStore.claim_webhook
    failures = []

    async def flaky_claim(self, *args, **kwargs):
        if not failures:
            failures.append(True)
            raise OSError("connection lost storemark42")
        return await claim_webhook(self, *args, **kwargs)

    monkeypatch.setattr(PostgresJobStore, "claim_webhook", flaky_claim)
    await cancelled_with_webhook(store, "https://hooks.example.org/done")

    async with running_queue(store, FakeEngine(), settings) as queue:
        await wait_until(lambda: route.called)
        await wait_for_row(store, "SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM webhook_outbox)")
        assert queue.healthy

    assert failures
    (failed,) = store_failures(caplog)
    # no row was claimed, so there is no job id to name
    assert failed.exc_info is None and failed.job_id is None
    line = TEXT_LOG.format(failed)
    assert "webhook store call failed: OSError" in line and "for job" not in line
    for record in webhook_records(caplog):
        assert "storemark42" not in json.dumps(record.__dict__, default=str)


@respx.mock
async def test_webhook_is_delivered_again_when_its_receipt_is_not_stored(
    store: JobStore, settings: Settings, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings.job_lease_s = 0.5
    settings.lease_heartbeat_s = 0.1
    settings.webhook_poll_interval_s = 0.05
    # were the lost receipt counted as an attempt, the row would end here
    settings.webhook_max_attempts = 1
    route = respx.post("https://hooks.example.org/done").mock(return_value=Response(200))
    mark_delivered = PostgresJobStore.mark_webhook_delivered
    receipt_failed = asyncio.Event()

    async def flaky_mark(self, *args, **kwargs):
        if not receipt_failed.is_set():
            receipt_failed.set()
            raise OSError("connection lost")
        await mark_delivered(self, *args, **kwargs)

    monkeypatch.setattr(PostgresJobStore, "mark_webhook_delivered", flaky_mark)
    job_id = await cancelled_with_webhook(store, "https://hooks.example.org/done")

    async with running_queue(store, FakeEngine(), settings):
        await asyncio.wait_for(receipt_failed.wait(), timeout=5)
        pending = await store.pool.fetchrow("SELECT * FROM webhook_outbox")
        assert pending is not None
        assert pending["attempt"] == 0 and pending["last_error"] is None
        assert pending["lease_owner"] is not None
        await wait_for_row(store, "SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM webhook_outbox)")

    assert route.call_count == 2
    (failed,) = store_failures(caplog)
    assert failed.job_id == job_id
    assert f"webhook store call failed for job {job_id}: OSError" in TEXT_LOG.format(failed)


@respx.mock
@pytest.mark.parametrize("wake_during_wait", [False, True])
async def test_webhook_loop_waits_one_poll_interval_after_a_store_error(
    store: JobStore, settings: Settings, monkeypatch, wake_during_wait: bool
):
    poll = 1.0
    settings.webhook_poll_interval_s = poll
    loop = asyncio.get_running_loop()
    times: dict[str, float] = {}
    reschedule = PostgresJobStore.reschedule_webhook

    async def flaky_reschedule(self, *args, **kwargs):
        if "failed" not in times:
            times["failed"] = loop.time()
            raise OSError("connection lost")
        return await reschedule(self, *args, **kwargs)

    def deliver(request):
        times.setdefault("delivered", loop.time())
        return Response(200)

    monkeypatch.setattr(PostgresJobStore, "reschedule_webhook", flaky_reschedule)
    respx.post("https://hooks.example.org/failing").mock(return_value=Response(500))
    respx.post("https://hooks.example.org/working").mock(side_effect=deliver)
    failing_job = await cancelled_with_webhook(store, "https://hooks.example.org/failing")
    await cancelled_with_webhook(store, "https://hooks.example.org/working")

    async with running_queue(store, FakeEngine(), settings) as queue:
        await wait_until(lambda: "failed" in times)
        if wake_during_wait:
            # a wakeup while the store is failing does not cut the wait short
            queue._webhook_wakeup.set()
        await wait_until(lambda: "delivered" in times)
        failing = await store.pool.fetchrow(
            "SELECT * FROM webhook_outbox WHERE job_id = $1", failing_job
        )

    assert poll <= times["delivered"] - times["failed"] < 1.75 * poll
    # the failed row keeps its lease and is claimed again once it lapses
    assert failing["attempt"] == 0 and failing["last_error"] is None
    assert failing["lease_owner"] is not None


async def test_stop_is_prompt_while_the_webhook_store_keeps_failing(
    store: JobStore, settings: Settings, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings.webhook_poll_interval_s = 10.0
    settings.shutdown_grace_s = 2.0

    async def broken_claim(self, *args, **kwargs):
        raise OSError("connection lost")

    monkeypatch.setattr(PostgresJobStore, "claim_webhook", broken_claim)
    async with running_queue(store, FakeEngine(), settings) as queue:
        await wait_until(lambda: bool(store_failures(caplog)))
        assert queue.healthy

    assert not any("graceful shutdown timeout" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("failure", ["http_error", "rejected"])
async def test_webhook_taken_over_during_its_attempt_is_left_alone(
    store: JobStore, settings: Settings, monkeypatch, caplog, failure: str
):
    caplog.set_level(logging.INFO)
    settings.webhook_poll_interval_s = 0.05

    async def take_over() -> None:
        await store.pool.execute("UPDATE webhook_outbox SET lease_owner = 'other-worker'")

    async def failing_receiver(request):
        await take_over()
        return Response(500)

    router = respx.mock(assert_all_called=False)
    router.post("https://hooks.example.org/done").mock(side_effect=failing_receiver)
    if failure == "rejected":
        settings.webhook_allowed_hosts = ["elsewhere.example.org"]
        validate = queue_module.validate_outbound_url

        async def validate_after_take_over(*args, **kwargs):
            await take_over()
            await validate(*args, **kwargs)

        monkeypatch.setattr(queue_module, "validate_outbound_url", validate_after_take_over)
    job_id = await cancelled_with_webhook(store, "https://hooks.example.org/done")
    before = webhook_deliveries()

    with router:
        async with running_queue(store, FakeEngine(), settings):
            await wait_until(
                lambda: any("no longer leased" in r.getMessage() for r in caplog.records)
            )

    assert webhook_deliveries() == before
    messages = [r.getMessage() for r in webhook_records(caplog)]
    assert f"webhook for job {job_id} is no longer leased by this worker" in messages
    assert not any("was rejected" in m or "failed on attempt" in m for m in messages)
    row = await store.pool.fetchrow("SELECT * FROM webhook_outbox")
    assert row["lease_owner"] == "other-worker" and row["attempt"] == 0
    assert row["last_error"] is None


async def test_expired_worker_lease_is_reclaimed(
    store: JobStore, settings: Settings, tmp_path: Path
):
    job, _ = upload_job(tmp_path)
    await store.create(job)
    claimed = await store.claim_next_queued(lease_for_s=-1)
    assert claimed is not None and claimed.status == JobStatus.RUNNING

    async with running_queue(store, FakeEngine(), settings):
        await wait_for_status(store, job.id, JobStatus.COMPLETED)


async def test_purge_once_deletes_rows_and_audio(
    store: JobStore, settings: Settings, tmp_path: Path
):
    from conftest import make_result

    job, audio = upload_job(tmp_path)
    await store.create(job)
    await store.claim_next_queued()
    await store.finish(job.id, make_result())

    settings.retention_hours = 0.0
    queue = JobQueue(store, FakeEngine(), settings)
    await queue.purge_once()

    assert await store.get(job.id) is None
    assert not audio.exists()


# --- bounded in-process concurrency -------------------------------------------


def upload_jobs(tmp_path: Path, count: int, **request_kwargs):
    """`count` upload jobs, each with its own audio file, in creation order."""
    created = []
    for index in range(count):
        directory = tmp_path / f"job-{index}"
        directory.mkdir()
        created.append(upload_job(directory, **request_kwargs))
    return created


async def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


async def test_two_jobs_run_concurrently_when_concurrency_is_two(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings = settings.model_copy(update={"worker_concurrency": 2, "gpu_concurrency": 2})
    (job_a, audio_a), (job_b, audio_b) = upload_jobs(tmp_path, 2)
    await store.create(job_a)
    await store.create(job_b)
    engine = GateEngine()

    async with running_queue(store, engine, settings) as queue:
        entered = {await engine.wait_entered(), await engine.wait_entered()}
        assert {path for path, _ in entered} == {audio_a, audio_b}
        assert queue.in_flight == 2
        for job in (job_a, job_b):
            current = await store.get(job.id)
            assert current is not None and current.status == JobStatus.RUNNING
        engine.gate.set()
        await wait_for_status(store, job_a.id, JobStatus.COMPLETED)
        await wait_for_status(store, job_b.id, JobStatus.COMPLETED)

    assert engine.peak == 2
    assert not audio_a.exists() and not audio_b.exists()


async def test_jobs_stay_serial_by_default(store: JobStore, settings: Settings, tmp_path: Path):
    assert settings.worker_concurrency == 1
    (job_a, audio_a), (job_b, audio_b) = upload_jobs(tmp_path, 2)
    await store.create(job_a)
    await store.create(job_b)
    engine = GateEngine()

    async with running_queue(store, engine, settings):
        first, _ = await engine.wait_entered()
        assert first == audio_a
        await asyncio.sleep(0.3)
        second = await store.get(job_b.id)
        assert second is not None and second.status == JobStatus.QUEUED
        assert engine.entered.empty()
        engine.gate.set()
        await wait_for_status(store, job_a.id, JobStatus.COMPLETED)
        await wait_for_status(store, job_b.id, JobStatus.COMPLETED)
        assert (await engine.wait_entered())[0] == audio_b

    assert engine.peak == 1


async def test_third_job_waits_for_a_free_slot(store: JobStore, settings: Settings, tmp_path: Path):
    settings = settings.model_copy(update={"worker_concurrency": 2, "gpu_concurrency": 2})
    jobs = upload_jobs(tmp_path, 3)
    for job, _ in jobs:
        await store.create(job)
    engine = GateEngine()

    async with running_queue(store, engine, settings):
        await engine.wait_entered()
        await engine.wait_entered()
        await asyncio.sleep(0.3)
        third = await store.get(jobs[2][0].id)
        assert third is not None and third.status == JobStatus.QUEUED
        assert engine.entered.empty()
        engine.gate.set()
        for job, _ in jobs:
            await wait_for_status(store, job.id, JobStatus.COMPLETED)

    assert engine.peak == 2


async def test_stop_drains_in_flight_jobs(store: JobStore, settings: Settings, tmp_path: Path):
    settings = settings.model_copy(update={"worker_concurrency": 2, "gpu_concurrency": 2})
    (job_a, _), (job_b, _) = upload_jobs(tmp_path, 2)
    await store.create(job_a)
    await store.create(job_b)
    engine = GateEngine()
    queue = JobQueue(store, engine, settings)
    await queue.start()
    await engine.wait_entered()
    await engine.wait_entered()

    stop_task = asyncio.create_task(queue.stop())
    await asyncio.sleep(0.2)
    assert not stop_task.done()
    engine.gate.set()
    await asyncio.wait_for(stop_task, timeout=10.0)

    for job in (job_a, job_b):
        current = await store.get(job.id)
        assert current is not None and current.status == JobStatus.COMPLETED


async def test_stop_does_not_claim_new_jobs(store: JobStore, settings: Settings, tmp_path: Path):
    (job_a, _), (job_b, audio_b) = upload_jobs(tmp_path, 2)
    await store.create(job_a)
    await store.create(job_b)
    engine = GateEngine()
    queue = JobQueue(store, engine, settings)
    await queue.start()
    await engine.wait_entered()

    stop_task = asyncio.create_task(queue.stop())
    await asyncio.sleep(0.1)
    engine.gate.set()
    await asyncio.wait_for(stop_task, timeout=10.0)

    assert (await wait_for_status(store, job_a.id, JobStatus.COMPLETED)).status
    second = await store.get(job_b.id)
    assert second is not None and second.status == JobStatus.QUEUED
    assert audio_b not in engine.calls_per_path


async def test_worker_does_not_reclaim_its_own_in_flight_job(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings = settings.model_copy(update={"worker_concurrency": 2, "gpu_concurrency": 2})
    job, audio = upload_job(tmp_path)
    await store.create(job)
    engine = GateEngine()

    async with running_queue(store, engine, settings) as queue:
        await engine.wait_entered()
        # the lease lapses while the pipeline thread is still running the job
        await store.pool.execute(  # type: ignore[attr-defined]
            "UPDATE jobs SET lease_expires_at = now() - interval '1 hour' WHERE id = $1", job.id
        )
        queue.notify()
        await asyncio.sleep(0.3)
        assert engine.calls_per_path[audio] == 1
        assert engine.entered.empty()
        engine.gate.set()
        completed = await wait_for_status(store, job.id, JobStatus.COMPLETED)

    assert completed.attempt == 1
    assert engine.calls_per_path[audio] == 1


class OutOfMemoryError(RuntimeError):  # noqa: N818 — mirrors torch.cuda.OutOfMemoryError
    pass


@pytest.mark.parametrize("failure", [MemoryError, OutOfMemoryError])
async def test_out_of_memory_requeues_and_succeeds_on_retry(
    store: JobStore, settings: Settings, tmp_path: Path, failure: type[BaseException]
):
    settings = settings.model_copy(update={"oom_retry_delay_s": 0.0})
    job, audio = upload_job(tmp_path)
    await store.create(job)
    engine = GateEngine(fail_first_with=failure)
    engine.gate.set()

    async with running_queue(store, engine, settings):
        completed = await wait_for_status(store, job.id, JobStatus.COMPLETED)

    assert completed.attempt == 2
    assert completed.error is None
    assert completed.failure_kind is None
    assert engine.calls_per_path[audio] == 2
    assert not audio.exists()


async def test_out_of_memory_keeps_audio_and_cooldown_between_attempts(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings = settings.model_copy(update={"oom_retry_delay_s": 3600.0})
    job, audio = upload_job(tmp_path)
    await store.create(job)
    engine = GateEngine(fail_first_with=MemoryError)
    engine.gate.set()

    async with running_queue(store, engine, settings) as queue:
        await engine.wait_entered()
        await wait_until(lambda: queue.in_flight == 0)
        await asyncio.sleep(0.3)
        requeued = await store.get(job.id)

    assert requeued is not None and requeued.status == JobStatus.QUEUED
    assert requeued.failure_kind is None
    assert requeued.lease_owner is None and requeued.attempt == 1
    assert audio.exists()
    assert engine.calls_per_path[audio] == 1


async def test_out_of_memory_fails_when_attempts_are_exhausted(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings = settings.model_copy(update={"oom_max_attempts": 1, "oom_retry_delay_s": 0.0})
    job, audio = upload_job(tmp_path)
    await store.create(job)
    engine = GateEngine(fail_first_with=MemoryError)
    engine.gate.set()

    async with running_queue(store, engine, settings):
        failed = await wait_for_status(store, job.id, JobStatus.FAILED)

    assert failed.attempt == 1
    assert failed.failure_kind == "capacity"
    assert failed.error is not None and failed.error.startswith("MemoryError")
    assert not audio.exists()


async def test_out_of_memory_release_respects_cancellation(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings = settings.model_copy(update={"oom_retry_delay_s": 0.0})
    job, audio = upload_job(tmp_path)
    await store.create(job)
    engine = GateEngine(fail_first_with=MemoryError)

    async with running_queue(store, engine, settings) as queue:
        await engine.wait_entered()
        assert await store.cancel(job.id, client_id=job.client_id) is not None
        engine.gate.set()
        await wait_until(lambda: queue.in_flight == 0)
        await asyncio.sleep(0.2)
        current = await store.get(job.id)

    assert current is not None and current.status == JobStatus.CANCELLED
    assert engine.calls_per_path[audio] == 1
    assert not audio.exists()


async def test_pipeline_threads_carry_their_own_job_id(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings = settings.model_copy(update={"worker_concurrency": 2, "gpu_concurrency": 2})
    (job_a, audio_a), (job_b, audio_b) = upload_jobs(tmp_path, 2)
    await store.create(job_a)
    await store.create(job_b)
    engine = GateEngine()

    async with running_queue(store, engine, settings):
        entered = dict([await engine.wait_entered(), await engine.wait_entered()])
        engine.gate.set()
        await wait_for_status(store, job_a.id, JobStatus.COMPLETED)
        await wait_for_status(store, job_b.id, JobStatus.COMPLETED)

    assert entered == {audio_a: job_a.id, audio_b: job_b.id}


async def test_completed_job_records_stats(store: JobStore, settings: Settings, tmp_path: Path):
    job, _audio = upload_job(tmp_path, language="sv")
    await store.create(job)

    async with running_queue(store, FakeEngine(), settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.COMPLETED)

    row = await store.pool.fetchrow("SELECT * FROM job_stats WHERE job_id = $1", job.id)
    assert row is not None
    assert row["status"] == "completed" and row["task"] == "transcribe"
    assert row["engine"] == "fake" and row["model"] == "KBLab/kb-whisper-large"
    assert row["language"] == "sv" and row["device"] in {"cpu", "cuda"}
    assert row["worker_id"] is not None and row["attempts"] == 1
    # the fixture bytes are not decodable audio; the result's duration wins
    assert row["audio_seconds"] == 2.6
    assert row["processing_s"] > 0
    # finalizing is the commit itself, so it cannot be part of the committed row
    stages = json.loads(row["stage_seconds"])
    assert set(stages) == {"queued", "transcribing"}
    assert all(seconds >= 0 for seconds in stages.values())


async def test_failed_job_records_error_class(store: JobStore, settings: Settings, tmp_path: Path):
    job, _audio = upload_job(tmp_path)
    await store.create(job)

    async with running_queue(store, FailingEngine(), settings) as queue:
        queue.notify()
        await wait_for_status(store, job.id, JobStatus.FAILED)

    row = await store.pool.fetchrow("SELECT * FROM job_stats WHERE job_id = $1", job.id)
    assert row is not None
    assert row["status"] == "failed" and row["error_class"] == "RuntimeError"
    assert row["audio_seconds"] is None
    # FailingEngine explodes before reporting a stage: only the setup time is known
    assert set(json.loads(row["stage_seconds"])) == {"queued"}


async def test_out_of_memory_retry_is_counted_in_stats(
    store: JobStore, settings: Settings, tmp_path: Path
):
    settings = settings.model_copy(update={"oom_retry_delay_s": 0.0})
    job, _audio = upload_job(tmp_path)
    await store.create(job)
    engine = GateEngine(fail_first_with=MemoryError)
    engine.gate.set()

    async with running_queue(store, engine, settings):
        await wait_for_status(store, job.id, JobStatus.COMPLETED)

    row = await store.pool.fetchrow("SELECT * FROM job_stats WHERE job_id = $1", job.id)
    assert row is not None and row["status"] == "completed" and row["attempts"] == 2
    assert row["started_at"] is not None


async def test_heartbeat_records_a_worker_sample(store: JobStore, settings: Settings):
    async with running_queue(store, FakeEngine(), settings):
        row = await wait_for_row(store, "SELECT * FROM worker_samples")

    assert row["engine"] == "fake" and row["hostname"]
    assert row["concurrency"] == 1 and row["gpu_concurrency"] == 1 and row["in_flight"] == 0
    assert row["device"] in {"cpu", "cuda"}
    assert row["mem_total_bytes"] is not None and row["mem_total_bytes"] > 0


async def test_purge_once_drops_stats_past_retention(store: JobStore, settings: Settings):
    now = datetime.now(UTC)
    for job_id, finished_at in (("old", now - timedelta(days=100)), ("recent", now)):
        await store.pool.execute(
            "INSERT INTO job_stats (job_id, client_id, status, task, attempts, created_at,"
            " finished_at) VALUES ($1, 'alpha', 'completed', 'transcribe', 1, $2, $2)",
            job_id,
            finished_at,
        )
    settings.stats_retention_days = 90.0
    queue = JobQueue(store, FakeEngine(), settings)

    await queue.purge_once()

    assert [row["job_id"] for row in await store.pool.fetch("SELECT job_id FROM job_stats")] == [
        "recent"
    ]
