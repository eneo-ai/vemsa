import asyncio
import contextlib
from pathlib import Path

import pytest
import respx
from httpx import Response

from conftest import FailingEngine, FakeEngine, GateEngine, wait_for_status
from vemsa.config import Settings
from vemsa.jobs.models import JobRequest, JobStatus, new_job
from vemsa.jobs.queue import JobQueue
from vemsa.jobs.store import JobStore


@contextlib.asynccontextmanager
async def running_queue(store: JobStore, engine, settings: Settings):
    queue = JobQueue(store, engine, settings)
    queue.webhook_retry_delays = ()
    await queue.start()
    try:
        yield queue
    finally:
        await queue.stop()


def upload_job(tmp_path: Path, **request_kwargs):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake audio bytes")
    return new_job(JobRequest(**request_kwargs), audio_path=str(audio)), audio


async def test_upload_job_completes_and_audio_deleted(
    store: JobStore, settings: Settings, tmp_path: Path
):
    job, audio = upload_job(tmp_path, language="sv")
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

    assert route.called
    assert completed.status == JobStatus.COMPLETED


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
