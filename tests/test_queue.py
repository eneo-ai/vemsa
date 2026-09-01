import contextlib
from pathlib import Path

import respx
from httpx import Response

from conftest import FailingEngine, FakeEngine, wait_for_status
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
