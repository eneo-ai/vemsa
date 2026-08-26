from datetime import UTC, datetime, timedelta

from conftest import make_result
from tolka.jobs.models import Job, JobRequest, JobStatus
from tolka.jobs.store import SqliteJobStore


def job_at(seconds: int, **request_kwargs) -> Job:
    when = datetime(2026, 1, 1, 12, 0, seconds, tzinfo=UTC)
    return Job(
        id=f"job-{seconds:02d}",
        status=JobStatus.QUEUED,
        created_at=when,
        updated_at=when,
        request=JobRequest(source_url="https://example.org/a.mp3", **request_kwargs),
    )


async def test_create_and_get_roundtrip(store: SqliteJobStore):
    job = job_at(0, language="sv", diarize=False)
    await store.create(job)

    loaded = await store.get(job.id)
    assert loaded is not None
    assert loaded.status == JobStatus.QUEUED
    assert loaded.request.language == "sv"
    assert loaded.request.diarize is False
    assert str(loaded.request.source_url) == "https://example.org/a.mp3"
    assert loaded.created_at == job.created_at

    assert await store.get("nope") is None


async def test_claim_is_fifo_and_transitions_to_running(store: SqliteJobStore):
    await store.create(job_at(1))
    await store.create(job_at(0))

    first = await store.claim_next_queued()
    second = await store.claim_next_queued()
    assert first is not None and first.id == "job-00"
    assert second is not None and second.id == "job-01"
    assert first.status == JobStatus.RUNNING
    assert await store.claim_next_queued() is None
    assert await store.count_queued() == 0


async def test_queue_position_is_global_fifo_but_scoped_to_job_owner(store: SqliteJobStore):
    first = job_at(0)
    first.client_id = "alpha"
    second = job_at(1)
    second.client_id = "beta"
    await store.create(first)
    await store.create(second)

    assert await store.queue_position(first.id, client_id="alpha") == 1
    assert await store.queue_position(second.id, client_id="beta") == 2
    assert await store.queue_position(second.id, client_id="alpha") is None


async def test_cancel_is_terminal_and_rejects_late_worker_result(store: SqliteJobStore):
    job = job_at(0, webhook_url="https://hooks.example.org/cancelled")
    job.client_id = "alpha"
    await store.create(job)
    claimed = await store.claim_next_queued(worker_id="worker-one")
    assert claimed is not None

    cancelled = await store.cancel(
        job.id,
        client_id="alpha",
        webhook_url="https://hooks.example.org/cancelled",
    )
    assert cancelled is not None
    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.cancellation_requested_at is not None
    assert await store.count_active(client_id="alpha") == 0
    assert not await store.finish(job.id, make_result(), worker_id="worker-one")
    assert not await store.fail(job.id, "late failure", worker_id="worker-one")
    assert await store.get_result(job.id, client_id="alpha") is None
    assert await store.cancel(job.id, client_id="alpha") is None
    event = await store.claim_webhook("webhook-worker", 60)
    assert event is not None
    assert event.payload == {"job_id": job.id, "status": "cancelled"}


async def test_finish_stores_result(store: SqliteJobStore):
    job = job_at(0)
    await store.create(job)
    await store.claim_next_queued()
    await store.finish(job.id, make_result())

    loaded = await store.get(job.id)
    assert loaded is not None and loaded.status == JobStatus.COMPLETED
    result = await store.get_result(job.id)
    assert result is not None
    assert result.language == "sv"
    assert result.segments[0].speaker == "SPEAKER_00"


async def test_fail_stores_error(store: SqliteJobStore):
    job = job_at(0)
    await store.create(job)
    await store.claim_next_queued()
    await store.fail(job.id, "RuntimeError: boom")

    loaded = await store.get(job.id)
    assert loaded is not None
    assert loaded.status == JobStatus.FAILED
    assert loaded.error == "RuntimeError: boom"
    assert await store.get_result(job.id) is None


async def test_expired_lease_is_reclaimed(store: SqliteJobStore):
    await store.create(job_at(0))
    first = await store.claim_next_queued(worker_id="worker-one", lease_for_s=-1)

    second = await store.claim_next_queued(worker_id="worker-two")
    assert first is not None and first.attempt == 1
    assert second is not None and second.attempt == 2
    assert second.lease_owner == "worker-two"


async def test_purge_removes_only_old_terminal_jobs(store: SqliteJobStore):
    old_done = job_at(0)
    old_queued = job_at(1)
    await store.create(old_done)
    await store.create(old_queued)
    await store.claim_next_queued()
    await store.finish(old_done.id, make_result())

    fresh_cutoff = datetime.now(UTC) - timedelta(hours=1)
    future_cutoff = datetime.now(UTC) + timedelta(hours=1)

    assert await store.purge_older_than(fresh_cutoff) == []

    purged = await store.purge_older_than(future_cutoff)
    assert [job.id for job in purged] == [old_done.id]
    assert await store.get(old_done.id) is None
    assert await store.get(old_queued.id) is not None
