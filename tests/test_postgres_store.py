import asyncio
import os

import pytest

from conftest import make_result
from vemsa.jobs.models import JobRequest, JobStage, JobStatus, new_job
from vemsa.jobs.postgres_store import PostgresJobStore


@pytest.fixture
async def postgres_store():
    database_url = os.getenv("VEMSA_TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("VEMSA_TEST_POSTGRES_URL is not configured")
    store = PostgresJobStore(database_url)
    await store.open()
    await store.pool.execute("TRUNCATE webhook_outbox, jobs, worker_heartbeats")
    try:
        yield store
    finally:
        await store.close()


async def test_postgres_leases_ownership_and_outbox(postgres_store: PostgresJobStore):
    first = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    second = new_job(JobRequest(source_url="https://example.org/b.mp3"), client_id="beta")
    await postgres_store.create(first)
    await postgres_store.create(second)

    claimed = await asyncio.gather(
        postgres_store.claim_next_queued(worker_id="worker-a", lease_for_s=60),
        postgres_store.claim_next_queued(worker_id="worker-b", lease_for_s=60),
    )
    assert {job.id for job in claimed if job is not None} == {first.id, second.id}

    alpha_job = next(job for job in claimed if job is not None and job.client_id == "alpha")
    assert await postgres_store.renew_lease(alpha_job.id, alpha_job.lease_owner or "", 60)
    assert await postgres_store.finish(
        alpha_job.id,
        make_result(),
        worker_id=alpha_job.lease_owner,
        webhook_url="https://hooks.example.org/done",
    )
    assert await postgres_store.get(alpha_job.id, client_id="beta") is None
    assert await postgres_store.get_result(alpha_job.id, client_id="alpha") is not None

    event = await postgres_store.claim_webhook("webhook-worker", 60)
    assert event is not None and event.job_id == alpha_job.id
    assert event.payload["status"] == "completed"
    await postgres_store.mark_webhook_delivered(event.id, "webhook-worker")
    assert await postgres_store.claim_webhook("webhook-worker", 60) is None


async def test_postgres_cancellation_rejects_late_completion(postgres_store: PostgresJobStore):
    job = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    await postgres_store.create(job)
    claimed = await postgres_store.claim_next_queued(worker_id="worker-a", lease_for_s=60)
    assert claimed is not None
    assert await postgres_store.set_stage(job.id, JobStage.TRANSCRIBING, worker_id="worker-a")

    cancelled = await postgres_store.cancel(job.id, client_id="alpha")
    assert cancelled is not None and cancelled.status == JobStatus.CANCELLED
    assert not await postgres_store.finish(job.id, make_result(), worker_id="worker-a")
    assert await postgres_store.get_result(job.id, client_id="alpha") is None


async def test_claim_skips_excluded_ids(postgres_store: PostgresJobStore):
    job = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    await postgres_store.create(job)
    claimed = await postgres_store.claim_next_queued(worker_id="worker-a", lease_for_s=-1)
    assert claimed is not None and claimed.status == JobStatus.RUNNING

    # the lease has lapsed, but the worker still runs the job: not claimable by it
    assert (
        await postgres_store.claim_next_queued(worker_id="worker-a", exclude_ids=[job.id]) is None
    )
    reclaimed = await postgres_store.claim_next_queued(worker_id="worker-b")
    assert reclaimed is not None and reclaimed.id == job.id and reclaimed.attempt == 2


async def test_release_for_retry_requeues_with_cooldown(postgres_store: PostgresJobStore):
    job = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    await postgres_store.create(job)
    assert await postgres_store.claim_next_queued(worker_id="worker-a") is not None
    assert await postgres_store.set_stage(job.id, JobStage.TRANSCRIBING, worker_id="worker-a")

    assert await postgres_store.release_for_retry(job.id, worker_id="worker-a", retry_after_s=60)
    released = await postgres_store.get(job.id)
    assert released is not None
    assert released.status == JobStatus.QUEUED and released.stage == JobStage.QUEUED
    assert released.lease_owner is None and released.lease_expires_at is None
    assert released.attempt == 1
    # cooling down: nobody may claim it yet
    assert await postgres_store.claim_next_queued(worker_id="worker-b") is None

    await postgres_store.pool.execute(
        "UPDATE jobs SET retry_after = now() - interval '1 second' WHERE id = $1", job.id
    )
    retried = await postgres_store.claim_next_queued(worker_id="worker-b")
    assert retried is not None and retried.id == job.id and retried.attempt == 2
    assert (
        await postgres_store.pool.fetchval("SELECT retry_after FROM jobs WHERE id = $1", job.id)
        is None
    )


async def test_release_for_retry_without_cooldown_is_claimable_at_once(
    postgres_store: PostgresJobStore,
):
    job = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    await postgres_store.create(job)
    assert await postgres_store.claim_next_queued(worker_id="worker-a") is not None
    assert await postgres_store.release_for_retry(job.id, worker_id="worker-a", retry_after_s=0)
    retried = await postgres_store.claim_next_queued(worker_id="worker-a")
    assert retried is not None and retried.id == job.id and retried.attempt == 2


async def test_release_for_retry_rejects_foreign_owner_and_non_running(
    postgres_store: PostgresJobStore,
):
    job = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    await postgres_store.create(job)
    assert await postgres_store.claim_next_queued(worker_id="worker-a") is not None

    assert not await postgres_store.release_for_retry(job.id, worker_id="worker-b", retry_after_s=0)
    assert await postgres_store.finish(job.id, make_result(), worker_id="worker-a")
    assert not await postgres_store.release_for_retry(job.id, worker_id="worker-a", retry_after_s=0)
    completed = await postgres_store.get(job.id)
    assert completed is not None and completed.status == JobStatus.COMPLETED
