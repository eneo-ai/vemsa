import asyncio
import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from conftest import make_result
from vemsa.jobs.models import (
    JobOutcome,
    JobRequest,
    JobStage,
    JobStatus,
    WorkerSample,
    new_job,
)
from vemsa.jobs.postgres_store import PostgresJobStore


@pytest.fixture
async def postgres_store():
    database_url = os.getenv("VEMSA_TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("VEMSA_TEST_POSTGRES_URL is not configured")
    store = PostgresJobStore(database_url)
    await store.open()
    await store.pool.execute(
        "TRUNCATE webhook_outbox, jobs, worker_heartbeats, job_stats, worker_samples"
    )
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.parametrize("scope", ["global", "client"])
async def test_concurrent_admission_enforces_active_cap(postgres_store: PostgresJobStore, scope):
    first = new_job(JobRequest(source_url="https://example.org/running.mp3"), client_id="alpha")
    await postgres_store.create(first)
    assert await postgres_store.claim_next_queued(worker_id="worker-a") is not None
    jobs = [
        new_job(
            JobRequest(source_url=f"https://example.org/{index}.mp3"),
            client_id="alpha" if scope == "client" else f"client-{index}",
        )
        for index in range(12)
    ]
    other_store = PostgresJobStore(postgres_store._database_url)
    await other_store.open()
    try:
        results = await asyncio.gather(
            *(
                (postgres_store if index % 2 else other_store).create(
                    job, max_active=3 if scope == "global" else 20, max_active_per_client=3
                )
                for index, job in enumerate(jobs)
            ),
            return_exceptions=True,
        )
    finally:
        await other_store.close()
    admitted = [result for result in results if not isinstance(result, BaseException)]
    assert len(admitted) == 2
    assert await postgres_store.count_active() == 3
    from vemsa.jobs.store import QueueCapacityError

    rejected = [result for result in results if isinstance(result, BaseException)]
    assert all(
        isinstance(result, QueueCapacityError) and result.scope == scope for result in rejected
    )
    expected = sorted(admitted, key=lambda job: job.created_at)
    for job in expected:
        claimed = await postgres_store.claim_next_queued(worker_id="worker-b")
        assert claimed is not None and claimed.id == job.id


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


async def test_claim_records_first_start_only(postgres_store: PostgresJobStore):
    job = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    await postgres_store.create(job)
    claimed = await postgres_store.claim_next_queued(worker_id="worker-a")
    assert claimed is not None and claimed.started_at is not None
    first_start = claimed.started_at

    assert await postgres_store.release_for_retry(job.id, worker_id="worker-a", retry_after_s=0)
    retried = await postgres_store.claim_next_queued(worker_id="worker-b")
    assert retried is not None and retried.attempt == 2
    # queue wait measures time to the first pickup, not the last
    assert retried.started_at == first_start


async def test_finish_records_job_stats_exactly_once(postgres_store: PostgresJobStore):
    job = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    await postgres_store.create(job)
    assert await postgres_store.claim_next_queued(worker_id="worker-a") is not None
    outcome = JobOutcome(
        engine="fake",
        model="KBLab/kb-whisper-large",
        alignment="forced",
        device="cpu",
        processing_s=1.5,
        audio_seconds=2.6,
        stage_seconds={"queued": 0.1, "transcribing": 1.0},
    )
    assert await postgres_store.finish(job.id, make_result(), worker_id="worker-a", outcome=outcome)
    # a second completion is rejected and must not add a row
    assert not await postgres_store.finish(job.id, make_result(), worker_id="worker-a")

    rows = await postgres_store.pool.fetch("SELECT * FROM job_stats WHERE job_id = $1", job.id)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed" and row["task"] == "transcribe"
    assert row["client_id"] == "alpha" and row["worker_id"] == "worker-a"
    assert row["engine"] == "fake" and row["alignment"] == "forced" and row["device"] == "cpu"
    assert row["attempts"] == 1
    assert row["started_at"] is not None and row["finished_at"] >= row["started_at"]
    assert row["processing_s"] == 1.5 and row["audio_seconds"] == 2.6
    assert json.loads(row["stage_seconds"]) == {"queued": 0.1, "transcribing": 1.0}
    assert row["error_class"] is None


async def test_fail_records_error_class(postgres_store: PostgresJobStore):
    job = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    await postgres_store.create(job)
    assert await postgres_store.claim_next_queued(worker_id="worker-a") is not None
    outcome = JobOutcome(engine="fake", processing_s=0.2, error_class="RuntimeError")
    assert await postgres_store.fail(job.id, "boom", worker_id="worker-a", outcome=outcome)

    row = await postgres_store.pool.fetchrow("SELECT * FROM job_stats WHERE job_id = $1", job.id)
    assert row is not None
    assert row["status"] == "failed" and row["error_class"] == "RuntimeError"
    assert row["audio_seconds"] is None and row["stage_seconds"] is None


async def test_cancel_records_stats_for_queued_and_running_jobs(
    postgres_store: PostgresJobStore,
):
    queued = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
    running = new_job(JobRequest(source_url="https://example.org/b.mp3"), client_id="alpha")
    await postgres_store.create(queued)
    await postgres_store.create(running)
    claimed = await postgres_store.claim_next_queued(worker_id="worker-a")
    assert claimed is not None and claimed.id == queued.id
    # the "queued" job is the second one now
    assert await postgres_store.cancel(running.id, client_id="alpha") is not None
    assert await postgres_store.cancel(queued.id, client_id="alpha") is not None
    # a late completion from the worker adds nothing
    assert not await postgres_store.finish(queued.id, make_result(), worker_id="worker-a")

    rows = {
        row["job_id"]: row
        for row in await postgres_store.pool.fetch("SELECT * FROM job_stats ORDER BY job_id")
    }
    assert set(rows) == {queued.id, running.id}
    assert rows[running.id]["status"] == "cancelled" and rows[running.id]["started_at"] is None
    assert rows[queued.id]["status"] == "cancelled" and rows[queued.id]["started_at"] is not None
    assert rows[queued.id]["worker_id"] is None and rows[queued.id]["attempts"] == 1


async def test_purge_stats_older_than_keeps_recent_rows(postgres_store: PostgresJobStore):
    now = datetime.now(UTC)
    old, recent = now - timedelta(days=100), now - timedelta(days=1)
    for job_id, finished_at in (("old", old), ("recent", recent)):
        await postgres_store.pool.execute(
            "INSERT INTO job_stats (job_id, client_id, status, task, attempts, created_at,"
            " finished_at) VALUES ($1, 'alpha', 'completed', 'transcribe', 1, $2, $2)",
            job_id,
            finished_at,
        )
    for worker_id, sampled_at in (("old-worker", old), ("live-worker", recent)):
        await postgres_store.record_worker_sample(
            WorkerSample(
                worker_id=worker_id,
                sampled_at=sampled_at,
                hostname="box",
                engine="fake",
                in_flight=0,
                concurrency=1,
                gpu_concurrency=1,
            )
        )
        await postgres_store.pool.execute(
            "INSERT INTO worker_heartbeats (worker_id, updated_at) VALUES ($1, $2)",
            worker_id,
            sampled_at,
        )

    removed = await postgres_store.purge_stats_older_than(now - timedelta(days=90))

    assert removed == 3
    assert await postgres_store.pool.fetchval("SELECT job_id FROM job_stats") == "recent"
    assert await postgres_store.pool.fetchval("SELECT worker_id FROM worker_samples") == (
        "live-worker"
    )
    assert await postgres_store.pool.fetchval("SELECT worker_id FROM worker_heartbeats") == (
        "live-worker"
    )
