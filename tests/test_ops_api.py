import json
from datetime import UTC, datetime, timedelta

import asyncpg

from conftest import FakeEngine
from test_api_jobs import api_client
from vemsa import main
from vemsa.config import Settings
from vemsa.jobs.models import JobRequest, WorkerSample, new_job
from vemsa.main import create_app

OPS = ("ops", "pw")


async def seed_stat(
    pool: asyncpg.Pool,
    *,
    job_id: str,
    finished_at: datetime,
    status: str = "completed",
    task: str = "transcribe",
    client_id: str = "alpha",
    processing_s: float | None = 10.0,
    audio_seconds: float | None = 100.0,
    queue_wait_s: float = 5.0,
    attempts: int = 1,
    alignment: str | None = "forced",
    error_class: str | None = None,
    stage_seconds: dict[str, float] | None = None,
) -> None:
    started = finished_at - timedelta(seconds=processing_s or 0)
    created = started - timedelta(seconds=queue_wait_s)
    await pool.execute(
        """
        INSERT INTO job_stats (job_id, client_id, status, task, engine, model, language,
            alignment, device, worker_id, attempts, created_at, started_at, finished_at,
            processing_s, audio_seconds, stage_seconds, error_class)
        VALUES ($1, $2, $3, $4, 'fake', 'm', 'sv', $5, 'cpu', 'w', $6, $7, $8, $9, $10, $11,
                $12::jsonb, $13)
        """,
        job_id,
        client_id,
        status,
        task,
        alignment,
        attempts,
        created,
        started,
        finished_at,
        processing_s,
        audio_seconds,
        json.dumps(stage_seconds) if stage_seconds else None,
        error_class,
    )


async def test_page_and_api_are_open_without_credentials(settings: Settings):
    async with api_client(settings) as (client, _):
        page = await client.get("/ops")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "vemsa ops" in page.text
        overview = await client.get("/ops/api/overview")
        assert overview.status_code == 200
        assert overview.headers["cache-control"] == "no-store"


async def test_basic_auth_is_enforced_when_configured(settings: Settings):
    settings = settings.model_copy(update={"ops_user": "ops", "ops_password": "pw"})
    async with api_client(settings) as (client, _):
        anonymous = await client.get("/ops/api/overview")
        assert anonymous.status_code == 401
        assert anonymous.headers["www-authenticate"].startswith("Basic")
        assert (await client.get("/ops", auth=("ops", "wrong"))).status_code == 401
        assert (await client.get("/ops/api/overview", auth=OPS)).status_code == 200
        # the v1 bearer token is not an ops credential
        bearer = await client.get(
            "/ops/api/overview", headers={"Authorization": "Bearer secret-token"}
        )
        assert bearer.status_code == 401


async def test_dashboard_can_be_disabled(settings: Settings):
    settings = settings.model_copy(update={"ops_enabled": False})
    async with api_client(settings) as (client, _):
        assert (await client.get("/ops")).status_code == 404
        assert (await client.get("/ops/api/overview")).status_code == 404


def test_production_without_credentials_warns_but_mounts(monkeypatch):
    settings = Settings(
        _env_file=None,
        environment="production",
        api_tokens="eneo=secret-token",
        database_url="postgresql://x",
        engine="remote",
    )
    warnings: list[str] = []
    # create_app reconfigures logging, so capture the call rather than the record
    monkeypatch.setattr(main.logger, "warning", lambda message, *args: warnings.append(message))

    app = create_app(settings=settings, engine=FakeEngine())

    assert any("VEMSA_OPS_USER" in message for message in warnings)
    assert app.url_path_for("page") == "/ops"


async def test_overview_reports_queue_and_workers(settings: Settings):
    async with api_client(settings) as (client, app):
        pool = app.state.deps.store.pool
        job = new_job(JobRequest(source_url="https://example.org/a.mp3"), client_id="alpha")
        # created before the in-process worker's claim loop would take it: the
        # worker only claims on notify/poll, and the poll interval is 1 s
        await app.state.deps.store.create(job)
        await pool.execute(
            "INSERT INTO worker_heartbeats (worker_id, updated_at) VALUES ('w-old', $1)",
            datetime.now(UTC) - timedelta(hours=2),
        )
        await app.state.deps.store.record_worker_sample(
            WorkerSample(
                worker_id="w-old",
                sampled_at=datetime.now(UTC) - timedelta(hours=2),
                hostname="oldbox",
                engine="fake",
                device="cuda",
                gpu_name="NVIDIA L4",
                in_flight=0,
                concurrency=2,
                gpu_concurrency=1,
                gpu_util_pct=42.0,
            )
        )

        body = (await client.get("/ops/api/overview")).json()

    assert body["service"]["engine"] == "fake"
    assert body["service"]["worker_concurrency"] == 1
    assert body["readiness"]["database_ready"] is True
    assert body["jobs"]["queued"] + body["jobs"]["running"] + body["jobs"]["completed"] >= 1
    workers = {worker["worker_id"]: worker for worker in body["workers"]}
    assert workers["w-old"]["alive"] is False
    assert workers["w-old"]["hostname"] == "oldbox" and workers["w-old"]["gpu_util_pct"] == 42.0
    assert workers["w-old"]["gpu_name"] == "NVIDIA L4"
    # the in-process worker heartbeats too and is alive
    assert any(worker["alive"] for worker in body["workers"])


async def test_throughput_buckets_are_contiguous_and_gap_filled(settings: Settings):
    now = datetime.now(UTC)
    # buckets are whole UTC hours, so the three recent jobs sit at :26-:28 of the
    # latest hour that is already half over: seeding them "a few minutes ago"
    # splits them across two buckets when the test runs just past the hour
    recent = now.replace(minute=30, second=0, microsecond=0)
    if recent > now:
        recent -= timedelta(hours=1)
    recent_bucket = recent.replace(minute=0)
    async with api_client(settings) as (client, app):
        pool = app.state.deps.store.pool
        await seed_stat(pool, job_id="a", finished_at=recent - timedelta(minutes=2))
        await seed_stat(
            pool, job_id="b", finished_at=recent - timedelta(minutes=3), audio_seconds=50
        )
        await seed_stat(
            pool, job_id="c", finished_at=recent - timedelta(minutes=4), status="failed"
        )
        await seed_stat(pool, job_id="d", finished_at=now - timedelta(hours=3), task="diarize")
        await seed_stat(pool, job_id="old", finished_at=now - timedelta(days=2))

        body = (await client.get("/ops/api/throughput", params={"window": "24h"})).json()

    assert body["window"] == "24h" and body["bucket"] == "PT1H"
    buckets = body["buckets"]
    starts = [datetime.fromisoformat(bucket["start"]) for bucket in buckets]
    assert starts == sorted(starts)
    assert all(
        later - earlier == timedelta(hours=1)
        for earlier, later in zip(starts, starts[1:], strict=False)
    )
    assert starts[0].minute == 0 and 24 <= len(starts) <= 26
    assert body["totals"] == {
        "completed": 3,
        "failed": 1,
        "cancelled": 0,
        "audio_seconds": 250.0,
        "processing_s": 30.0,
    }
    latest = buckets[starts.index(recent_bucket)]
    assert latest["completed"] == 2 and latest["failed"] == 1
    assert latest["audio_seconds"] == 150.0
    assert sum(1 for bucket in buckets if bucket["completed"] == 0 and bucket["failed"] == 0) >= 20
    assert {row["task"]: row["completed"] for row in body["by_task"]} == {
        "transcribe": 2,
        "diarize": 1,
    }


async def test_performance_percentiles_and_stages(settings: Settings):
    now = datetime.now(UTC)
    async with api_client(settings) as (client, app):
        pool = app.state.deps.store.pool
        for index, processing in enumerate((10.0, 20.0, 30.0)):
            await seed_stat(
                pool,
                job_id=f"p{index}",
                finished_at=now - timedelta(minutes=index + 1),
                processing_s=processing,
                stage_seconds={"queued": 1.0, "transcribing": processing - 1.0},
            )
        await seed_stat(
            pool,
            job_id="retry",
            finished_at=now - timedelta(minutes=5),
            attempts=2,
            status="failed",
            error_class="MemoryError",
        )

        body = (await client.get("/ops/api/performance", params={"window": "1h"})).json()

    (task,) = body["tasks"]
    assert task["task"] == "transcribe"
    assert task["completed"] == 3 and task["failed"] == 1 and task["retried"] == 1
    assert task["rtf_p50"] == 0.2 and abs(task["rtf_p95"] - 0.29) < 1e-9
    assert task["queue_wait_p50_s"] == 5.0
    assert task["processing_p50_s"] == 20.0
    stages = {row["stage"]: row for row in body["stages"]}
    assert stages["transcribing"]["p50"] == 19.0 and stages["transcribing"]["total_s"] == 57.0
    assert stages["queued"]["jobs"] == 3


async def test_quality_clients_and_recent_jobs(settings: Settings):
    now = datetime.now(UTC)
    async with api_client(settings) as (client, app):
        pool = app.state.deps.store.pool
        await seed_stat(pool, job_id="f", finished_at=now - timedelta(minutes=1))
        await seed_stat(
            pool, job_id="s", finished_at=now - timedelta(minutes=2), alignment="segment_only"
        )
        await seed_stat(
            pool,
            job_id="e",
            finished_at=now - timedelta(minutes=3),
            status="failed",
            error_class="RuntimeError",
            client_id="beta",
            audio_seconds=None,
        )

        quality = (await client.get("/ops/api/quality", params={"window": "6h"})).json()
        clients = (await client.get("/ops/api/clients", params={"window": "6h"})).json()
        recent = (await client.get("/ops/api/jobs/recent", params={"limit": 2})).json()
        bad_window = await client.get("/ops/api/quality", params={"window": "2y"})
        bad_limit = await client.get("/ops/api/jobs/recent", params={"limit": 0})

    assert {(row["alignment"], row["jobs"]) for row in quality["alignment"]} == {
        ("forced", 1),
        ("segment_only", 1),
    }
    assert quality["errors"] == [{"error_class": "RuntimeError", "task": "transcribe", "jobs": 1}]
    by_client = {row["client_id"]: row for row in clients["clients"]}
    assert by_client["alpha"]["completed"] == 2 and by_client["alpha"]["audio_seconds"] == 200.0
    assert by_client["alpha"]["rtf_p50"] == 0.1
    assert by_client["beta"]["failed"] == 1 and by_client["beta"]["rtf_p50"] is None
    assert [job["job_id"] for job in recent["jobs"]] == ["f", "s"]
    assert recent["jobs"][0]["rtf"] == 0.1 and recent["jobs"][0]["queue_wait_s"] == 5.0
    assert bad_window.status_code == 422 and bad_limit.status_code == 422


async def test_host_series_is_bucketed_per_worker(settings: Settings):
    now = datetime.now(UTC)
    async with api_client(settings) as (client, app):
        store = app.state.deps.store
        for minutes, cpu in ((10, 10.0), (9, 30.0), (3, 50.0)):
            await store.record_worker_sample(
                WorkerSample(
                    worker_id="w1",
                    sampled_at=now - timedelta(minutes=minutes),
                    hostname="box",
                    engine="fake",
                    device="cuda",
                    gpu_name="NVIDIA L4",
                    in_flight=1,
                    concurrency=1,
                    gpu_concurrency=1,
                    cpu_pct=cpu,
                    gpu_util_pct=cpu * 2,
                    gpu_mem_used_bytes=1_000,
                    gpu_mem_total_bytes=10_000,
                )
            )

        body = (await client.get("/ops/api/host", params={"window": "6h"})).json()

    assert body["bucket"] == "PT1M"
    workers = {worker["worker_id"]: worker for worker in body["workers"]}
    w1 = workers["w1"]
    assert w1["hostname"] == "box" and w1["gpu_name"] == "NVIDIA L4"
    assert [point["cpu_pct"] for point in w1["points"]] == [10.0, 30.0, 50.0]
    assert w1["points"][-1]["gpu_util_pct"] == 100.0
    assert w1["points"][-1]["gpu_mem_total_bytes"] == 10_000.0


async def test_ops_bodies_never_carry_secrets_or_payloads(settings: Settings):
    settings = settings.model_copy(
        update={"ops_user": "ops", "ops_password": "pw-secret", "hf_token": "hf-secret"}
    )
    now = datetime.now(UTC)
    async with api_client(settings) as (client, app):
        store = app.state.deps.store
        job = new_job(JobRequest(source_url="https://example.org/private.mp3"), client_id="a")
        await store.create(job)
        await seed_stat(store.pool, job_id="x", finished_at=now - timedelta(minutes=1))
        bodies = [
            (await client.get(path, auth=OPS)).text
            for path in (
                "/ops/api/overview",
                "/ops/api/throughput",
                "/ops/api/performance",
                "/ops/api/quality",
                "/ops/api/clients",
                "/ops/api/host",
                "/ops/api/jobs/recent",
            )
        ]

    for body in bodies:
        for forbidden in ("secret-token", "pw-secret", "hf-secret", "private.mp3", "example.org"):
            assert forbidden not in body
        for key in ("source_url", "audio_path", "result", "request", 'error"'):
            assert key not in body
