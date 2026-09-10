"""Read-only analytics for the operator dashboard, over the store's asyncpg pool.

Kept out of the JobStore protocol on purpose: that protocol is the job
lifecycle the worker depends on, while these are presentation queries the API
alone runs. Nothing here selects `result_json`, `request_json` (it carries the
source URL), `audio_path`, or `error`."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
from pydantic import BaseModel

# window -> (span, throughput bucket, host-sample bucket)
WINDOWS: dict[str, tuple[timedelta, timedelta, timedelta]] = {
    "1h": (timedelta(hours=1), timedelta(minutes=5), timedelta(seconds=30)),
    "6h": (timedelta(hours=6), timedelta(minutes=15), timedelta(minutes=1)),
    "24h": (timedelta(hours=24), timedelta(hours=1), timedelta(minutes=5)),
    "7d": (timedelta(days=7), timedelta(hours=6), timedelta(minutes=30)),
    "30d": (timedelta(days=30), timedelta(days=1), timedelta(hours=2)),
}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class Range(BaseModel):
    """A query window aligned to whole buckets; buckets are UTC-anchored."""

    window: str
    since: datetime
    until: datetime
    bucket: timedelta

    @classmethod
    def for_window(cls, window: str, *, bucket: timedelta, now: datetime | None = None) -> "Range":
        span = WINDOWS[window][0]
        until = now or datetime.now(UTC)
        since = until - span
        since -= (since - _EPOCH) % bucket
        return cls(window=window, since=since, until=until, bucket=bucket)

    def starts(self) -> list[datetime]:
        starts: list[datetime] = []
        start = self.since
        while start < self.until:
            starts.append(start)
            start += self.bucket
        return starts


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _percentiles(value: Any) -> tuple[float | None, float | None]:
    if not value:
        return None, None
    return _float(value[0]), _float(value[1])


async def status_counts(pool: asyncpg.Pool) -> dict[str, int]:
    rows = await pool.fetch("SELECT status, COUNT(*) AS jobs FROM jobs GROUP BY status")
    counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0}
    counts.update({row["status"]: int(row["jobs"]) for row in rows})
    return counts


async def oldest_queued_age_s(pool: asyncpg.Pool, *, now: datetime) -> float | None:
    """Passing `now` from Python keeps the age on the clock that wrote created_at."""
    value = await pool.fetchval(
        "SELECT EXTRACT(EPOCH FROM ($1::timestamptz - MIN(created_at)))::float8"
        " FROM jobs WHERE status = 'queued'",
        now,
    )
    return _float(value)


async def running_jobs(pool: asyncpg.Pool, *, now: datetime) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT id, client_id, request_json->>'task' AS task, stage, attempt, started_at,
               lease_owner, lease_expires_at,
               EXTRACT(EPOCH FROM ($1::timestamptz - COALESCE(started_at, created_at)))::float8
                   AS elapsed_s
        FROM jobs WHERE status = 'running' ORDER BY started_at
        """,
        now,
    )
    return [dict(row) for row in rows]


async def workers(pool: asyncpg.Pool, *, now: datetime, stale_after_s: float) -> list[dict]:
    """Every worker heard from in the last day, with its latest host sample."""
    rows = await pool.fetch(
        """
        SELECT h.worker_id, h.updated_at AS last_heartbeat,
               EXTRACT(EPOCH FROM ($1::timestamptz - h.updated_at))::float8 AS heartbeat_age_s,
               s.sampled_at, s.hostname, s.engine, s.device, s.gpu_name, s.in_flight,
               s.concurrency, s.gpu_concurrency, s.cpu_pct, s.load1, s.mem_used_bytes,
               s.mem_total_bytes, s.disk_used_bytes, s.disk_total_bytes, s.gpu_util_pct,
               s.gpu_mem_used_bytes, s.gpu_mem_total_bytes, s.gpu_temp_c
        FROM worker_heartbeats h
        LEFT JOIN LATERAL (
            SELECT * FROM worker_samples w WHERE w.worker_id = h.worker_id
            ORDER BY sampled_at DESC LIMIT 1
        ) s ON TRUE
        WHERE h.updated_at >= $1::timestamptz - interval '1 day'
        ORDER BY h.updated_at DESC
        """,
        now,
    )
    return [
        {**dict(row), "alive": (row["heartbeat_age_s"] or 0.0) <= stale_after_s} for row in rows
    ]


async def throughput(pool: asyncpg.Pool, span: Range) -> dict[str, Any]:
    rows = await pool.fetch(
        """
        SELECT date_bin($3::interval, finished_at, $1::timestamptz) AS bucket, status,
               COUNT(*) AS jobs,
               COALESCE(SUM(audio_seconds), 0)::float8 AS audio_seconds,
               COALESCE(SUM(processing_s), 0)::float8 AS processing_s
        FROM job_stats
        WHERE finished_at >= $1 AND finished_at < $2
        GROUP BY 1, 2
        """,
        span.since,
        span.until,
        span.bucket,
    )
    buckets = {
        start: {
            "start": start,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "audio_seconds": 0.0,
            "processing_s": 0.0,
        }
        for start in span.starts()
    }
    for row in rows:
        bucket = buckets.get(row["bucket"])
        if bucket is None:
            continue
        bucket[row["status"]] += int(row["jobs"])
        if row["status"] == "completed":
            bucket["audio_seconds"] += row["audio_seconds"]
            bucket["processing_s"] += row["processing_s"]
    by_task = await pool.fetch(
        """
        SELECT task,
               COUNT(*) FILTER (WHERE status = 'completed') AS completed,
               COUNT(*) FILTER (WHERE status = 'failed') AS failed,
               COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled,
               COALESCE(SUM(audio_seconds) FILTER (WHERE status = 'completed'), 0)::float8
                   AS audio_seconds,
               COALESCE(SUM(processing_s) FILTER (WHERE status = 'completed'), 0)::float8
                   AS processing_s
        FROM job_stats
        WHERE finished_at >= $1 AND finished_at < $2
        GROUP BY task ORDER BY task
        """,
        span.since,
        span.until,
    )
    totals = {
        key: sum(bucket[key] for bucket in buckets.values())
        for key in ("completed", "failed", "cancelled", "audio_seconds", "processing_s")
    }
    return {
        "buckets": list(buckets.values()),
        "by_task": [dict(row) for row in by_task],
        "totals": totals,
    }


async def performance(pool: asyncpg.Pool, span: Range) -> dict[str, Any]:
    rows = await pool.fetch(
        """
        SELECT task,
               COUNT(*) FILTER (WHERE status = 'completed') AS completed,
               COUNT(*) FILTER (WHERE status = 'failed') AS failed,
               COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled,
               COUNT(*) FILTER (WHERE attempts > 1) AS retried,
               percentile_cont(ARRAY[0.5, 0.95]) WITHIN GROUP
                   (ORDER BY processing_s / audio_seconds)
                   FILTER (WHERE status = 'completed' AND audio_seconds > 0
                           AND processing_s IS NOT NULL) AS rtf,
               percentile_cont(ARRAY[0.5, 0.95]) WITHIN GROUP
                   (ORDER BY EXTRACT(EPOCH FROM (started_at - created_at)))
                   FILTER (WHERE started_at IS NOT NULL) AS queue_wait_s,
               percentile_cont(ARRAY[0.5, 0.95]) WITHIN GROUP (ORDER BY processing_s)
                   FILTER (WHERE status = 'completed' AND processing_s IS NOT NULL)
                   AS processing_s
        FROM job_stats
        WHERE finished_at >= $1 AND finished_at < $2
        GROUP BY task ORDER BY task
        """,
        span.since,
        span.until,
    )
    tasks = []
    for row in rows:
        rtf_p50, rtf_p95 = _percentiles(row["rtf"])
        wait_p50, wait_p95 = _percentiles(row["queue_wait_s"])
        proc_p50, proc_p95 = _percentiles(row["processing_s"])
        tasks.append(
            {
                "task": row["task"],
                "completed": int(row["completed"]),
                "failed": int(row["failed"]),
                "cancelled": int(row["cancelled"]),
                "retried": int(row["retried"]),
                "rtf_p50": rtf_p50,
                "rtf_p95": rtf_p95,
                "queue_wait_p50_s": wait_p50,
                "queue_wait_p95_s": wait_p95,
                "processing_p50_s": proc_p50,
                "processing_p95_s": proc_p95,
            }
        )
    stages = await pool.fetch(
        """
        SELECT stage,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY seconds)::float8 AS p50,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY seconds)::float8 AS p95,
               SUM(seconds)::float8 AS total_s, COUNT(*) AS jobs
        FROM job_stats, jsonb_each_text(stage_seconds) AS s(stage, value),
             LATERAL (SELECT value::float8 AS seconds) v
        WHERE finished_at >= $1 AND finished_at < $2 AND status = 'completed'
        GROUP BY stage ORDER BY total_s DESC
        """,
        span.since,
        span.until,
    )
    return {"tasks": tasks, "stages": [dict(row) for row in stages]}


async def quality(pool: asyncpg.Pool, span: Range) -> dict[str, Any]:
    alignment = await pool.fetch(
        """
        SELECT task, COALESCE(alignment, 'none') AS alignment, COUNT(*) AS jobs
        FROM job_stats
        WHERE finished_at >= $1 AND finished_at < $2 AND status = 'completed'
        GROUP BY 1, 2 ORDER BY 1, 3 DESC
        """,
        span.since,
        span.until,
    )
    errors = await pool.fetch(
        """
        SELECT COALESCE(error_class, 'unknown') AS error_class, task, COUNT(*) AS jobs
        FROM job_stats
        WHERE finished_at >= $1 AND finished_at < $2 AND status = 'failed'
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 20
        """,
        span.since,
        span.until,
    )
    return {"alignment": [dict(row) for row in alignment], "errors": [dict(row) for row in errors]}


async def clients(pool: asyncpg.Pool, span: Range) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT client_id,
               COUNT(*) FILTER (WHERE status = 'completed') AS completed,
               COUNT(*) FILTER (WHERE status = 'failed') AS failed,
               COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled,
               COALESCE(SUM(audio_seconds) FILTER (WHERE status = 'completed'), 0)::float8
                   AS audio_seconds,
               COALESCE(SUM(processing_s) FILTER (WHERE status = 'completed'), 0)::float8
                   AS processing_s,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY processing_s / audio_seconds)
                   FILTER (WHERE status = 'completed' AND audio_seconds > 0
                           AND processing_s IS NOT NULL)::float8 AS rtf_p50
        FROM job_stats
        WHERE finished_at >= $1 AND finished_at < $2
        GROUP BY client_id ORDER BY audio_seconds DESC
        """,
        span.since,
        span.until,
    )
    return [dict(row) for row in rows]


async def host_series(pool: asyncpg.Pool, span: Range) -> list[dict[str, Any]]:
    """Per-worker host/GPU series, averaged into `span.bucket` bins."""
    rows = await pool.fetch(
        """
        SELECT worker_id, date_bin($3::interval, sampled_at, $1::timestamptz) AS bucket,
               MAX(hostname) AS hostname, MAX(gpu_name) AS gpu_name, MAX(device) AS device,
               AVG(cpu_pct)::float8 AS cpu_pct, AVG(load1)::float8 AS load1,
               AVG(mem_used_bytes)::float8 AS mem_used_bytes,
               MAX(mem_total_bytes)::float8 AS mem_total_bytes,
               AVG(disk_used_bytes)::float8 AS disk_used_bytes,
               MAX(disk_total_bytes)::float8 AS disk_total_bytes,
               AVG(gpu_util_pct)::float8 AS gpu_util_pct,
               AVG(gpu_mem_used_bytes)::float8 AS gpu_mem_used_bytes,
               MAX(gpu_mem_total_bytes)::float8 AS gpu_mem_total_bytes,
               AVG(gpu_temp_c)::float8 AS gpu_temp_c,
               AVG(in_flight)::float8 AS in_flight, MAX(concurrency) AS concurrency
        FROM worker_samples
        WHERE sampled_at >= $1 AND sampled_at < $2
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
        span.since,
        span.until,
        span.bucket,
    )
    series: dict[str, dict[str, Any]] = {}
    for row in rows:
        worker = series.setdefault(
            row["worker_id"],
            {
                "worker_id": row["worker_id"],
                "hostname": row["hostname"],
                "device": row["device"],
                "gpu_name": row["gpu_name"],
                "points": [],
            },
        )
        worker["hostname"] = row["hostname"] or worker["hostname"]
        worker["gpu_name"] = row["gpu_name"] or worker["gpu_name"]
        worker["points"].append(
            {
                "t": row["bucket"],
                "cpu_pct": row["cpu_pct"],
                "load1": row["load1"],
                "mem_used_bytes": row["mem_used_bytes"],
                "mem_total_bytes": row["mem_total_bytes"],
                "disk_used_bytes": row["disk_used_bytes"],
                "disk_total_bytes": row["disk_total_bytes"],
                "gpu_util_pct": row["gpu_util_pct"],
                "gpu_mem_used_bytes": row["gpu_mem_used_bytes"],
                "gpu_mem_total_bytes": row["gpu_mem_total_bytes"],
                "gpu_temp_c": row["gpu_temp_c"],
                "in_flight": row["in_flight"],
                "concurrency": row["concurrency"],
            }
        )
    return list(series.values())


async def recent_jobs(pool: asyncpg.Pool, *, limit: int) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT job_id, client_id, task, status, engine, model, language, alignment, device,
               attempts, created_at, started_at, finished_at, processing_s, audio_seconds,
               stage_seconds, error_class,
               EXTRACT(EPOCH FROM (started_at - created_at))::float8 AS queue_wait_s,
               CASE WHEN audio_seconds > 0 THEN processing_s / audio_seconds END AS rtf
        FROM job_stats ORDER BY finished_at DESC LIMIT $1
        """,
        limit,
    )
    jobs = []
    for row in rows:
        job = dict(row)
        stages = job.pop("stage_seconds")
        job["stage_seconds"] = json.loads(stages) if isinstance(stages, str) else stages
        jobs.append(job)
    return jobs
