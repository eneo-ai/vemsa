import json
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from tolka.jobs.models import (
    Job,
    JobRequest,
    JobStage,
    JobStatus,
    TranscriptionResult,
    WebhookOutboxEvent,
)

_MIGRATIONS = (
    """
    CREATE TABLE IF NOT EXISTS tolka_schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        client_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        request_json JSONB NOT NULL,
        audio_path TEXT,
        result_json JSONB,
        error TEXT,
        attempt INTEGER NOT NULL DEFAULT 0,
        lease_owner TEXT,
        lease_expires_at TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_claim
        ON jobs (status, lease_expires_at, created_at);
    CREATE INDEX IF NOT EXISTS idx_jobs_client_status
        ON jobs (client_id, status);
    CREATE TABLE IF NOT EXISTS worker_heartbeats (
        worker_id TEXT PRIMARY KEY,
        updated_at TIMESTAMPTZ NOT NULL
    );
    CREATE TABLE IF NOT EXISTS webhook_outbox (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        url TEXT NOT NULL,
        payload_json JSONB NOT NULL,
        attempt INTEGER NOT NULL DEFAULT 0,
        next_attempt_at TIMESTAMPTZ,
        lease_owner TEXT,
        lease_expires_at TIMESTAMPTZ,
        last_error TEXT,
        created_at TIMESTAMPTZ NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_webhook_outbox_due
        ON webhook_outbox (next_attempt_at, lease_expires_at);
    """,
    """
    ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_status_check;
    ALTER TABLE jobs
        ADD CONSTRAINT jobs_status_check
        CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled'));
    ALTER TABLE jobs ADD COLUMN IF NOT EXISTS stage TEXT NOT NULL DEFAULT 'queued';
    UPDATE jobs SET stage = 'transcribing' WHERE status = 'running' AND stage = 'queued';
    UPDATE jobs SET stage = 'finalizing'
        WHERE status IN ('completed', 'failed') AND stage = 'queued';
    ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_stage_check;
    ALTER TABLE jobs
        ADD CONSTRAINT jobs_stage_check
        CHECK (stage IN ('queued', 'transcribing', 'aligning', 'diarizing', 'finalizing'));
    ALTER TABLE jobs ADD COLUMN IF NOT EXISTS cancellation_requested_at TIMESTAMPTZ;
    """,
)

_JOB_COLUMNS = (
    "id, client_id, status, stage, created_at, updated_at, request_json, audio_path, error, "
    "attempt, lease_owner, lease_expires_at, cancellation_requested_at"
)
_CLAIM_JOB_COLUMNS = (
    "jobs.id, jobs.client_id, jobs.status, jobs.stage, jobs.created_at, jobs.updated_at, "
    "jobs.request_json, jobs.audio_path, jobs.error, jobs.attempt, jobs.lease_owner, "
    "jobs.lease_expires_at, jobs.cancellation_requested_at"
)


def _row_to_job(row: asyncpg.Record) -> Job:
    request_data = row["request_json"]
    if isinstance(request_data, str):
        request_data = json.loads(request_data)
    return Job(
        id=row["id"],
        client_id=row["client_id"],
        status=JobStatus(row["status"]),
        stage=JobStage(row["stage"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        request=JobRequest.model_validate(request_data),
        audio_path=row["audio_path"],
        error=row["error"],
        attempt=row["attempt"],
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        cancellation_requested_at=row["cancellation_requested_at"],
    )


class PostgresJobStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        assert self._pool is not None, "store is not open"
        return self._pool

    async def open(self) -> None:
        self._pool = await asyncpg.create_pool(self._database_url, min_size=1, max_size=10)
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext('tolka_schema_migrations'))"
            )
            await connection.execute(_MIGRATIONS[0])
            await connection.execute(
                "INSERT INTO tolka_schema_migrations (version) VALUES (1)"
                " ON CONFLICT (version) DO NOTHING"
            )
            for version, migration in enumerate(_MIGRATIONS[1:], start=2):
                applied = await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM tolka_schema_migrations WHERE version = $1)",
                    version,
                )
                if applied:
                    continue
                await connection.execute(migration)
                await connection.execute(
                    "INSERT INTO tolka_schema_migrations (version) VALUES ($1)", version
                )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def create(self, job: Job) -> None:
        await self.pool.execute(
            "INSERT INTO jobs"
            " (id, client_id, status, stage, created_at, updated_at, request_json, audio_path)"
            " VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)",
            job.id,
            job.client_id,
            job.status.value,
            job.stage.value,
            job.created_at,
            job.updated_at,
            job.request.model_dump_json(),
            job.audio_path,
        )

    async def get(self, job_id: str, *, client_id: str | None = None) -> Job | None:
        query = f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = $1"
        values: list[Any] = [job_id]
        if client_id is not None:
            query += " AND client_id = $2"
            values.append(client_id)
        row = await self.pool.fetchrow(query, *values)
        return _row_to_job(row) if row else None

    async def claim_next_queued(
        self, *, worker_id: str = "legacy-worker", lease_for_s: float = 3600.0
    ) -> Job | None:
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_for_s)
        row = await self.pool.fetchrow(
            f"""
            WITH candidate AS (
                SELECT id FROM jobs
                WHERE status = $1 OR (status = $2 AND lease_expires_at < $3)
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE jobs
            SET status = $2, updated_at = $3, attempt = attempt + 1,
                lease_owner = $4, lease_expires_at = $5
            FROM candidate
            WHERE jobs.id = candidate.id
            RETURNING {_CLAIM_JOB_COLUMNS}
            """,
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            now,
            worker_id,
            lease_expires_at,
        )
        return _row_to_job(row) if row else None

    async def set_audio_path(self, job_id: str, audio_path: str) -> None:
        await self.pool.execute(
            "UPDATE jobs SET audio_path = $1, updated_at = $2 WHERE id = $3",
            audio_path,
            datetime.now(UTC),
            job_id,
        )

    async def set_stage(
        self, job_id: str, stage: JobStage, *, worker_id: str | None = None
    ) -> bool:
        owner_clause = "" if worker_id is None else " AND lease_owner = $5"
        values: list[Any] = [
            stage.value,
            datetime.now(UTC),
            job_id,
            JobStatus.RUNNING.value,
        ]
        if worker_id is not None:
            values.append(worker_id)
        command = await self.pool.execute(
            "UPDATE jobs SET stage = $1, updated_at = $2"
            " WHERE id = $3 AND status = $4" + owner_clause,
            *values,
        )
        return command == "UPDATE 1"

    async def queue_position(self, job_id: str, *, client_id: str) -> int | None:
        job = await self.get(job_id, client_id=client_id)
        if job is None or job.status != JobStatus.QUEUED:
            return None
        return int(
            await self.pool.fetchval(
                "SELECT COUNT(*) + 1 FROM jobs"
                " WHERE status = $1 AND (created_at < $2 OR (created_at = $2 AND id < $3))",
                JobStatus.QUEUED.value,
                job.created_at,
                job.id,
            )
        )

    async def cancel(
        self,
        job_id: str,
        *,
        client_id: str,
        webhook_url: str | None = None,
    ) -> Job | None:
        now = datetime.now(UTC)
        async with self.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "UPDATE jobs SET status = $1, updated_at = $2,"
                " cancellation_requested_at = $2, lease_owner = NULL, lease_expires_at = NULL"
                " WHERE id = $3 AND client_id = $4 AND status IN ($5, $6)"
                f" RETURNING {_JOB_COLUMNS}",
                JobStatus.CANCELLED.value,
                now,
                job_id,
                client_id,
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
            )
            if row is not None and webhook_url is not None:
                await self._insert_webhook(
                    connection,
                    job_id,
                    webhook_url,
                    {"job_id": job_id, "status": JobStatus.CANCELLED.value},
                )
        return _row_to_job(row) if row else None

    async def renew_lease(self, job_id: str, worker_id: str, lease_for_s: float) -> bool:
        result = await self.pool.execute(
            "UPDATE jobs SET lease_expires_at = $1, updated_at = $2"
            " WHERE id = $3 AND status = $4 AND lease_owner = $5",
            datetime.now(UTC) + timedelta(seconds=lease_for_s),
            datetime.now(UTC),
            job_id,
            JobStatus.RUNNING.value,
            worker_id,
        )
        return result == "UPDATE 1"

    async def finish(
        self,
        job_id: str,
        result: TranscriptionResult,
        *,
        worker_id: str | None = None,
        webhook_url: str | None = None,
    ) -> bool:
        owner_clause = "" if worker_id is None else " AND lease_owner = $6"
        values: list[Any] = [
            JobStatus.COMPLETED.value,
            result.model_dump_json(),
            datetime.now(UTC),
            job_id,
            JobStatus.RUNNING.value,
        ]
        if worker_id is not None:
            values.append(worker_id)
        async with self.pool.acquire() as connection, connection.transaction():
            command = await connection.execute(
                "UPDATE jobs SET status = $1, result_json = $2::jsonb, updated_at = $3,"
                " lease_owner = NULL, lease_expires_at = NULL"
                " WHERE id = $4 AND status = $5" + owner_clause,
                *values,
            )
            if command == "UPDATE 1" and webhook_url is not None:
                await self._insert_webhook(
                    connection,
                    job_id,
                    webhook_url,
                    {
                        "job_id": job_id,
                        "status": JobStatus.COMPLETED.value,
                        "result": result.model_dump(mode="json"),
                    },
                )
        return command == "UPDATE 1"

    async def fail(
        self,
        job_id: str,
        error: str,
        *,
        worker_id: str | None = None,
        webhook_url: str | None = None,
    ) -> bool:
        owner_clause = "" if worker_id is None else " AND lease_owner = $6"
        values: list[Any] = [
            JobStatus.FAILED.value,
            error,
            datetime.now(UTC),
            job_id,
            JobStatus.RUNNING.value,
        ]
        if worker_id is not None:
            values.append(worker_id)
        async with self.pool.acquire() as connection, connection.transaction():
            command = await connection.execute(
                "UPDATE jobs SET status = $1, error = $2, updated_at = $3,"
                " lease_owner = NULL, lease_expires_at = NULL"
                " WHERE id = $4 AND status = $5" + owner_clause,
                *values,
            )
            if command == "UPDATE 1" and webhook_url is not None:
                await self._insert_webhook(
                    connection,
                    job_id,
                    webhook_url,
                    {"job_id": job_id, "status": JobStatus.FAILED.value, "error": error},
                )
        return command == "UPDATE 1"

    async def _insert_webhook(
        self,
        connection: asyncpg.Connection,
        job_id: str,
        url: str,
        payload: dict[str, object],
    ) -> None:
        from uuid import uuid4

        now = datetime.now(UTC)
        await connection.execute(
            "INSERT INTO webhook_outbox"
            " (id, job_id, url, payload_json, next_attempt_at, created_at)"
            " VALUES ($1, $2, $3, $4::jsonb, $5, $5)",
            uuid4().hex,
            job_id,
            url,
            json.dumps(payload),
            now,
        )

    async def get_result(
        self, job_id: str, *, client_id: str | None = None
    ) -> TranscriptionResult | None:
        query = "SELECT result_json FROM jobs WHERE id = $1"
        values: list[Any] = [job_id]
        if client_id is not None:
            query += " AND client_id = $2"
            values.append(client_id)
        value = await self.pool.fetchval(query, *values)
        if value is None:
            return None
        if isinstance(value, str):
            return TranscriptionResult.model_validate_json(value)
        return TranscriptionResult.model_validate(value)

    async def purge_older_than(self, cutoff: datetime) -> list[Job]:
        rows = await self.pool.fetch(
            f"DELETE FROM jobs WHERE status IN ($1, $2, $3) AND updated_at < $4"
            f" RETURNING {_JOB_COLUMNS}",
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            cutoff,
        )
        return [_row_to_job(row) for row in rows]

    async def count_queued(self) -> int:
        return int(
            await self.pool.fetchval(
                "SELECT COUNT(*) FROM jobs WHERE status = $1", JobStatus.QUEUED.value
            )
        )

    async def count_active(self, *, client_id: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM jobs WHERE status IN ($1, $2)"
        values: list[Any] = [JobStatus.QUEUED.value, JobStatus.RUNNING.value]
        if client_id is not None:
            query += " AND client_id = $3"
            values.append(client_id)
        return int(await self.pool.fetchval(query, *values))

    async def ping(self) -> bool:
        try:
            return await self.pool.fetchval("SELECT TRUE") is True
        except (asyncpg.PostgresError, OSError):
            return False

    async def record_worker_heartbeat(self, worker_id: str) -> None:
        await self.pool.execute(
            "INSERT INTO worker_heartbeats (worker_id, updated_at) VALUES ($1, $2)"
            " ON CONFLICT (worker_id) DO UPDATE SET updated_at = excluded.updated_at",
            worker_id,
            datetime.now(UTC),
        )

    async def has_recent_worker(self, stale_after_s: float) -> bool:
        cutoff = datetime.now(UTC) - timedelta(seconds=stale_after_s)
        return bool(
            await self.pool.fetchval(
                "SELECT EXISTS(SELECT 1 FROM worker_heartbeats WHERE updated_at >= $1)", cutoff
            )
        )

    async def claim_webhook(self, worker_id: str, lease_for_s: float) -> WebhookOutboxEvent | None:
        now = datetime.now(UTC)
        row = await self.pool.fetchrow(
            """
            WITH candidate AS (
                SELECT id FROM webhook_outbox
                WHERE next_attempt_at IS NOT NULL AND next_attempt_at <= $1
                  AND (lease_expires_at IS NULL OR lease_expires_at < $1)
                ORDER BY next_attempt_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE webhook_outbox
            SET lease_owner = $2, lease_expires_at = $3
            FROM candidate
            WHERE webhook_outbox.id = candidate.id
            RETURNING webhook_outbox.id, job_id, url, payload_json, attempt
            """,
            now,
            worker_id,
            now + timedelta(seconds=lease_for_s),
        )
        if row is None:
            return None
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return WebhookOutboxEvent(
            id=row["id"],
            job_id=row["job_id"],
            url=row["url"],
            payload=payload,
            attempt=row["attempt"],
        )

    async def mark_webhook_delivered(self, event_id: str, worker_id: str) -> None:
        await self.pool.execute(
            "DELETE FROM webhook_outbox WHERE id = $1 AND lease_owner = $2",
            event_id,
            worker_id,
        )

    async def reschedule_webhook(
        self,
        event_id: str,
        worker_id: str,
        error: str,
        delay_s: float,
        max_attempts: int,
    ) -> bool:
        next_attempt = datetime.now(UTC) + timedelta(seconds=delay_s)
        value = await self.pool.fetchval(
            "UPDATE webhook_outbox SET attempt = attempt + 1,"
            " next_attempt_at = CASE WHEN attempt + 1 >= $1 THEN NULL ELSE $2 END,"
            " lease_owner = NULL, lease_expires_at = NULL, last_error = $3"
            " WHERE id = $4 AND lease_owner = $5 RETURNING next_attempt_at",
            max_attempts,
            next_attempt,
            error[:1000],
            event_id,
            worker_id,
        )
        return value is None
