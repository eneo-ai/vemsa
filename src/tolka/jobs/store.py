from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import aiosqlite

from tolka.jobs.models import (
    Job,
    JobRequest,
    JobStatus,
    TranscriptionResult,
    WebhookOutboxEvent,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL DEFAULT 'legacy',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    audio_path TEXT,
    result_json TEXT,
    error TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status, created_at);
CREATE TABLE IF NOT EXISTS worker_heartbeats (
    worker_id TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS webhook_outbox (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT,
    lease_owner TEXT,
    lease_expires_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_webhook_outbox_due
    ON webhook_outbox (next_attempt_at, lease_expires_at);
"""

_JOB_COLUMNS = (
    "id, client_id, status, created_at, updated_at, request_json, audio_path, error, "
    "attempt, lease_owner, lease_expires_at"
)


class JobStore(Protocol):
    async def open(self) -> None: ...
    async def close(self) -> None: ...
    async def create(self, job: Job) -> None: ...
    async def get(self, job_id: str, *, client_id: str | None = None) -> Job | None: ...
    async def claim_next_queued(
        self, *, worker_id: str = "legacy-worker", lease_for_s: float = 3600.0
    ) -> Job | None: ...
    async def set_audio_path(self, job_id: str, audio_path: str) -> None: ...
    async def renew_lease(self, job_id: str, worker_id: str, lease_for_s: float) -> bool: ...
    async def finish(
        self,
        job_id: str,
        result: TranscriptionResult,
        *,
        worker_id: str | None = None,
        webhook_url: str | None = None,
    ) -> bool: ...
    async def fail(
        self,
        job_id: str,
        error: str,
        *,
        worker_id: str | None = None,
        webhook_url: str | None = None,
    ) -> bool: ...
    async def get_result(
        self, job_id: str, *, client_id: str | None = None
    ) -> TranscriptionResult | None: ...
    async def purge_older_than(self, cutoff: datetime) -> list[Job]: ...
    async def count_queued(self) -> int: ...
    async def count_active(self, *, client_id: str | None = None) -> int: ...
    async def ping(self) -> bool: ...
    async def record_worker_heartbeat(self, worker_id: str) -> None: ...
    async def has_recent_worker(self, stale_after_s: float) -> bool: ...
    async def claim_webhook(
        self, worker_id: str, lease_for_s: float
    ) -> WebhookOutboxEvent | None: ...
    async def mark_webhook_delivered(self, event_id: str, worker_id: str) -> None: ...
    async def reschedule_webhook(
        self,
        event_id: str,
        worker_id: str,
        error: str,
        delay_s: float,
        max_attempts: int,
    ) -> bool: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_job(row: aiosqlite.Row) -> Job:
    return Job(
        id=row["id"],
        client_id=row["client_id"],
        status=JobStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        request=JobRequest.model_validate_json(row["request_json"]),
        audio_path=row["audio_path"],
        error=row["error"],
        attempt=row["attempt"],
        lease_owner=row["lease_owner"],
        lease_expires_at=(
            datetime.fromisoformat(row["lease_expires_at"]) if row["lease_expires_at"] else None
        ),
    )


class SqliteJobStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "store is not open"
        return self._db

    async def open(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA)
        columns = {
            row["name"]
            for row in await (await self._db.execute("PRAGMA table_info(jobs)")).fetchall()
        }
        migrations = {
            "client_id": "ALTER TABLE jobs ADD COLUMN client_id TEXT NOT NULL DEFAULT 'legacy'",
            "attempt": "ALTER TABLE jobs ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0",
            "lease_owner": "ALTER TABLE jobs ADD COLUMN lease_owner TEXT",
            "lease_expires_at": "ALTER TABLE jobs ADD COLUMN lease_expires_at TEXT",
        }
        for column, statement in migrations.items():
            if column not in columns:
                await self._db.execute(statement)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def create(self, job: Job) -> None:
        await self.db.execute(
            "INSERT INTO jobs"
            " (id, client_id, status, created_at, updated_at, request_json, audio_path)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                job.id,
                job.client_id,
                job.status.value,
                job.created_at.isoformat(),
                job.updated_at.isoformat(),
                job.request.model_dump_json(),
                job.audio_path,
            ),
        )
        await self.db.commit()

    async def get(self, job_id: str, *, client_id: str | None = None) -> Job | None:
        query = f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?"
        parameters: tuple[str, ...] = (job_id,)
        if client_id is not None:
            query += " AND client_id = ?"
            parameters = (job_id, client_id)
        cursor = await self.db.execute(query, parameters)
        row = await cursor.fetchone()
        return _row_to_job(row) if row else None

    async def claim_next_queued(
        self, *, worker_id: str = "legacy-worker", lease_for_s: float = 3600.0
    ) -> Job | None:
        now = datetime.now(UTC)
        lease_expires_at = (now + timedelta(seconds=lease_for_s)).isoformat()
        cursor = await self.db.execute(
            "UPDATE jobs SET status = ?, updated_at = ?, attempt = attempt + 1,"
            " lease_owner = ?, lease_expires_at = ?"
            " WHERE id = (SELECT id FROM jobs"
            " WHERE status = ? OR (status = ? AND lease_expires_at < ?)"
            " ORDER BY created_at LIMIT 1)"
            f" RETURNING {_JOB_COLUMNS}",
            (
                JobStatus.RUNNING.value,
                now.isoformat(),
                worker_id,
                lease_expires_at,
                JobStatus.QUEUED.value,
                JobStatus.RUNNING.value,
                now.isoformat(),
            ),
        )
        row = await cursor.fetchone()
        await self.db.commit()
        return _row_to_job(row) if row else None

    async def set_audio_path(self, job_id: str, audio_path: str) -> None:
        await self.db.execute(
            "UPDATE jobs SET audio_path = ?, updated_at = ? WHERE id = ?",
            (audio_path, _now(), job_id),
        )
        await self.db.commit()

    async def renew_lease(self, job_id: str, worker_id: str, lease_for_s: float) -> bool:
        lease_expires_at = (datetime.now(UTC) + timedelta(seconds=lease_for_s)).isoformat()
        cursor = await self.db.execute(
            "UPDATE jobs SET lease_expires_at = ?, updated_at = ?"
            " WHERE id = ? AND status = ? AND lease_owner = ?",
            (lease_expires_at, _now(), job_id, JobStatus.RUNNING.value, worker_id),
        )
        await self.db.commit()
        return cursor.rowcount == 1

    async def finish(
        self,
        job_id: str,
        result: TranscriptionResult,
        *,
        worker_id: str | None = None,
        webhook_url: str | None = None,
    ) -> bool:
        owner_clause = "" if worker_id is None else " AND lease_owner = ?"
        parameters: tuple[str, ...] = (
            JobStatus.COMPLETED.value,
            result.model_dump_json(),
            _now(),
            job_id,
        )
        if worker_id is not None:
            parameters += (worker_id,)
        cursor = await self.db.execute(
            "UPDATE jobs SET status = ?, result_json = ?, updated_at = ?,"
            " lease_owner = NULL, lease_expires_at = NULL WHERE id = ?" + owner_clause,
            parameters,
        )
        if cursor.rowcount == 1 and webhook_url is not None:
            await self._insert_webhook(
                job_id,
                webhook_url,
                {
                    "job_id": job_id,
                    "status": JobStatus.COMPLETED.value,
                    "result": result.model_dump(mode="json"),
                },
            )
        await self.db.commit()
        return cursor.rowcount == 1

    async def fail(
        self,
        job_id: str,
        error: str,
        *,
        worker_id: str | None = None,
        webhook_url: str | None = None,
    ) -> bool:
        owner_clause = "" if worker_id is None else " AND lease_owner = ?"
        parameters = (JobStatus.FAILED.value, error, _now(), job_id)
        if worker_id is not None:
            parameters += (worker_id,)
        cursor = await self.db.execute(
            "UPDATE jobs SET status = ?, error = ?, updated_at = ?,"
            " lease_owner = NULL, lease_expires_at = NULL WHERE id = ?" + owner_clause,
            parameters,
        )
        if cursor.rowcount == 1 and webhook_url is not None:
            await self._insert_webhook(
                job_id,
                webhook_url,
                {"job_id": job_id, "status": JobStatus.FAILED.value, "error": error},
            )
        await self.db.commit()
        return cursor.rowcount == 1

    async def _insert_webhook(self, job_id: str, url: str, payload: dict[str, object]) -> None:
        import json
        from uuid import uuid4

        now = _now()
        await self.db.execute(
            "INSERT INTO webhook_outbox"
            " (id, job_id, url, payload_json, next_attempt_at, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (uuid4().hex, job_id, url, json.dumps(payload), now, now),
        )

    async def get_result(
        self, job_id: str, *, client_id: str | None = None
    ) -> TranscriptionResult | None:
        query = "SELECT result_json FROM jobs WHERE id = ?"
        parameters: tuple[str, ...] = (job_id,)
        if client_id is not None:
            query += " AND client_id = ?"
            parameters = (job_id, client_id)
        cursor = await self.db.execute(query, parameters)
        row = await cursor.fetchone()
        if row is None or row["result_json"] is None:
            return None
        return TranscriptionResult.model_validate_json(row["result_json"])

    async def purge_older_than(self, cutoff: datetime) -> list[Job]:
        cursor = await self.db.execute(
            f"DELETE FROM jobs WHERE status IN (?, ?) AND updated_at < ? RETURNING {_JOB_COLUMNS}",
            (JobStatus.COMPLETED.value, JobStatus.FAILED.value, cutoff.isoformat()),
        )
        rows = await cursor.fetchall()
        await self.db.commit()
        return [_row_to_job(row) for row in rows]

    async def count_queued(self) -> int:
        cursor = await self.db.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = ?", (JobStatus.QUEUED.value,)
        )
        row = await cursor.fetchone()
        assert row is not None
        return row["n"]

    async def count_active(self, *, client_id: str | None = None) -> int:
        query = "SELECT COUNT(*) AS n FROM jobs WHERE status IN (?, ?)"
        parameters: tuple[str, ...] = (JobStatus.QUEUED.value, JobStatus.RUNNING.value)
        if client_id is not None:
            query += " AND client_id = ?"
            parameters += (client_id,)
        cursor = await self.db.execute(query, parameters)
        row = await cursor.fetchone()
        assert row is not None
        return row["n"]

    async def ping(self) -> bool:
        try:
            cursor = await self.db.execute("SELECT 1")
            return await cursor.fetchone() is not None
        except aiosqlite.Error:
            return False

    async def record_worker_heartbeat(self, worker_id: str) -> None:
        await self.db.execute(
            "INSERT INTO worker_heartbeats (worker_id, updated_at) VALUES (?, ?)"
            " ON CONFLICT(worker_id) DO UPDATE SET updated_at = excluded.updated_at",
            (worker_id, _now()),
        )
        await self.db.commit()

    async def has_recent_worker(self, stale_after_s: float) -> bool:
        cutoff = datetime.now(UTC) - timedelta(seconds=stale_after_s)
        cursor = await self.db.execute(
            "SELECT EXISTS(SELECT 1 FROM worker_heartbeats WHERE updated_at >= ?) AS alive",
            (cutoff.isoformat(),),
        )
        row = await cursor.fetchone()
        return bool(row["alive"]) if row else False

    async def claim_webhook(self, worker_id: str, lease_for_s: float) -> WebhookOutboxEvent | None:
        import json

        now = datetime.now(UTC)
        cursor = await self.db.execute(
            "UPDATE webhook_outbox SET lease_owner = ?, lease_expires_at = ?"
            " WHERE id = (SELECT id FROM webhook_outbox"
            " WHERE next_attempt_at IS NOT NULL AND next_attempt_at <= ?"
            " AND (lease_expires_at IS NULL OR lease_expires_at < ?)"
            " ORDER BY next_attempt_at LIMIT 1)"
            " RETURNING id, job_id, url, payload_json, attempt",
            (
                worker_id,
                (now + timedelta(seconds=lease_for_s)).isoformat(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        row = await cursor.fetchone()
        await self.db.commit()
        if row is None:
            return None
        return WebhookOutboxEvent(
            id=row["id"],
            job_id=row["job_id"],
            url=row["url"],
            payload=json.loads(row["payload_json"]),
            attempt=row["attempt"],
        )

    async def mark_webhook_delivered(self, event_id: str, worker_id: str) -> None:
        await self.db.execute(
            "DELETE FROM webhook_outbox WHERE id = ? AND lease_owner = ?",
            (event_id, worker_id),
        )
        await self.db.commit()

    async def reschedule_webhook(
        self,
        event_id: str,
        worker_id: str,
        error: str,
        delay_s: float,
        max_attempts: int,
    ) -> bool:
        next_attempt = datetime.now(UTC) + timedelta(seconds=delay_s)
        cursor = await self.db.execute(
            "UPDATE webhook_outbox SET attempt = attempt + 1,"
            " next_attempt_at = CASE WHEN attempt + 1 >= ? THEN NULL ELSE ? END,"
            " lease_owner = NULL, lease_expires_at = NULL, last_error = ?"
            " WHERE id = ? AND lease_owner = ? RETURNING next_attempt_at",
            (max_attempts, next_attempt.isoformat(), error[:1000], event_id, worker_id),
        )
        row = await cursor.fetchone()
        await self.db.commit()
        return row is not None and row["next_attempt_at"] is None
