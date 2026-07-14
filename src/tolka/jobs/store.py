from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import aiosqlite

from tolka.jobs.models import Job, JobRequest, JobStatus, TranscriptionResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    audio_path TEXT,
    result_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status, created_at);
"""

_JOB_COLUMNS = "id, status, created_at, updated_at, request_json, audio_path, error"


class JobStore(Protocol):
    async def create(self, job: Job) -> None: ...
    async def get(self, job_id: str) -> Job | None: ...
    async def claim_next_queued(self) -> Job | None: ...
    async def set_audio_path(self, job_id: str, audio_path: str) -> None: ...
    async def finish(self, job_id: str, result: TranscriptionResult) -> None: ...
    async def fail(self, job_id: str, error: str) -> None: ...
    async def get_result(self, job_id: str) -> TranscriptionResult | None: ...
    async def requeue_stuck(self) -> int: ...
    async def purge_older_than(self, cutoff: datetime) -> list[Job]: ...
    async def count_queued(self) -> int: ...


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_job(row: aiosqlite.Row) -> Job:
    return Job(
        id=row["id"],
        status=JobStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        request=JobRequest.model_validate_json(row["request_json"]),
        audio_path=row["audio_path"],
        error=row["error"],
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
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def create(self, job: Job) -> None:
        await self.db.execute(
            "INSERT INTO jobs (id, status, created_at, updated_at, request_json, audio_path)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                job.id,
                job.status.value,
                job.created_at.isoformat(),
                job.updated_at.isoformat(),
                job.request.model_dump_json(),
                job.audio_path,
            ),
        )
        await self.db.commit()

    async def get(self, job_id: str) -> Job | None:
        cursor = await self.db.execute(f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        return _row_to_job(row) if row else None

    async def claim_next_queued(self) -> Job | None:
        cursor = await self.db.execute(
            "UPDATE jobs SET status = ?, updated_at = ?"
            " WHERE id = (SELECT id FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1)"
            f" RETURNING {_JOB_COLUMNS}",
            (JobStatus.RUNNING.value, _now(), JobStatus.QUEUED.value),
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

    async def finish(self, job_id: str, result: TranscriptionResult) -> None:
        await self.db.execute(
            "UPDATE jobs SET status = ?, result_json = ?, updated_at = ? WHERE id = ?",
            (JobStatus.COMPLETED.value, result.model_dump_json(), _now(), job_id),
        )
        await self.db.commit()

    async def fail(self, job_id: str, error: str) -> None:
        await self.db.execute(
            "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (JobStatus.FAILED.value, error, _now(), job_id),
        )
        await self.db.commit()

    async def get_result(self, job_id: str) -> TranscriptionResult | None:
        cursor = await self.db.execute("SELECT result_json FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        if row is None or row["result_json"] is None:
            return None
        return TranscriptionResult.model_validate_json(row["result_json"])

    async def requeue_stuck(self) -> int:
        cursor = await self.db.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE status = ?",
            (JobStatus.QUEUED.value, _now(), JobStatus.RUNNING.value),
        )
        await self.db.commit()
        return cursor.rowcount

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
