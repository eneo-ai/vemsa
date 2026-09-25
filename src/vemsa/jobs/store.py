from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol

from vemsa.jobs.models import (
    FailureKind,
    Job,
    JobOutcome,
    JobStage,
    TranscriptionResult,
    WebhookOutboxEvent,
    WorkerSample,
)


class QueueCapacityError(RuntimeError):
    def __init__(self, scope: Literal["global", "client"], active_jobs: int, limit: int) -> None:
        self.scope = scope
        self.active_jobs = active_jobs
        self.limit = limit
        super().__init__(
            "job queue is full" if scope == "global" else "client has reached its active job limit"
        )


class IdempotencyConflictError(RuntimeError):
    pass


class JobStore(Protocol):
    async def open(self) -> None: ...
    async def close(self) -> None: ...
    async def create(
        self,
        job: Job,
        *,
        max_active: int | None = None,
        max_active_per_client: int | None = None,
    ) -> Job: ...
    async def get(self, job_id: str, *, client_id: str | None = None) -> Job | None: ...
    async def claim_next_queued(
        self,
        *,
        worker_id: str = "legacy-worker",
        lease_for_s: float = 3600.0,
        exclude_ids: Sequence[str] = (),
    ) -> Job | None: ...
    async def release_for_retry(
        self, job_id: str, *, worker_id: str, retry_after_s: float
    ) -> bool: ...
    async def set_audio_path(self, job_id: str, audio_path: str) -> None: ...
    async def set_stage(
        self, job_id: str, stage: JobStage, *, worker_id: str | None = None
    ) -> bool: ...
    async def queue_position(self, job_id: str, *, client_id: str) -> int | None: ...
    async def cancel(
        self,
        job_id: str,
        *,
        client_id: str,
        webhook_url: str | None = None,
    ) -> Job | None: ...
    async def renew_lease(self, job_id: str, worker_id: str, lease_for_s: float) -> bool: ...
    async def finish(
        self,
        job_id: str,
        result: TranscriptionResult,
        *,
        worker_id: str | None = None,
        webhook_url: str | None = None,
        outcome: JobOutcome | None = None,
    ) -> bool: ...
    async def fail(
        self,
        job_id: str,
        error: str,
        *,
        failure_kind: FailureKind = "internal",
        worker_id: str | None = None,
        webhook_url: str | None = None,
        outcome: JobOutcome | None = None,
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
    async def record_worker_sample(self, sample: WorkerSample) -> None: ...
    async def purge_stats_older_than(self, cutoff: datetime) -> int: ...
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
