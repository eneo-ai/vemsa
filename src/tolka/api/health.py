import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from pydantic import BaseModel

from tolka.deps import AppDeps

logger = logging.getLogger(__name__)


def _service_version() -> str:
    try:
        return version("tolka")
    except PackageNotFoundError:
        return "unknown"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    service_version: str
    database_ready: bool
    worker_ready: bool
    queue_accepting_jobs: bool
    queued_jobs: int | None


async def load_readiness(deps: AppDeps, *, client_id: str | None = None) -> ReadinessResponse:
    database_ready = False
    worker_ready = False
    queued_jobs: int | None = None
    active_jobs: int | None = None
    client_active_jobs: int | None = None
    try:
        if deps.store is not None:
            database_ready = await deps.ready_store.ping()
        if database_ready:
            worker_ready = (
                deps.ready_queue.healthy
                if deps.queue is not None
                else await deps.ready_store.has_recent_worker(deps.settings.worker_stale_s)
            )
            queued_jobs = await deps.ready_store.count_queued()
            active_jobs = await deps.ready_store.count_active()
            if client_id is not None:
                client_active_jobs = await deps.ready_store.count_active(client_id=client_id)
    except Exception:
        logger.exception("readiness probe failed")
        database_ready = False
        worker_ready = False
        queued_jobs = None
        active_jobs = None
        client_active_jobs = None

    queue_accepting_jobs = bool(
        database_ready
        and worker_ready
        and active_jobs is not None
        and active_jobs < deps.settings.max_queued_jobs
        and (
            client_id is None
            or (
                client_active_jobs is not None
                and client_active_jobs < deps.settings.max_queued_jobs_per_client
            )
        )
    )
    ready = database_ready and worker_ready
    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        service_version=_service_version(),
        database_ready=database_ready,
        worker_ready=worker_ready,
        queue_accepting_jobs=queue_accepting_jobs,
        queued_jobs=queued_jobs,
    )
