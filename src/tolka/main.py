import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Response, status
from fastmcp.utilities.lifespan import combine_lifespans
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from tolka.api.auth import require_token
from tolka.api.jobs import router as jobs_router
from tolka.config import Settings
from tolka.deps import AppDeps
from tolka.jobs.queue import JobQueue
from tolka.jobs.store_factory import open_job_store
from tolka.mcp.server import build_mcp
from tolka.observability import QUEUE_DEPTH, configure_logging, request_observability_middleware
from tolka.pipeline.base import TranscriptionEngine
from tolka.pipeline.factory import build_engine

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, engine: TranscriptionEngine | None = None):
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_format)
    deps = AppDeps(settings=settings, engine=engine)

    mcp = build_mcp(deps)
    mcp_app = mcp.http_app(path="/", stateless_http=True)

    @asynccontextmanager
    async def app_lifespan(app: FastAPI):
        store = await open_job_store(settings)
        deps.store = store
        if settings.run_worker:
            if deps.engine is None:
                deps.engine = build_engine(settings)
            if settings.preload_models:
                await asyncio.to_thread(deps.engine.warm_up)
            queue = JobQueue(store, deps.engine, settings)
            deps.queue = queue
            await queue.start()
        yield
        if deps.queue is not None:
            await deps.queue.stop()
        await store.close()
        deps.store = None
        deps.queue = None

    app = FastAPI(
        title="tolka",
        description="Transcription and speaker-diarization service",
        lifespan=combine_lifespans(app_lifespan, mcp_app.lifespan),
    )
    app.state.deps = deps
    app.middleware("http")(request_observability_middleware)
    app.include_router(jobs_router, prefix="/v1", dependencies=[Depends(require_token)])
    app.mount("/mcp", mcp_app)

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response) -> dict[str, object]:
        database_ready = deps.store is not None and await deps.ready_store.ping()
        worker_ready = (
            deps.ready_queue.healthy
            if deps.queue is not None
            else await deps.ready_store.has_recent_worker(settings.worker_stale_s)
        )
        queued = await deps.ready_store.count_queued() if deps.store else None
        if queued is not None:
            QUEUE_DEPTH.set(queued)
        ready = database_ready and worker_ready
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "ready" if ready else "not_ready",
            "database": database_ready,
            "worker": worker_ready,
            "queued_jobs": queued,
        }

    @app.get("/healthz")
    async def healthz(response: Response) -> dict[str, object]:
        return await readyz(response)

    @app.get("/metrics", dependencies=[Depends(require_token)])
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
