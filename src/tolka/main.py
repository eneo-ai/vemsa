import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastmcp.utilities.lifespan import combine_lifespans

from tolka.api.auth import require_token
from tolka.api.jobs import router as jobs_router
from tolka.config import Settings
from tolka.deps import AppDeps
from tolka.jobs.queue import JobQueue
from tolka.jobs.store import SqliteJobStore
from tolka.mcp.server import build_mcp
from tolka.pipeline.base import TranscriptionEngine

logger = logging.getLogger(__name__)


def _build_engine(settings: Settings) -> TranscriptionEngine:
    engine = settings.resolve_engine()
    logger.info("transcription engine: %s (TOLKA_ENGINE=%s)", engine, settings.engine)
    if engine == "fake":
        logger.warning("fake engine is enabled — returning canned transcripts")
        from tolka.pipeline.fake import CannedEngine

        return CannedEngine()
    if engine == "local":
        from tolka.pipeline.transcribe import EasyTranscriberEngine

        return EasyTranscriberEngine(settings)
    if engine == "hybrid":
        from tolka.pipeline.hybrid import HybridEngine

        return HybridEngine(settings)
    from tolka.pipeline.whisper_api import OpenAIWhisperEngine

    return OpenAIWhisperEngine(settings)


def create_app(settings: Settings | None = None, engine: TranscriptionEngine | None = None):
    settings = settings or Settings()
    deps = AppDeps(settings=settings, engine=engine)

    mcp = build_mcp(deps)
    mcp_app = mcp.http_app(path="/", stateless_http=True)

    @asynccontextmanager
    async def app_lifespan(app: FastAPI):
        store = SqliteJobStore(settings.db_path)
        await store.open()
        if deps.engine is None:
            deps.engine = _build_engine(settings)
        if settings.preload_models:
            await asyncio.to_thread(deps.engine.warm_up)
        queue = JobQueue(store, deps.engine, settings)
        deps.store = store
        deps.queue = queue
        await queue.start()
        yield
        await queue.stop()
        await store.close()
        deps.store = None
        deps.queue = None

    app = FastAPI(
        title="tolka",
        description="Transcription and speaker-diarization service",
        lifespan=combine_lifespans(app_lifespan, mcp_app.lifespan),
    )
    app.state.deps = deps
    app.include_router(jobs_router, prefix="/v1", dependencies=[Depends(require_token)])
    app.mount("/mcp", mcp_app)

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        queued = await deps.ready_store.count_queued() if deps.store else None
        return {"status": "ok", "queued_jobs": queued}

    return app
