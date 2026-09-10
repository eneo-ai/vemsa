"""Operator dashboard: the built UI page plus the JSON it polls.

Mounted at /ops by the app factory, behind `require_ops_auth`. Unlike /v1 these
reads span every client: the audience is whoever runs the service."""

import logging
from datetime import UTC, datetime
from importlib.resources import files
from typing import Any, Literal

import asyncpg
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from vemsa.api.health import _service_version, load_readiness
from vemsa.deps import AppDeps
from vemsa.jobs.postgres_store import PostgresJobStore
from vemsa.ops import queries
from vemsa.ops.queries import WINDOWS, Range

logger = logging.getLogger(__name__)
router = APIRouter()

Window = Literal["1h", "6h", "24h", "7d", "30d"]
STATIC_DIR = files("vemsa.ops").joinpath("static")

_UNBUILT_PAGE = """<!doctype html>
<title>vemsa ops</title>
<body style="font-family: system-ui; margin: 3rem; max-width: 40rem">
<h1>vemsa ops</h1>
<p>The dashboard UI has not been built into this installation.</p>
<p>The JSON endpoints under <code>/ops/api/</code> work regardless; to get the page,
run <code>npm ci &amp;&amp; npm run build</code> in <code>ui/</code> and restart, or use
the container image, which builds it in.</p>
</body>
"""


def _deps(request: Request) -> AppDeps:
    return request.app.state.deps


def _pool(request: Request) -> asyncpg.Pool:
    store = _deps(request).ready_store
    if not isinstance(store, PostgresJobStore):
        raise HTTPException(503, detail="statistics need the PostgreSQL job store")
    return store.pool


def _range(window: str, *, bucket_index: int) -> Range:
    return Range.for_window(window, bucket=WINDOWS[window][bucket_index])


def _settings_summary(deps: AppDeps) -> dict[str, Any]:
    """Whitelist only: no tokens, keys, URLs with credentials, or secrets."""
    settings = deps.settings
    return {
        "service_version": _service_version(),
        "environment": settings.environment,
        "engine": settings.resolve_engine(),
        "default_model": settings.default_model,
        "diarization_model": settings.diarization_model,
        "emissions_model": settings.emissions_model,
        "worker_concurrency": settings.worker_concurrency,
        "gpu_concurrency": settings.gpu_concurrency,
        "oom_max_attempts": settings.oom_max_attempts,
        "retention_hours": settings.retention_hours,
        "stats_retention_days": settings.stats_retention_days,
        "max_queued_jobs": settings.max_queued_jobs,
        "max_queued_jobs_per_client": settings.max_queued_jobs_per_client,
        "min_alignment": settings.min_alignment,
        "diarize_prefer_align": settings.diarize_prefer_align,
        "worker_stale_s": settings.worker_stale_s,
        "lease_heartbeat_s": settings.lease_heartbeat_s,
        "in_process_worker": deps.queue is not None,
    }


@router.get("", response_class=HTMLResponse)
async def page() -> HTMLResponse:
    index = STATIC_DIR.joinpath("index.html")
    if not index.is_file():
        return HTMLResponse(_UNBUILT_PAGE, status_code=200)
    return HTMLResponse(index.read_text(encoding="utf-8"))


@router.get("/api/overview")
async def overview(request: Request) -> dict[str, Any]:
    deps = _deps(request)
    pool = _pool(request)
    now = datetime.now(UTC)
    readiness = await load_readiness(deps)
    return {
        "now": now,
        "service": _settings_summary(deps),
        "readiness": readiness.model_dump(),
        "jobs": await queries.status_counts(pool),
        "oldest_queued_s": await queries.oldest_queued_age_s(pool, now=now),
        "running": await queries.running_jobs(pool, now=now),
        "workers": await queries.workers(pool, now=now, stale_after_s=deps.settings.worker_stale_s),
    }


@router.get("/api/throughput")
async def throughput(request: Request, window: Window = "24h") -> dict[str, Any]:
    span = _range(window, bucket_index=1)
    return {**span.model_dump(), **await queries.throughput(_pool(request), span)}


@router.get("/api/performance")
async def performance(request: Request, window: Window = "24h") -> dict[str, Any]:
    span = _range(window, bucket_index=1)
    return {**span.model_dump(), **await queries.performance(_pool(request), span)}


@router.get("/api/quality")
async def quality(request: Request, window: Window = "24h") -> dict[str, Any]:
    span = _range(window, bucket_index=1)
    return {**span.model_dump(), **await queries.quality(_pool(request), span)}


@router.get("/api/clients")
async def clients(request: Request, window: Window = "24h") -> dict[str, Any]:
    span = _range(window, bucket_index=1)
    return {**span.model_dump(), "clients": await queries.clients(_pool(request), span)}


@router.get("/api/host")
async def host(request: Request, window: Window = "6h") -> dict[str, Any]:
    span = _range(window, bucket_index=2)
    return {**span.model_dump(), "workers": await queries.host_series(_pool(request), span)}


@router.get("/api/jobs/recent")
async def recent_jobs(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    return {"jobs": await queries.recent_jobs(_pool(request), limit=limit)}
