import asyncio

import httpx
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import StaticTokenVerifier
from fastmcp.server.dependencies import get_access_token
from pydantic import ValidationError

from vemsa.deps import AppDeps
from vemsa.jobs.models import JobRequest, JobStatus, new_job
from vemsa.security import ForbiddenUrlError, validate_outbound_url

INSTRUCTIONS = """Transcribe audio (Swedish-optimized) with word timestamps and speaker
diarization. Use transcribe_audio for recordings that finish within minutes; for long
recordings use submit_transcription and poll get_transcription with the returned job id."""


def build_mcp(deps: AppDeps) -> FastMCP:
    """MCP tools are transcribe-only by design; task=diarize and task=align
    (caller-supplied transcripts) are REST-only — see the Job API section of the
    README."""
    token_clients = deps.settings.token_clients
    if not token_clients:
        raise ValueError("VEMSA_API_TOKENS must be configured before MCP can start")
    auth = StaticTokenVerifier(
        tokens={token: {"client_id": client_id} for token, client_id in token_clients.items()}
    )
    mcp = FastMCP(name="vemsa", instructions=INSTRUCTIONS, auth=auth)

    def _client_id() -> str:
        access_token = get_access_token()
        if access_token is not None:
            return access_token.client_id
        # Direct in-memory FastMCP clients do not pass through HTTP authentication.
        # Keep that transport useful for tests and trusted embedded use only when
        # there is exactly one unambiguous configured identity.
        client_ids = set(token_clients.values())
        if len(client_ids) == 1:
            return next(iter(client_ids))
        raise ToolError("authenticated client identity is unavailable")

    async def _submit(
        url: str, language: str, diarize: bool, vocabulary: list[str] | None = None
    ) -> str:
        client_id = _client_id()
        total = await deps.ready_store.count_active()
        client_total = await deps.ready_store.count_active(client_id=client_id)
        if total >= deps.settings.max_queued_jobs:
            raise ToolError("job queue is full")
        if client_total >= deps.settings.max_queued_jobs_per_client:
            raise ToolError("client has reached its active job limit")
        try:
            job_request = JobRequest(
                source_url=url,  # type: ignore[arg-type]
                language=language,  # type: ignore[arg-type]
                diarize=diarize,
                vocabulary=vocabulary,
            )
        except ValidationError as exc:
            raise ToolError(f"invalid arguments: {exc}") from exc
        job = new_job(job_request, client_id=client_id)
        await deps.ready_store.create(job)
        if deps.queue is not None:
            deps.queue.notify()
        return job.id

    @mcp.tool
    async def submit_transcription(
        url: str,
        language: str = "auto",
        diarize: bool = True,
        vocabulary: list[str] | None = None,
    ) -> str:
        """Submit an audio URL for transcription; returns a job id to poll with
        get_transcription. Use for long recordings. language: sv, en, or auto.
        vocabulary: names/terms expected in the audio (e.g. participant names),
        hinting the recognizer's spelling; max 50 short entries."""
        return await _submit(url, language, diarize, vocabulary)

    @mcp.tool
    async def get_transcription(job_id: str) -> str:
        """Get the transcript for a job id, or its status if not finished yet."""
        client_id = _client_id()
        job = await deps.ready_store.get(job_id, client_id=client_id)
        if job is None:
            raise ToolError(f"unknown job id {job_id!r} (results are purged after retention)")
        if job.status == JobStatus.FAILED:
            raise ToolError(f"transcription failed: {job.error}")
        if job.status == JobStatus.CANCELLED:
            raise ToolError("transcription was cancelled")
        if job.status != JobStatus.COMPLETED:
            return f"status: {job.status.value} — not finished yet, ask again shortly"
        result = await deps.ready_store.get_result(job_id, client_id=client_id)
        assert result is not None
        return result.text

    @mcp.tool
    async def transcribe_audio(
        url: str,
        language: str = "auto",
        diarize: bool = True,
        vocabulary: list[str] | None = None,
        ctx: Context | None = None,
    ) -> str:
        """Transcribe an audio URL and wait for the result. Returns the transcript with
        timestamps and speaker labels. For very long recordings prefer
        submit_transcription + get_transcription. vocabulary: names/terms expected
        in the audio (e.g. participant names), hinting the recognizer's spelling."""
        await _reject_oversize_source(url, deps)
        job_id = await _submit(url, language, diarize, vocabulary)
        settings = deps.settings
        deadline = asyncio.get_running_loop().time() + settings.mcp_sync_timeout_s
        while asyncio.get_running_loop().time() < deadline:
            client_id = _client_id()
            job = await deps.ready_store.get(job_id, client_id=client_id)
            assert job is not None
            if job.status == JobStatus.COMPLETED:
                result = await deps.ready_store.get_result(job_id, client_id=client_id)
                assert result is not None
                return result.text
            if job.status == JobStatus.FAILED:
                raise ToolError(f"transcription failed: {job.error}")
            if job.status == JobStatus.CANCELLED:
                raise ToolError("transcription was cancelled")
            if ctx is not None:
                await ctx.report_progress(
                    progress=0,
                    message=f"job {job_id}: {job.status.value} ({job.stage.value})",
                )
            await asyncio.sleep(settings.mcp_poll_interval_s)
        return (
            f"Transcription is still running after {settings.mcp_sync_timeout_s:.0f}s. "
            f"Job id: {job_id} — use get_transcription to fetch the result later."
        )

    return mcp


async def _reject_oversize_source(url: str, deps: AppDeps) -> None:
    """Best-effort size preflight so the synchronous tool is not used for huge files."""
    try:
        await validate_outbound_url(
            url,
            allow_private=deps.settings.allow_private_urls,
            allowed_hosts=deps.settings.source_allowed_hosts,
        )
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            response = await client.head(url)
        content_length = int(response.headers.get("content-length", 0))
    except ForbiddenUrlError as exc:
        raise ToolError(f"source URL was rejected: {exc}") from exc
    except (httpx.HTTPError, ValueError):
        return
    if content_length > deps.settings.mcp_max_audio_bytes:
        raise ToolError(
            f"source is {content_length} bytes, over the {deps.settings.mcp_max_audio_bytes} "
            "byte limit for synchronous transcription — use submit_transcription instead"
        )
