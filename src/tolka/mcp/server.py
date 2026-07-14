import asyncio

import httpx
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import StaticTokenVerifier
from pydantic import ValidationError

from tolka.deps import AppDeps
from tolka.jobs.models import JobRequest, JobStatus, new_job

INSTRUCTIONS = """Transcribe audio (Swedish-optimized) with word timestamps and speaker
diarization. Use transcribe_audio for recordings that finish within minutes; for long
recordings use submit_transcription and poll get_transcription with the returned job id."""


def build_mcp(deps: AppDeps) -> FastMCP:
    auth = None
    if deps.settings.api_tokens:
        auth = StaticTokenVerifier(
            tokens={token: {"client_id": "tolka"} for token in deps.settings.api_tokens}
        )
    mcp = FastMCP(name="tolka", instructions=INSTRUCTIONS, auth=auth)

    async def _submit(url: str, language: str, diarize: bool) -> str:
        try:
            job_request = JobRequest(source_url=url, language=language, diarize=diarize)  # type: ignore[arg-type]
        except ValidationError as exc:
            raise ToolError(f"invalid arguments: {exc}") from exc
        job = new_job(job_request)
        await deps.ready_store.create(job)
        deps.ready_queue.notify()
        return job.id

    @mcp.tool
    async def submit_transcription(url: str, language: str = "auto", diarize: bool = True) -> str:
        """Submit an audio URL for transcription; returns a job id to poll with
        get_transcription. Use for long recordings. language: sv, en, or auto."""
        return await _submit(url, language, diarize)

    @mcp.tool
    async def get_transcription(job_id: str) -> str:
        """Get the transcript for a job id, or its status if not finished yet."""
        job = await deps.ready_store.get(job_id)
        if job is None:
            raise ToolError(f"unknown job id {job_id!r} (results are purged after retention)")
        if job.status == JobStatus.FAILED:
            raise ToolError(f"transcription failed: {job.error}")
        if job.status != JobStatus.COMPLETED:
            return f"status: {job.status.value} — not finished yet, ask again shortly"
        result = await deps.ready_store.get_result(job_id)
        assert result is not None
        return result.text

    @mcp.tool
    async def transcribe_audio(
        url: str, language: str = "auto", diarize: bool = True, ctx: Context | None = None
    ) -> str:
        """Transcribe an audio URL and wait for the result. Returns the transcript with
        timestamps and speaker labels. For very long recordings prefer
        submit_transcription + get_transcription."""
        await _reject_oversize_source(url, deps)
        job_id = await _submit(url, language, diarize)
        settings = deps.settings
        deadline = asyncio.get_running_loop().time() + settings.mcp_sync_timeout_s
        while asyncio.get_running_loop().time() < deadline:
            job = await deps.ready_store.get(job_id)
            assert job is not None
            if job.status == JobStatus.COMPLETED:
                result = await deps.ready_store.get_result(job_id)
                assert result is not None
                return result.text
            if job.status == JobStatus.FAILED:
                raise ToolError(f"transcription failed: {job.error}")
            if ctx is not None:
                await ctx.report_progress(progress=0, message=f"job {job_id}: {job.status.value}")
            await asyncio.sleep(settings.mcp_poll_interval_s)
        return (
            f"Transcription is still running after {settings.mcp_sync_timeout_s:.0f}s. "
            f"Job id: {job_id} — use get_transcription to fetch the result later."
        )

    return mcp


async def _reject_oversize_source(url: str, deps: AppDeps) -> None:
    """Best-effort size preflight so the synchronous tool is not used for huge files."""
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            response = await client.head(url)
        content_length = int(response.headers.get("content-length", 0))
    except (httpx.HTTPError, ValueError):
        return
    if content_length > deps.settings.mcp_max_audio_bytes:
        raise ToolError(
            f"source is {content_length} bytes, over the {deps.settings.mcp_max_audio_bytes} "
            "byte limit for synchronous transcription — use submit_transcription instead"
        )
