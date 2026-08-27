import asyncio
import contextlib

import pytest
import respx
from fastmcp import Client
from fastmcp.exceptions import ToolError
from httpx import Response

from conftest import FailingEngine, FakeEngine
from tolka.config import Settings
from tolka.deps import AppDeps
from tolka.jobs.postgres_store import PostgresJobStore
from tolka.jobs.queue import JobQueue
from tolka.mcp.server import build_mcp

AUDIO_URL = "https://example.org/meeting.mp3"


def mock_audio(size_header: str = "5") -> None:
    respx.get(AUDIO_URL).mock(return_value=Response(200, content=b"audio"))
    respx.head(AUDIO_URL).mock(return_value=Response(200, headers={"content-length": size_header}))


@contextlib.asynccontextmanager
async def mcp_client(settings: Settings, engine=None):
    settings.mcp_poll_interval_s = 0.01
    deps = AppDeps(settings=settings, engine=engine or FakeEngine())
    store = PostgresJobStore(settings.database_url)
    await store.open()
    queue = JobQueue(store, deps.engine, settings)
    deps.store = store
    deps.queue = queue
    await queue.start()
    try:
        async with Client(build_mcp(deps)) as client:
            yield client
    finally:
        await queue.stop()
        await store.close()


@respx.mock
async def test_submit_and_get_transcription(settings: Settings):
    mock_audio()
    async with mcp_client(settings) as client:
        submitted = await client.call_tool("submit_transcription", {"url": AUDIO_URL})
        job_id = submitted.data
        assert isinstance(job_id, str) and job_id

        async with asyncio.timeout(5):
            while True:
                result = await client.call_tool("get_transcription", {"job_id": job_id})
                if not result.data.startswith("status:"):
                    break
                await asyncio.sleep(0.01)
        assert "hej och välkomna" in result.data


async def test_get_transcription_unknown_job(settings: Settings):
    async with mcp_client(settings) as client:
        with pytest.raises(ToolError, match="unknown job id"):
            await client.call_tool("get_transcription", {"job_id": "nope"})


async def test_submit_rejects_bad_language(settings: Settings):
    async with mcp_client(settings) as client:
        with pytest.raises(ToolError, match="invalid arguments"):
            await client.call_tool(
                "submit_transcription", {"url": AUDIO_URL, "language": "klingon"}
            )


@respx.mock
async def test_transcribe_audio_waits_for_result(settings: Settings):
    mock_audio()
    async with mcp_client(settings) as client:
        result = await client.call_tool("transcribe_audio", {"url": AUDIO_URL})
        assert "hej och välkomna" in result.data


@respx.mock
async def test_transcribe_audio_degrades_to_job_id_on_timeout(settings: Settings):
    mock_audio()
    settings.mcp_sync_timeout_s = 0.0
    async with mcp_client(settings) as client:
        result = await client.call_tool("transcribe_audio", {"url": AUDIO_URL})
        assert "get_transcription" in result.data


@respx.mock
async def test_transcribe_audio_surfaces_job_failure(settings: Settings):
    mock_audio()
    async with mcp_client(settings, engine=FailingEngine()) as client:
        with pytest.raises(ToolError, match="transcription failed"):
            await client.call_tool("transcribe_audio", {"url": AUDIO_URL})


@respx.mock
async def test_transcribe_audio_rejects_oversize_source(settings: Settings):
    mock_audio(size_header=str(10**12))
    async with mcp_client(settings) as client:
        with pytest.raises(ToolError, match="submit_transcription instead"):
            await client.call_tool("transcribe_audio", {"url": AUDIO_URL})
