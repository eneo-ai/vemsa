import asyncio
import contextlib

import pytest
import respx
from fastmcp import Client
from fastmcp.exceptions import ToolError
from httpx import Response

from conftest import FailingEngine, FakeEngine
from vemsa.config import Settings
from vemsa.deps import AppDeps
from vemsa.jobs.postgres_store import PostgresJobStore
from vemsa.jobs.queue import JobQueue
from vemsa.mcp.server import build_mcp

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
async def test_submit_vocabulary_reaches_the_engine(settings: Settings):
    mock_audio()
    engine = FakeEngine()
    async with mcp_client(settings, engine) as client:
        submitted = await client.call_tool(
            "submit_transcription", {"url": AUDIO_URL, "vocabulary": ["Çagri", "Vemsa"]}
        )
        async with asyncio.timeout(5):
            while True:
                result = await client.call_tool("get_transcription", {"job_id": submitted.data})
                if not result.data.startswith("status:"):
                    break
                await asyncio.sleep(0.01)
    assert engine.calls[0]["vocabulary"] == ["Çagri", "Vemsa"]


async def test_submit_rejects_oversized_vocabulary(settings: Settings):
    async with mcp_client(settings) as client:
        with pytest.raises(ToolError, match="invalid arguments"):
            await client.call_tool(
                "submit_transcription", {"url": AUDIO_URL, "vocabulary": ["x"] * 51}
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


# --- HTTP transport: the documented /mcp URL must work without a redirect ------

MCP_HEADERS = {
    "Authorization": "Bearer secret-token",
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "probe", "version": "0"},
    },
}


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
async def test_mcp_endpoint_serves_with_and_without_trailing_slash(settings: Settings, path: str):
    import httpx

    from vemsa.main import create_app

    settings.run_worker = False
    app = create_app(settings=settings, engine=FakeEngine())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(path, headers=MCP_HEADERS, json=INITIALIZE)

    assert response.status_code == 200, (response.status_code, response.headers.get("location"))
    assert '"protocolVersion"' in response.text
    assert "vemsa" in response.text


async def test_mcp_alias_does_not_shadow_other_routes(settings: Settings):
    import httpx

    from vemsa.main import create_app

    settings.run_worker = False
    app = create_app(settings=settings, engine=FakeEngine())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (await client.get("/livez")).status_code == 200
            assert (await client.get("/mcpx")).status_code == 404
