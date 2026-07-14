import asyncio
import contextlib

import httpx
import pytest
import respx
from httpx import Response

from conftest import FakeEngine
from tolka.config import Settings
from tolka.jobs.models import JobRequest, new_job
from tolka.main import create_app

AUTH = {"Authorization": "Bearer secret-token"}


@contextlib.asynccontextmanager
async def api_client(settings: Settings, engine=None):
    app = create_app(settings=settings, engine=engine or FakeEngine())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, app


async def poll_until(client: httpx.AsyncClient, job_id: str, wanted: str, timeout: float = 5.0):
    async with asyncio.timeout(timeout):
        while True:
            response = await client.get(f"/v1/jobs/{job_id}", headers=AUTH)
            assert response.status_code == 200
            if response.json()["status"] == wanted:
                return response.json()
            await asyncio.sleep(0.01)


async def test_missing_or_wrong_token_gets_401(settings: Settings):
    async with api_client(settings) as (client, _):
        assert (await client.get("/v1/jobs/x")).status_code == 401
        response = await client.get("/v1/jobs/x", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401


@respx.mock
async def test_json_submission_full_lifecycle(settings: Settings):
    respx.get("https://example.org/m.mp3").mock(return_value=Response(200, content=b"audio"))
    async with api_client(settings) as (client, _):
        response = await client.post(
            "/v1/jobs",
            json={"source_url": "https://example.org/m.mp3", "language": "sv"},
            headers=AUTH,
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        job_id = body["job_id"]

        status = await poll_until(client, job_id, "completed")
        assert status["error"] is None
        assert "created_at" in status

        result = (await client.get(f"/v1/jobs/{job_id}/result", headers=AUTH)).json()
        assert result["language"] == "sv"
        assert result["model"] == settings.default_model
        assert result["segments"][0]["speaker"] == "SPEAKER_00"
        assert result["segments"][0]["words"][0]["word"] == "hej"


async def test_multipart_submission_without_diarization(settings: Settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        response = await client.post(
            "/v1/jobs",
            files={"file": ("meeting.wav", b"fake bytes", "audio/wav")},
            data={"language": "en", "diarize": "false"},
            headers=AUTH,
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        await poll_until(client, job_id, "completed")

        result = (await client.get(f"/v1/jobs/{job_id}/result", headers=AUTH)).json()
        assert result["segments"][0]["speaker"] is None
        assert engine.calls[0]["language"] == "en"
        assert engine.calls[0]["diarize"] is False


async def test_unknown_job_is_404(settings: Settings):
    async with api_client(settings) as (client, _):
        assert (await client.get("/v1/jobs/nope", headers=AUTH)).status_code == 404
        assert (await client.get("/v1/jobs/nope/result", headers=AUTH)).status_code == 404


async def test_result_before_completion_is_409(settings: Settings):
    async with api_client(settings) as (client, app):
        # insert a queued job directly, without waking the worker
        job = new_job(JobRequest(source_url="https://example.org/x.mp3"))
        await app.state.deps.store.create(job)

        response = await client.get(f"/v1/jobs/{job.id}/result", headers=AUTH)
        assert response.status_code == 409
        assert response.json()["detail"]["status"] == "queued"


async def test_oversize_upload_is_413(settings: Settings):
    settings.max_audio_bytes = 10
    async with api_client(settings) as (client, _):
        response = await client.post(
            "/v1/jobs",
            files={"file": ("big.wav", b"x" * 100, "audio/wav")},
            headers=AUTH,
        )
        assert response.status_code == 413


@pytest.mark.parametrize(
    "payload",
    [
        {"source_url": "https://example.org/a.mp3", "language": "da"},
        {"language": "sv"},  # no source_url
        {"source_url": "not a url"},
    ],
)
async def test_invalid_json_submission_is_422(settings: Settings, payload):
    async with api_client(settings) as (client, _):
        response = await client.post("/v1/jobs", json=payload, headers=AUTH)
        assert response.status_code == 422


async def test_multipart_without_file_is_422(settings: Settings):
    async with api_client(settings) as (client, _):
        response = await client.post("/v1/jobs", data={"language": "sv"}, headers=AUTH)
        # form-encoded body without a file part is not valid JSON either
        assert response.status_code == 422


async def test_healthz_needs_no_auth(settings: Settings):
    async with api_client(settings) as (client, _):
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
