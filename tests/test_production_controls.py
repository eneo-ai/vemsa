import contextlib

import httpx
import pytest
import respx
from httpx import Response
from pydantic import ValidationError

from conftest import FakeEngine
from tolka.config import Settings
from tolka.jobs.models import JobRequest, new_job
from tolka.main import create_app


@contextlib.asynccontextmanager
async def api_client(settings: Settings):
    app = create_app(settings=settings, engine=FakeEngine())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, app


def test_production_requires_named_credentials(tmp_path):
    with pytest.raises(ValidationError, match="client_id=token"):
        Settings(
            _env_file=None,
            environment="production",
            api_tokens="bare-token",
            engine="remote",
            whisper_api_base="https://whisper.example/v1",
            db_path=tmp_path / "db.sqlite3",
        )


def test_mcp_fails_closed_without_credentials(tmp_path):
    settings = Settings(
        _env_file=None,
        api_tokens="",
        engine="fake",
        db_path=tmp_path / "db.sqlite3",
    )
    with pytest.raises(ValueError, match="TOLKA_API_TOKENS"):
        create_app(settings=settings, engine=FakeEngine())


@respx.mock
async def test_clients_cannot_read_each_others_jobs(settings: Settings):
    settings.api_tokens = ["alpha=alpha-secret", "beta=beta-secret"]
    respx.get("https://example.org/private.mp3").mock(return_value=Response(200, content=b"audio"))
    async with api_client(settings) as (client, _):
        response = await client.post(
            "/v1/jobs",
            json={"source_url": "https://example.org/private.mp3"},
            headers={"Authorization": "Bearer alpha-secret"},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        other_client = {"Authorization": "Bearer beta-secret"}
        assert (await client.get(f"/v1/jobs/{job_id}", headers=other_client)).status_code == 404
        assert (
            await client.get(f"/v1/jobs/{job_id}/result", headers=other_client)
        ).status_code == 404


async def test_queue_limit_rejects_before_accepting_upload(settings: Settings):
    settings.max_queued_jobs = 1
    settings.max_queued_jobs_per_client = 1
    client_id = next(iter(settings.token_clients.values()))
    async with api_client(settings) as (client, app):
        await app.state.deps.ready_store.create(
            new_job(JobRequest(source_url="https://example.org/queued.mp3"), client_id=client_id)
        )
        response = await client.post(
            "/v1/jobs",
            files={"file": ("meeting.wav", b"audio", "audio/wav")},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 429


async def test_metrics_requires_auth(settings: Settings):
    async with api_client(settings) as (client, _):
        assert (await client.get("/metrics")).status_code == 401
        response = await client.get("/metrics", headers={"Authorization": "Bearer secret-token"})
        assert response.status_code == 200
        assert "tolka_jobs_submitted_total" in response.text
