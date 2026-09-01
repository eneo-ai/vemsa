import contextlib
import json
import logging

import httpx
import pytest
import respx
from httpx import Response
from pydantic import ValidationError

from conftest import FakeEngine, make_wav_bytes
from vemsa.config import Settings
from vemsa.jobs.models import JobRequest, new_job
from vemsa.main import create_app
from vemsa.observability import JsonFormatter


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
            database_url="postgresql://vemsa:vemsa@localhost/vemsa",
        )


def test_mcp_fails_closed_without_credentials(tmp_path):
    settings = Settings(
        _env_file=None,
        api_tokens="",
        engine="fake",
        database_url="postgresql://vemsa:vemsa@localhost/vemsa",
    )
    with pytest.raises(ValueError, match="VEMSA_API_TOKENS"):
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
        assert (await client.delete(f"/v1/jobs/{job_id}", headers=other_client)).status_code == 404


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
        assert "vemsa_jobs_submitted_total" in response.text


async def test_submission_and_cancellation_logs_exclude_sensitive_content(
    settings: Settings,
):
    settings.run_worker = False
    sensitive_name = "Highly Sensitive Participant"
    async with api_client(settings) as (client, _):
        records: list[logging.LogRecord] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = CaptureHandler()
        logging.getLogger().addHandler(handler)
        try:
            response = await client.post(
                "/v1/jobs",
                files={"file": ("private-meeting.wav", make_wav_bytes(), "audio/wav")},
                data={
                    "task": "diarize",
                    "words": json.dumps([{"word": sensitive_name, "start": 0.0, "end": 1.0}]),
                },
                headers={"Authorization": "Bearer secret-token"},
            )
            assert response.status_code == 202
            await client.delete(
                f"/v1/jobs/{response.json()['job_id']}",
                headers={"Authorization": "Bearer secret-token"},
            )
        finally:
            logging.getLogger().removeHandler(handler)

    rendered = "\n".join(JsonFormatter().format(record) for record in records)
    events = {getattr(record, "event", None) for record in records}
    assert {"job.submitted", "job.cancellation_accepted"} <= events
    assert sensitive_name not in rendered
    assert "secret-token" not in rendered
    assert "private-meeting.wav" not in rendered
