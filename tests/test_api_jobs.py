import asyncio
import contextlib
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
import respx
from httpx import Response

from conftest import FakeEngine, GateEngine, make_result, make_wav_bytes
from vemsa.config import Settings
from vemsa.jobs.models import (
    JobRequest,
    JobStage,
    SpeakerBounds,
    TranscriptionResult,
    new_job,
)
from vemsa.jobs.queue import JobLeaseLostError, failure_kind
from vemsa.main import create_app
from vemsa.pipeline.base import StageReporter, report_stage

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


@pytest.mark.parametrize(
    ("failure", "failure_kind"),
    [
        ("value_error", "internal"),
        ("alignment_shape", "internal"),
        ("decode", "input"),
        ("oom", "capacity"),
        ("provider", "provider"),
        ("provider_timeout", "provider"),
        ("provider_oom", "provider"),
        ("internal", "internal"),
        ("cancelled", "cancelled"),
    ],
)
@respx.mock
async def test_status_reports_failure_kind(
    settings: Settings, failure: str, failure_kind: str, monkeypatch
):
    from types import SimpleNamespace

    from vemsa.pipeline.align import words_from_alignments
    from vemsa.pipeline.diarize import _decodable_audio
    from vemsa.pipeline.whisper_api import request_transcription

    settings.oom_max_attempts = 3 if failure == "provider_oom" else 1
    settings.whisper_api_base = "https://whisper.example/v1"
    settings.run_worker = failure_kind != "cancelled"
    route = respx.post("https://whisper.example/v1/audio/transcriptions")
    if failure == "provider_timeout":
        route.mock(side_effect=httpx.ReadTimeout("provider timed out"))
    elif failure == "provider_oom":
        route.mock(return_value=Response(503, text="CUDA out of memory"))
    else:
        route.mock(return_value=Response(503, text="provider unavailable"))
    if failure == "decode":
        monkeypatch.setattr(
            "vemsa.pipeline.diarize.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1),
        )

    class ErrorEngine(FakeEngine):
        def transcribe(self, audio_path, **kwargs):
            if failure == "alignment_shape":
                words_from_alignments([object()])
            if failure == "decode":
                _decodable_audio(audio_path)
            if failure_kind == "provider":
                request_transcription(settings, audio_path, language="sv", model="test")
            errors = {
                "value_error": ValueError("unexpected value"),
                "oom": MemoryError("out of memory"),
                "internal": RuntimeError("unexpected failure"),
            }
            raise errors[failure]

    async with api_client(settings, ErrorEngine()) as (client, _):
        response = await client.post("/v1/jobs", files={"file": ("a.wav", b"audio")}, headers=AUTH)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        if failure_kind == "cancelled":
            assert (await client.delete(f"/v1/jobs/{job_id}", headers=AUTH)).status_code == 202
        status = await poll_until(
            client, job_id, "cancelled" if failure_kind == "cancelled" else "failed"
        )
        assert status["failure_kind"] == failure_kind


@pytest.mark.parametrize("engine_name", ["remote", "hybrid"])
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"words": [{"word": "hej", "start": "invalid", "end": 1.0}]},
            id="invalid-timestamp",
        ),
        pytest.param({"words": [{"start": 0.0, "end": 1.0}]}, id="missing-field"),
        pytest.param([], id="wrong-json-shape"),
        pytest.param({"words": 42}, id="wrong-collection-shape"),
        pytest.param({"words": [None]}, id="wrong-record-shape"),
        pytest.param({"text": "hej"}, id="unusable-transcript"),
        pytest.param(
            {"duration": "invalid", "words": [{"word": "hej", "start": 0.0, "end": 1.0}]},
            id="invalid-duration",
        ),
    ],
)
@respx.mock
async def test_malformed_provider_response_reports_provider_failure(
    settings: Settings, engine_name: str, payload
):
    from test_whisper_api import FakeDiarizer
    from vemsa.pipeline.hybrid import HybridEngine
    from vemsa.pipeline.whisper_api import OpenAIWhisperEngine

    settings.whisper_api_base = "https://whisper.example/v1"
    respx.post("https://whisper.example/v1/audio/transcriptions").mock(
        return_value=Response(200, json=payload)
    )
    engine_class = OpenAIWhisperEngine if engine_name == "remote" else HybridEngine
    engine = engine_class(settings, diarizer=FakeDiarizer())
    async with api_client(settings, engine) as (client, _):
        response = await client.post(
            "/v1/jobs",
            files={"file": ("a.wav", b"audio")},
            data={"diarize": "false"},
            headers=AUTH,
        )
        assert response.status_code == 202
        status = await poll_until(client, response.json()["job_id"], "failed")
        assert status["failure_kind"] == "provider"


def test_lease_loss_is_not_classified_as_capacity():
    assert failure_kind(JobLeaseLostError("worker no longer owns the job")) == "internal"


@pytest.mark.parametrize("failure", [JobLeaseLostError, ValueError])
async def test_former_worker_cannot_fail_reclaimed_job(settings: Settings, failure):
    engine = GateEngine(fail_first_with=failure)
    async with api_client(settings, engine) as (client, app):
        response = await client.post("/v1/jobs", files={"file": ("a.wav", b"audio")}, headers=AUTH)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        store = app.state.deps.ready_store
        try:
            await engine.wait_entered()
            await store.pool.execute(
                "UPDATE jobs SET lease_expires_at = now() - interval '1 hour' WHERE id = $1",
                job_id,
            )
            reclaimed = await store.claim_next_queued(worker_id="successor", lease_for_s=60)
            assert reclaimed is not None and reclaimed.id == job_id
        finally:
            engine.gate.set()
        async with asyncio.timeout(5):
            await app.state.deps.ready_queue.stop()
        status = (await client.get(f"/v1/jobs/{job_id}", headers=AUTH)).json()
        assert status["status"] == "running"
        assert status["failure_kind"] is None
        assert status["error"] is None
        current = await store.get(job_id)
        assert current is not None and current.lease_owner == "successor"


@pytest.mark.parametrize("upload", [False, True])
async def test_idempotent_submit_returns_original_at_capacity(settings: Settings, upload: bool):
    settings.run_worker = False
    settings.max_queued_jobs = settings.max_queued_jobs_per_client = 1
    headers = {**AUTH, "Idempotency-Key": "meeting-1"}
    payload = (
        {"files": {"file": ("meeting.wav", b"audio")}, "data": {"language": "sv"}}
        if upload
        else {"json": {"source_url": "https://example.org/a.mp3", "language": "sv"}}
    )
    async with api_client(settings) as (client, app):
        first = await client.post("/v1/jobs", headers=headers, **payload)
        assert first.status_code == 202
        repeats = await asyncio.gather(
            *(client.post("/v1/jobs", headers=headers, **payload) for _ in range(8))
        )
        assert [response.status_code for response in repeats] == [202] * 8
        assert all(response.json() == first.json() for response in repeats)
        assert await app.state.deps.ready_store.count_active() == 1
        assert len(list(settings.work_dir.glob("*"))) == (1 if upload else 0)
        job_id = first.json()["job_id"]
        await client.delete(f"/v1/jobs/{job_id}", headers=AUTH)
        terminal = await client.post("/v1/jobs", headers=headers, **payload)
        assert terminal.status_code == 202
        assert terminal.json() == {"job_id": job_id, "status": "cancelled"}
        assert await app.state.deps.ready_store.pool.fetchval("SELECT COUNT(*) FROM jobs") == 1
        assert list(settings.work_dir.glob("*")) == []


@pytest.mark.parametrize("change", ["payload", "audio"])
async def test_idempotency_conflict_and_client_scope(settings: Settings, change: str):
    settings.run_worker = False
    settings.api_tokens = ["alpha=alpha-secret", "beta=beta-secret"]
    headers = {"Authorization": "Bearer alpha-secret", "Idempotency-Key": "meeting-1"}
    async with api_client(settings) as (client, app):
        first = await client.post(
            "/v1/jobs",
            headers=headers,
            files={"file": ("a.wav", b"audio")},
            data={"language": "sv"},
        )
        assert first.status_code == 202
        changed = await client.post(
            "/v1/jobs",
            headers=headers,
            files={"file": ("a.wav", b"changed" if change == "audio" else b"audio")},
            data={"language": "en" if change == "payload" else "sv"},
        )
        assert changed.status_code == 409
        other = await client.post(
            "/v1/jobs",
            headers={**headers, "Authorization": "Bearer beta-secret"},
            files={"file": ("a.wav", b"audio")},
            data={"language": "sv"},
        )
        assert other.status_code == 202
        assert other.json()["job_id"] != first.json()["job_id"]
        assert await app.state.deps.ready_store.count_active() == 2
        assert len(list(settings.work_dir.iterdir())) == 2


async def test_concurrent_idempotent_submits_create_one_job(settings: Settings):
    settings.run_worker = False
    async with api_client(settings) as (client, app):
        responses = await asyncio.gather(
            *(
                client.post(
                    "/v1/jobs",
                    json={"source_url": "https://example.org/a.mp3"},
                    headers={**AUTH, "Idempotency-Key": "meeting-1"},
                )
                for _ in range(8)
            )
        )
        assert all(response.status_code == 202 for response in responses)
        assert len({response.json()["job_id"] for response in responses}) == 1
        assert await app.state.deps.ready_store.count_active() == 1


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
        assert status["failure_kind"] is None
        assert status["stage"] == "finalizing"
        assert status["queue_position"] is None
        assert "created_at" in status

        result = (await client.get(f"/v1/jobs/{job_id}/result", headers=AUTH)).json()
        assert result["language"] == "sv"
        assert result["model"] == settings.default_model
        assert result["segments"][0]["speaker"] == "SPEAKER_00"
        assert result["segments"][0]["words"][0]["word"] == "hej"

        terminal_cancel = await client.delete(f"/v1/jobs/{job_id}", headers=AUTH)
        assert terminal_cancel.status_code == 200
        assert terminal_cancel.json()["status"] == "completed"
        assert terminal_cancel.json()["cancellation_requested"] is False


async def test_multipart_submission_without_diarization(settings: Settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, app):
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


async def test_vocabulary_is_plumbed_to_the_engine(settings: Settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        response = await client.post(
            "/v1/jobs",
            files={"file": ("meeting.wav", b"fake bytes", "audio/wav")},
            data={"language": "sv", "vocabulary": '["Anna Lindqvist", "Vemsa"]'},
            headers=AUTH,
        )
        assert response.status_code == 202
        await poll_until(client, response.json()["job_id"], "completed")
        assert engine.calls[0]["vocabulary"] == ["Anna Lindqvist", "Vemsa"]


@respx.mock
async def test_json_body_vocabulary_is_plumbed(settings: Settings):
    respx.get("https://example.org/m.mp3").mock(return_value=Response(200, content=b"audio"))
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        response = await client.post(
            "/v1/jobs",
            json={"source_url": "https://example.org/m.mp3", "vocabulary": ["Çagri"]},
            headers=AUTH,
        )
        assert response.status_code == 202
        await poll_until(client, response.json()["job_id"], "completed")
        assert engine.calls[0]["vocabulary"] == ["Çagri"]


async def test_vocabulary_validation_failures(settings: Settings):
    async with api_client(settings) as (client, _):

        async def submit_vocabulary(value: str, **extra: str) -> httpx.Response:
            return await client.post(
                "/v1/jobs",
                files={"file": ("a.wav", b"fake audio")},
                data={"vocabulary": value, **extra},
                headers=AUTH,
            )

        # only meaningful when Vemsa runs the ASR itself
        words = '[{"word":"hej","start":0.0,"end":0.4}]'
        response = await submit_vocabulary('["Anna"]', task="diarize", words=words)
        assert response.status_code == 422
        assert "task=transcribe" in str(response.json()["detail"])
        # malformed JSON part
        response = await submit_vocabulary("[{broken")
        assert response.status_code == 422
        assert "not valid JSON" in response.json()["detail"]
        # over the caps
        assert (await submit_vocabulary(json.dumps(["x"] * 51))).status_code == 422
        assert (await submit_vocabulary(json.dumps(["y" * 65]))).status_code == 422
        assert (await submit_vocabulary(json.dumps(["z" * 60] * 20))).status_code == 422


async def test_whitespace_vocabulary_normalizes_to_none(settings: Settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        response = await client.post(
            "/v1/jobs",
            files={"file": ("a.wav", b"fake audio")},
            data={"vocabulary": '["  ", ""]'},
            headers=AUTH,
        )
        assert response.status_code == 202
        await poll_until(client, response.json()["job_id"], "completed")
        assert engine.calls[0]["vocabulary"] is None


async def test_unknown_job_is_404(settings: Settings):
    async with api_client(settings) as (client, _):
        assert (await client.get("/v1/jobs/nope", headers=AUTH)).status_code == 404
        assert (await client.get("/v1/jobs/nope/result", headers=AUTH)).status_code == 404


async def test_result_before_completion_is_409(settings: Settings):
    async with api_client(settings) as (client, app):
        # insert a queued job directly, without waking the worker
        client_id = next(iter(settings.token_clients.values()))
        job = new_job(JobRequest(source_url="https://example.org/x.mp3"), client_id=client_id)
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
        assert response.json()["status"] == "ready"


async def test_authenticated_readiness_contract(settings: Settings):
    async with api_client(settings) as (client, _):
        assert (await client.get("/v1/health/ready")).status_code == 401
        response = await client.get("/v1/health/ready", headers=AUTH)
        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "service_version": "0.1.0",
            "database_ready": True,
            "worker_ready": True,
            "queue_accepting_jobs": True,
            "queued_jobs": 0,
        }


async def test_readiness_reports_saturated_queue_without_becoming_unready(
    settings: Settings,
):
    settings.run_worker = False
    settings.max_queued_jobs = 1
    settings.max_queued_jobs_per_client = 1
    async with api_client(settings) as (client, app):
        client_id = next(iter(settings.token_clients.values()))
        await app.state.deps.ready_store.record_worker_heartbeat("external-worker")
        await app.state.deps.ready_store.create(
            new_job(JobRequest(source_url="https://example.org/queued.mp3"), client_id=client_id)
        )
        response = await client.get("/v1/health/ready", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert response.json()["queue_accepting_jobs"] is False
        assert response.json()["queued_jobs"] == 1


async def test_readiness_is_503_without_a_worker(settings: Settings):
    settings.run_worker = False
    async with api_client(settings) as (client, _):
        response = await client.get("/v1/health/ready", headers=AUTH)
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"
        assert response.json()["worker_ready"] is False


async def test_openapi_advertises_job_lifecycle(settings: Settings):
    async with api_client(settings) as (client, _):
        schema = (await client.get("/openapi.json")).json()
        assert "delete" in schema["paths"]["/v1/jobs/{job_id}"]
        assert "get" in schema["paths"]["/v1/health/ready"]
        statuses = schema["components"]["schemas"]["JobStatus"]["enum"]
        stages = schema["components"]["schemas"]["JobStage"]["enum"]
        assert statuses == ["queued", "running", "completed", "failed", "cancelled"]
        assert stages == [
            "queued",
            "transcribing",
            "aligning",
            "diarizing",
            "finalizing",
        ]


async def test_queued_job_reports_position_and_cancels_idempotently(settings: Settings):
    settings.run_worker = False
    async with api_client(settings) as (client, app):
        response = await client.post(
            "/v1/jobs",
            files={"file": ("meeting.wav", make_wav_bytes(), "audio/wav")},
            headers=AUTH,
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        stored = await app.state.deps.ready_store.get(job_id)
        assert stored is not None and stored.audio_path is not None
        audio_path = Path(stored.audio_path)

        status_response = await client.get(f"/v1/jobs/{job_id}", headers=AUTH)
        assert status_response.json()["stage"] == "queued"
        assert status_response.json()["queue_position"] == 1

        cancelled = await client.delete(f"/v1/jobs/{job_id}", headers=AUTH)
        assert cancelled.status_code == 202
        assert cancelled.json() == {
            "job_id": job_id,
            "status": "cancelled",
            "stage": "queued",
            "cancellation_requested": True,
        }
        assert not audio_path.exists()
        assert await app.state.deps.ready_store.count_queued() == 0

        repeated = await client.delete(f"/v1/jobs/{job_id}", headers=AUTH)
        assert repeated.status_code == 200
        assert repeated.json()["status"] == "cancelled"
        assert repeated.json()["cancellation_requested"] is True


class BlockingEngine:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        model: str,
        diarize: bool,
        speakers: SpeakerBounds | None = None,
        include_speaker_review: bool = False,
        vocabulary: list[str] | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        report_stage(on_stage, JobStage.TRANSCRIBING)
        self.started.set()
        self.release.wait(timeout=5)
        self.finished.set()
        return make_result(diarize)

    def label_speakers(
        self,
        audio_path: Path,
        *,
        words,
        segments,
        language: str,
        model: str,
        speakers: SpeakerBounds | None = None,
        include_speaker_review: bool = False,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        report_stage(on_stage, JobStage.DIARIZING)
        self.started.set()
        self.release.wait(timeout=5)
        self.finished.set()
        return make_result().model_copy(update={"model": model})

    def warm_up(self) -> None:
        pass


def wait_until_missing(path: Path, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    return not path.exists()


async def test_running_cancellation_discards_late_result(settings: Settings):
    engine = BlockingEngine()
    async with api_client(settings, engine) as (client, app):
        response = await client.post(
            "/v1/jobs",
            files={"file": ("meeting.wav", make_wav_bytes(), "audio/wav")},
            headers=AUTH,
        )
        job_id = response.json()["job_id"]
        try:
            assert await asyncio.to_thread(engine.started.wait, 2)
            stored = await app.state.deps.ready_store.get(job_id)
            assert stored is not None and stored.audio_path is not None
            audio_path = Path(stored.audio_path)
            running = await client.get(f"/v1/jobs/{job_id}", headers=AUTH)
            assert running.json()["status"] == "running"
            assert running.json()["stage"] == "transcribing"

            cancelled = await client.delete(f"/v1/jobs/{job_id}", headers=AUTH)
            assert cancelled.status_code == 202
            assert cancelled.json()["status"] == "cancelled"
        finally:
            engine.release.set()

        assert await asyncio.to_thread(engine.finished.wait, 2)
        assert await asyncio.to_thread(wait_until_missing, audio_path)
        result = await client.get(f"/v1/jobs/{job_id}/result", headers=AUTH)
        assert result.status_code == 409
        assert result.json()["detail"]["status"] == "cancelled"
