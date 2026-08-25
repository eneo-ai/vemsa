"""task=diarize: speaker labels for an externally produced transcript."""

import asyncio
import contextlib
import json
from pathlib import Path

import httpx
import respx
from httpx import Response

from conftest import FakeEngine
from tolka.config import Settings
from tolka.jobs.models import Segment, Word
from tolka.main import create_app
from tolka.pipeline.diarize import Turn
from tolka.pipeline.fake import CannedEngine
from tolka.pipeline.label import label_speakers
from tolka.pipeline.whisper_api import OpenAIWhisperEngine

AUTH = {"Authorization": "Bearer secret-token"}
WORDS = [
    {"word": "hej", "start": 0.0, "end": 0.4},
    {"word": "och", "start": 0.5, "end": 0.7},
    {"word": "tack", "start": 2.0, "end": 2.2},
]


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


def multipart(**fields: str) -> dict:
    return {"files": {"file": ("a.wav", b"fake audio")}, "data": fields}


async def submit(client: httpx.AsyncClient, **fields: str) -> httpx.Response:
    return await client.post("/v1/jobs", headers=AUTH, **multipart(**fields))


async def test_multipart_diarize_lifecycle_echoes_external_model(settings: Settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        response = await submit(client, task="diarize", language="sv", words=json.dumps(WORDS))
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        await poll_until(client, job_id, "completed")

        result = (await client.get(f"/v1/jobs/{job_id}/result", headers=AUTH)).json()
        assert result["model"] == "external"

    call = engine.calls[-1]
    assert call["task"] == "diarize"
    assert [w.word for w in call["words"]] == ["hej", "och", "tack"]
    assert call["model"] == "external"
    assert call["language"] == "sv"


@respx.mock
async def test_json_submission_with_source_url(settings: Settings):
    respx.get("https://example.org/m.mp3").mock(return_value=Response(200, content=b"audio"))
    async with api_client(settings) as (client, _):
        response = await client.post(
            "/v1/jobs",
            json={
                "task": "diarize",
                "source_url": "https://example.org/m.mp3",
                "segments": [{"start": 0.0, "end": 1.4, "text": "hej och välkomna"}],
            },
            headers=AUTH,
        )
        assert response.status_code == 202
        await poll_until(client, response.json()["job_id"], "completed")


async def test_submission_validation_failures(settings: Settings):
    async with api_client(settings) as (client, _):
        # diarize without any transcript
        assert (await submit(client, task="diarize")).status_code == 422
        # transcribe with transcript fields
        assert (await submit(client, words=json.dumps(WORDS))).status_code == 422
        # diarize=false contradiction
        assert (
            await submit(client, task="diarize", words=json.dumps(WORDS), diarize="false")
        ).status_code == 422
        # malformed JSON part
        response = await submit(client, task="diarize", words="[{broken")
        assert response.status_code == 422
        assert "not valid JSON" in response.json()["detail"]
        # negative / inverted timestamps
        bad = [{"word": "x", "start": -1.0, "end": 0.5}]
        assert (await submit(client, task="diarize", words=json.dumps(bad))).status_code == 422
        inverted = [{"word": "x", "start": 2.0, "end": 1.0}]
        assert (await submit(client, task="diarize", words=json.dumps(inverted))).status_code == 422


async def test_oversized_transcript_is_413(settings: Settings):
    settings.max_transcript_bytes = 200
    async with api_client(settings) as (client, _):
        # over our cap but under the raised starlette part limit: our contract 413
        words = [{"word": f"w{i}", "start": float(i), "end": i + 0.5} for i in range(20)]
        response = await submit(client, task="diarize", words=json.dumps(words))
        assert response.status_code == 413
        assert "transcript exceeds" in response.json()["detail"]
        # big enough that starlette's part parser trips first: still a 413
        huge = "x" * (200 + 65536 + 1024)
        response = await submit(client, task="diarize", words=huge)
        assert response.status_code == 413


async def test_diarize_tier_refuses_transcribe_and_is_ready(settings: Settings):
    settings.engine = "diarize"
    async with api_client(settings, engine=None) as (client, _):
        ready = await client.get("/readyz")
        assert ready.status_code == 200 and ready.json()["status"] == "ready"
        response = await submit(client, language="sv")
        assert response.status_code == 422
        assert "task=diarize" in response.json()["detail"]


async def test_fake_engine_alternates_speakers(tmp_path: Path):
    engine = CannedEngine()
    result = engine.label_speakers(
        tmp_path / "a.wav",
        words=[],
        segments=[
            Segment(start=0.0, end=1.0, text="hej"),
            Segment(start=2.5, end=3.0, text="tack"),
        ],
        language="sv",
        model="external",
    )
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.model == "external"
    assert result.language == "sv"
    assert "SPEAKER_00" in result.text and "SPEAKER_01" in result.text


class FakeDiarizer:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def diarize(self, audio_path: Path) -> list[Turn]:
        self.calls.append(audio_path)
        return [Turn(0.0, 1.5, "SPEAKER_00"), Turn(1.9, 2.7, "SPEAKER_01")]

    def load(self) -> None:
        pass


def test_label_speakers_word_level_merge(tmp_path: Path):
    words = [Word(**w) for w in WORDS]
    result = label_speakers(
        FakeDiarizer(),
        tmp_path / "missing.wav",
        words=words,
        segments=[],
        language="auto",
        model="external",
    )
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.language == "unknown"
    # audio unreadable: duration falls back to the last labelled end
    assert result.duration_seconds == 2.2
    assert result.text.count("\n") == 1


def test_label_speakers_reuses_the_engine_diarizer(settings: Settings, tmp_path: Path):
    settings.whisper_api_base = "http://whisper.local/v1"
    diarizer = FakeDiarizer()
    engine = OpenAIWhisperEngine(settings, diarizer=diarizer)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake audio")
    engine.label_speakers(
        audio, words=[Word(**w) for w in WORDS], segments=[], language="sv", model="external"
    )
    assert diarizer.calls == [audio]
