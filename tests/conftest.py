import asyncio
import io
import os
import wave
from pathlib import Path

import pytest

from tolka.config import Settings
from tolka.jobs.models import (
    Job,
    JobStage,
    JobStatus,
    Segment,
    SpeakerBounds,
    TranscriptionResult,
    Word,
)
from tolka.jobs.postgres_store import PostgresJobStore
from tolka.jobs.store import JobStore
from tolka.pipeline.base import StageReporter, report_stage

TEST_DATABASE_URL = os.getenv("TOLKA_TEST_POSTGRES_URL")


def make_wav_bytes(duration_ms: int = 50) -> bytes:
    """Generate a tiny valid mono PCM WAV without checking in an audio fixture."""
    buffer = io.BytesIO()
    sample_rate = 8_000
    frame_count = sample_rate * duration_ms // 1_000
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


def make_result(diarize: bool = True) -> TranscriptionResult:
    speakers = ["SPEAKER_00", "SPEAKER_01"] if diarize else [None, None]
    segments = [
        Segment(
            start=0.0,
            end=1.4,
            speaker=speakers[0],
            text="hej och välkomna",
            words=[
                Word(word="hej", start=0.0, end=0.4, probability=0.98),
                Word(word="och", start=0.5, end=0.7),
                Word(word="välkomna", start=0.8, end=1.4),
            ],
        ),
        Segment(
            start=2.0,
            end=2.6,
            speaker=speakers[1],
            text="tack så mycket",
            words=[
                Word(word="tack", start=2.0, end=2.2),
                Word(word="så", start=2.3, end=2.4),
                Word(word="mycket", start=2.45, end=2.6),
            ],
        ),
    ]
    return TranscriptionResult(
        language="sv",
        duration_seconds=2.6,
        model="KBLab/kb-whisper-large",
        text="hej och välkomna\ntack så mycket",
        segments=segments,
    )


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        model: str,
        diarize: bool,
        speakers: SpeakerBounds | None = None,
        vocabulary: list[str] | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        report_stage(on_stage, JobStage.TRANSCRIBING)
        self.calls.append(
            {
                "audio_path": audio_path,
                "language": language,
                "model": model,
                "diarize": diarize,
                "speakers": speakers,
                "vocabulary": vocabulary,
            }
        )
        return make_result(diarize)

    def label_speakers(
        self,
        audio_path: Path,
        *,
        words: list[Word],
        segments: list[Segment],
        language: str,
        model: str,
        speakers: SpeakerBounds | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        report_stage(on_stage, JobStage.DIARIZING)
        self.calls.append(
            {
                "task": "diarize",
                "audio_path": audio_path,
                "words": words,
                "segments": segments,
                "language": language,
                "model": model,
                "speakers": speakers,
            }
        )
        result = make_result(True)
        return result.model_copy(update={"model": model})

    def warm_up(self) -> None:
        pass


class FailingEngine:
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        model: str,
        diarize: bool,
        speakers: SpeakerBounds | None = None,
        vocabulary: list[str] | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        raise RuntimeError("pipeline exploded")

    def label_speakers(
        self,
        audio_path: Path,
        *,
        words: list[Word],
        segments: list[Segment],
        language: str,
        model: str,
        speakers: SpeakerBounds | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        raise RuntimeError("pipeline exploded")

    def warm_up(self) -> None:
        pass


@pytest.fixture
async def settings(tmp_path: Path) -> Settings:
    if not TEST_DATABASE_URL:
        pytest.skip("TOLKA_TEST_POSTGRES_URL is not configured")
    settings = Settings(
        _env_file=None,
        api_tokens="secret-token",
        database_url=TEST_DATABASE_URL,
        work_dir=tmp_path / "work",
        model_cache_dir=tmp_path / "models",
        purge_interval_s=3600.0,
        allow_private_urls=True,
        engine="fake",
    )
    # Every test starts from an empty job store (open() also runs migrations).
    store = PostgresJobStore(TEST_DATABASE_URL)
    await store.open()
    await store.pool.execute("TRUNCATE webhook_outbox, jobs, worker_heartbeats")
    await store.close()
    return settings


@pytest.fixture
async def store(settings: Settings):
    store = PostgresJobStore(settings.database_url)
    await store.open()
    yield store
    await store.close()


async def wait_for_status(
    store: JobStore, job_id: str, status: JobStatus, timeout: float = 5.0
) -> Job:
    async with asyncio.timeout(timeout):
        while True:
            job = await store.get(job_id)
            if job is not None and job.status == status:
                return job
            await asyncio.sleep(0.01)
