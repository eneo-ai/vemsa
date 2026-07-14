import asyncio
from pathlib import Path

import pytest

from tolka.config import Settings
from tolka.jobs.models import Job, JobStatus, Segment, TranscriptionResult, Word
from tolka.jobs.store import SqliteJobStore


def make_result(diarize: bool = True) -> TranscriptionResult:
    speakers = ["SPEAKER_00", "SPEAKER_01"] if diarize else [None, None]
    segments = [
        Segment(
            start=0.0,
            end=1.4,
            speaker=speakers[0],
            text="hej och välkomna",
            words=[
                Word(word="hej", start=0.0, end=0.4),
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
        self, audio_path: Path, *, language: str, model: str, diarize: bool
    ) -> TranscriptionResult:
        self.calls.append(
            {"audio_path": audio_path, "language": language, "model": model, "diarize": diarize}
        )
        return make_result(diarize)

    def warm_up(self) -> None:
        pass


class FailingEngine:
    def transcribe(
        self, audio_path: Path, *, language: str, model: str, diarize: bool
    ) -> TranscriptionResult:
        raise RuntimeError("pipeline exploded")

    def warm_up(self) -> None:
        pass


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        api_tokens="secret-token",
        db_path=tmp_path / "tolka.sqlite3",
        work_dir=tmp_path / "work",
        model_cache_dir=tmp_path / "models",
        purge_interval_s=3600.0,
        allow_private_urls=True,
        fake_engine=True,
    )


@pytest.fixture
async def store(settings: Settings):
    store = SqliteJobStore(settings.db_path)
    await store.open()
    yield store
    await store.close()


async def wait_for_status(
    store: SqliteJobStore, job_id: str, status: JobStatus, timeout: float = 5.0
) -> Job:
    async with asyncio.timeout(timeout):
        while True:
            job = await store.get(job_id)
            if job is not None and job.status == status:
                return job
            await asyncio.sleep(0.01)
