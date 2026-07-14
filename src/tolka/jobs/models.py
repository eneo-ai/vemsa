from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, HttpUrl

Language = Literal["sv", "en", "auto"]


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Word(BaseModel):
    word: str
    start: float
    end: float


class Segment(BaseModel):
    start: float
    end: float
    speaker: str | None = None
    text: str
    words: list[Word]


class TranscriptionResult(BaseModel):
    language: str
    duration_seconds: float
    model: str
    text: str
    segments: list[Segment]


class JobRequest(BaseModel):
    source_url: HttpUrl | None = None
    language: Language = "auto"
    model: str | None = None
    diarize: bool = True
    webhook_url: HttpUrl | None = None


class Job(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    request: JobRequest
    audio_path: str | None = None
    error: str | None = None


def new_job(request: JobRequest, audio_path: str | None = None) -> Job:
    now = datetime.now(UTC)
    return Job(
        id=uuid4().hex,
        status=JobStatus.QUEUED,
        created_at=now,
        updated_at=now,
        request=request,
        audio_path=audio_path,
    )
