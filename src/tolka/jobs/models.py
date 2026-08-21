from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
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
    client_id: str = "legacy"
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    request: JobRequest
    audio_path: str | None = None
    error: str | None = None
    attempt: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None


class WebhookOutboxEvent(BaseModel):
    id: str
    job_id: str
    url: str
    payload: dict[str, Any]
    attempt: int


def new_job(
    request: JobRequest, audio_path: str | None = None, *, client_id: str = "legacy"
) -> Job:
    now = datetime.now(UTC)
    return Job(
        id=uuid4().hex,
        client_id=client_id,
        status=JobStatus.QUEUED,
        created_at=now,
        updated_at=now,
        request=request,
        audio_path=audio_path,
    )
