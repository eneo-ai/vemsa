from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, model_validator

Language = Literal["sv", "en", "auto"]
# transcribe: run the engine end to end. diarize: the caller supplies the transcript
# (word timestamps in seconds from the start of the audio); Tolka adds speaker labels.
JobTask = Literal["transcribe", "diarize"]
EXTERNAL_MODEL = "external"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(StrEnum):
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    DIARIZING = "diarizing"
    FINALIZING = "finalizing"


class Word(BaseModel):
    word: str
    start: float
    end: float
    # Confidence for this word when the source reports one. Semantics follow the
    # result's `alignment` field: provider_words = the ASR decoder's posterior
    # probability, forced = the CTC forced-alignment score. Not comparable
    # across rungs; None when the source reports no confidence.
    probability: float | None = None


class Segment(BaseModel):
    start: float
    end: float
    speaker: str | None = None
    text: str
    words: list[Word] = Field(default_factory=list)


# How the word timestamps behind the speaker labels were obtained, best to worst:
# the caller/provider supplied words, Tolka force-aligned the text locally, segments
# were split proportionally at turn boundaries, or whole segments were labelled.
Alignment = Literal["provider_words", "forced", "segment_split", "segment_only"]

# Quality ranking of the rungs (highest is best): audio-derived forced alignment,
# then provider-measured words, then the two segment-level approximations. Used by
# the TOLKA_MIN_ALIGNMENT floor; a missing rung ranks below every floor.
ALIGNMENT_RANK: dict[Alignment, int] = {
    "forced": 3,
    "provider_words": 2,
    "segment_split": 1,
    "segment_only": 0,
}


class TranscriptionResult(BaseModel):
    language: str
    duration_seconds: float
    model: str
    text: str
    segments: list[Segment]
    alignment: Alignment | None = None


class SpeakerBounds(BaseModel):
    """Speaker-count prior for diarization, passed through to pyannote."""

    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    def pipeline_kwargs(self) -> dict[str, int]:
        return {key: value for key, value in self.model_dump().items() if value is not None}


class JobRequest(BaseModel):
    task: JobTask = "transcribe"
    source_url: HttpUrl | None = None
    language: Language = "auto"
    model: str | None = None
    diarize: bool = True
    webhook_url: HttpUrl | None = None
    # task=diarize only: the externally produced transcript to label. Word-level
    # timestamps give word-precise speaker changes; segments alone fall back to
    # whole-segment labelling.
    words: list[Word] | None = None
    segments: list[Segment] | None = None
    # Speaker-count prior for diarization: exact count, or an expected range.
    # num_speakers excludes the other two.
    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_task(self) -> "JobRequest":
        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError("num_speakers excludes min_speakers and max_speakers")
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers cannot exceed max_speakers")
        if not self.diarize and self.speaker_bounds() is not None:
            raise ValueError("speaker bounds require diarize=true")
        if self.task == "transcribe":
            if self.words is not None or self.segments is not None:
                raise ValueError("words and segments are only accepted for task=diarize")
            return self
        if not self.words and not self.segments:
            raise ValueError("task=diarize requires a non-empty words or segments list")
        if not self.diarize:
            raise ValueError("task=diarize cannot set diarize=false")
        spans = [(word.start, word.end) for word in self.words or []] + [
            (segment.start, segment.end) for segment in self.segments or []
        ]
        for start, end in spans:
            if start < 0 or end < start:
                raise ValueError("transcript timestamps must satisfy 0 <= start <= end")
        return self

    def speaker_bounds(self) -> SpeakerBounds | None:
        if self.num_speakers is None and self.min_speakers is None and self.max_speakers is None:
            return None
        return SpeakerBounds(
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

    def transcript_bytes(self) -> int:
        """Serialized size of the supplied transcript, for the admission size cap."""
        if self.words is None and self.segments is None:
            return 0
        return len(self.model_dump_json(include={"words", "segments"}).encode())


class Job(BaseModel):
    id: str
    client_id: str = "legacy"
    status: JobStatus
    stage: JobStage = JobStage.QUEUED
    created_at: datetime
    updated_at: datetime
    request: JobRequest
    audio_path: str | None = None
    error: str | None = None
    attempt: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    cancellation_requested_at: datetime | None = None


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
