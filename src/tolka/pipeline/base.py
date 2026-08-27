from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeAlias

from tolka.jobs.models import JobStage, Segment, SpeakerBounds, TranscriptionResult, Word

StageReporter: TypeAlias = Callable[[JobStage], None]


def report_stage(reporter: StageReporter | None, stage: JobStage) -> None:
    if reporter is not None:
        reporter(stage)


class TranscriptionEngine(Protocol):
    """Blocking transcription pipeline; called from a worker thread."""

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        model: str,
        diarize: bool,
        speakers: SpeakerBounds | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult: ...

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
        """Diarize the audio and attach speakers to a transcript produced elsewhere."""
        ...

    def warm_up(self) -> None: ...
