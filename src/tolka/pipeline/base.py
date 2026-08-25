from pathlib import Path
from typing import Protocol

from tolka.jobs.models import Segment, SpeakerBounds, TranscriptionResult, Word


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
    ) -> TranscriptionResult:
        """Diarize the audio and attach speakers to a transcript produced elsewhere."""
        ...

    def warm_up(self) -> None: ...
