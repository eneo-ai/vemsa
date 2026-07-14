from pathlib import Path
from typing import Protocol

from tolka.jobs.models import TranscriptionResult


class TranscriptionEngine(Protocol):
    """Blocking transcription pipeline; called from a worker thread."""

    def transcribe(
        self, audio_path: Path, *, language: str, model: str, diarize: bool
    ) -> TranscriptionResult: ...

    def warm_up(self) -> None: ...
