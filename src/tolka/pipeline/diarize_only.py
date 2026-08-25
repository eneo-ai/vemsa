"""Diarization-only tier (TOLKA_ENGINE=diarize): no ASR engine at all.

For deployments where the consumer transcribes with its own models and only needs
speaker labels. transcribe jobs are refused at submission on this tier."""

from pathlib import Path

from tolka.config import Settings
from tolka.jobs.models import Segment, TranscriptionResult, Word
from tolka.pipeline.diarize import Diarizer
from tolka.pipeline.label import SpeakerDiarizer, label_speakers


class DiarizeOnlyEngine:
    def __init__(self, settings: Settings, diarizer: SpeakerDiarizer | None = None) -> None:
        self._diarizer = diarizer or Diarizer(settings)

    def transcribe(
        self, audio_path: Path, *, language: str, model: str, diarize: bool
    ) -> TranscriptionResult:
        raise RuntimeError("this deployment only labels speakers (TOLKA_ENGINE=diarize)")

    def label_speakers(
        self,
        audio_path: Path,
        *,
        words: list[Word],
        segments: list[Segment],
        language: str,
        model: str,
    ) -> TranscriptionResult:
        return label_speakers(
            self._diarizer,
            audio_path,
            words=words,
            segments=segments,
            language=language,
            model=model,
        )

    def warm_up(self) -> None:
        self._diarizer.load()
