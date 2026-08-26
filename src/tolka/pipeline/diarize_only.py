"""Diarization-only tier (TOLKA_ENGINE=diarize): no ASR engine at all.

For deployments where the consumer transcribes with its own models and only needs
speaker labels. transcribe jobs are refused at submission on this tier. Segment-only
transcripts are still force-aligned locally when the `align` extra is installed."""

from pathlib import Path

from tolka.config import Settings
from tolka.jobs.models import JobStage, Segment, SpeakerBounds, TranscriptionResult, Word
from tolka.pipeline.align import build_segment_aligner
from tolka.pipeline.base import StageReporter, report_stage
from tolka.pipeline.diarize import Diarizer
from tolka.pipeline.label import SpeakerDiarizer, label_speakers


class DiarizeOnlyEngine:
    def __init__(self, settings: Settings, diarizer: SpeakerDiarizer | None = None) -> None:
        self._diarizer = diarizer or Diarizer(settings)
        self._segment_aligner = build_segment_aligner(settings)

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        model: str,
        diarize: bool,
        speakers: SpeakerBounds | None = None,
        on_stage: StageReporter | None = None,
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
        speakers: SpeakerBounds | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        if not words:
            report_stage(on_stage, JobStage.ALIGNING)
        report_stage(on_stage, JobStage.DIARIZING)
        return label_speakers(
            self._diarizer,
            audio_path,
            words=words,
            segments=segments,
            language=language,
            model=model,
            aligner=self._segment_aligner,
            speakers=speakers,
        )

    def warm_up(self) -> None:
        self._diarizer.load()
