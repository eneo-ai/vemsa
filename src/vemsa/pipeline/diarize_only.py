"""Diarization-only tier (VEMSA_ENGINE=diarize): no ASR engine at all.

For deployments where the consumer transcribes with its own models and only needs
speaker labels. transcribe jobs are refused at submission on this tier. Segment-only
transcripts are still force-aligned locally when the `align` extra is installed."""

from pathlib import Path

from vemsa.config import Settings
from vemsa.jobs.models import JobStage, Segment, SpeakerBounds, TranscriptionResult, Word
from vemsa.pipeline.align import build_segment_aligner
from vemsa.pipeline.base import StageReporter, report_stage
from vemsa.pipeline.diarize import Diarizer
from vemsa.pipeline.label import SpeakerDiarizer, label_speakers


class DiarizeOnlyEngine:
    def __init__(self, settings: Settings, diarizer: SpeakerDiarizer | None = None) -> None:
        self._diarizer = diarizer or Diarizer(settings)
        self._segment_aligner = build_segment_aligner(settings)
        self._prefer_align = settings.diarize_prefer_align
        self._tuning = settings.attribution_tuning()

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
        raise RuntimeError("this deployment only labels speakers (VEMSA_ENGINE=diarize)")

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
            prefer_alignment=self._prefer_align,
            tuning=self._tuning,
        )

    def warm_up(self) -> None:
        self._diarizer.load()
