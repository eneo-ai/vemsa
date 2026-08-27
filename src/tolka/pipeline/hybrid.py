"""Hybrid engine: remote whisper produces the transcript (the GPU-heavy stage,
offloaded), then the text is force-aligned locally with easyaligner (wav2vec2 CTC
emissions + Viterbi — far lighter than whisper inference) for precise word-level
timestamps. Diarization runs locally via pyannote.

If the alignment stack is not installed or alignment fails at runtime, the engine
degrades to the provider's own timestamps (same behavior as TOLKA_ENGINE=remote),
so offloading never hard-fails on a machine without the align extra."""

import logging
from pathlib import Path
from typing import Any

from tolka.config import Settings
from tolka.jobs.models import Alignment, JobStage, Segment, SpeakerBounds, TranscriptionResult, Word
from tolka.pipeline.align import build_segment_aligner, force_align_segments
from tolka.pipeline.base import StageReporter, report_stage
from tolka.pipeline.diarize import Diarizer, resolve_segments
from tolka.pipeline.label import (
    alignment_input,
    label_speakers,
    segment_merge_alignment,
    words_plausible,
)
from tolka.pipeline.whisper_api import build_result, parse_verbose_json, request_transcription

logger = logging.getLogger(__name__)


class HybridEngine:
    """Remote whisper text + local forced alignment; runs in a worker thread."""

    def __init__(self, settings: Settings, diarizer: Diarizer | None = None) -> None:
        if not settings.whisper_api_base:
            raise ValueError(
                "TOLKA_WHISPER_API_BASE is required for the hybrid engine (or set"
                " TOLKA_ENGINE=local to run whisper in-process)"
            )
        self._settings = settings
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
        report_stage(on_stage, JobStage.TRANSCRIBING)
        payload = request_transcription(self._settings, audio_path, language=language, model=model)
        provider_words, plain_segments = parse_verbose_json(payload)
        if not provider_words and not plain_segments:
            raise RuntimeError("whisper API returned neither words nor segments")

        words = provider_words
        if words and plain_segments and not words_plausible(words):
            logger.warning(
                "provider word timestamps are implausible; preferring local alignment"
                " or segment-level merging"
            )
            words = []
        alignment: Alignment | None = "provider_words" if words else None
        # segment windows from the same timeline as discarded words cannot anchor
        # the alignment; align the whole audio in one window instead
        align_segments = alignment_input(
            plain_segments,
            windows_trusted=not (provider_words and not words),
            total_duration=float(payload.get("duration") or 0.0)
            or max((segment.end for segment in plain_segments), default=0.0),
        )
        aligned: list[Word] = []
        report_stage(on_stage, JobStage.ALIGNING)
        try:
            aligned = self._force_align(audio_path, payload, align_segments, language)
        except Exception:
            logger.warning(
                "local forced alignment unavailable or failed; falling back to provider"
                " timestamps (install the 'align' extra for word-precise output)",
                exc_info=True,
            )
        if aligned:
            words = aligned
            alignment = "forced"

        if diarize:
            report_stage(on_stage, JobStage.DIARIZING)
        turns = self._diarizer.diarize(audio_path, speakers=speakers) if diarize else None
        segments = resolve_segments(words, [] if aligned else plain_segments, turns)
        if alignment is None:
            alignment = segment_merge_alignment(plain_segments, segments)
        return build_result(payload, segments, model=model, language=language, alignment=alignment)

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

    def _force_align(
        self,
        audio_path: Path,
        payload: dict[str, Any],
        plain_segments: list[Segment],
        language: str,
    ) -> list[Word]:
        """Align the provider's transcript against the audio (shared helper in
        pipeline/align.py); kept as a method so tests can stub it per engine."""
        return force_align_segments(
            self._settings,
            audio_path,
            plain_segments,
            language,
            fallback_text=str(payload.get("text", "")),
        )
