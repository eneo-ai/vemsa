"""Hybrid engine: remote whisper produces the transcript (the GPU-heavy stage,
offloaded), then the text is force-aligned locally with easyaligner (wav2vec2 CTC
emissions + Viterbi — far lighter than whisper inference) for precise word-level
timestamps. Diarization runs locally via pyannote.

Forced alignment is mandatory on this tier: an alignment failure fails the job
loudly instead of degrading to the provider's decoder timestamps — the service
always runs with a GPU and the align extra, and quality beats completing coarsely.
Deployments that deliberately trust a provider's timestamps use VEMSA_ENGINE=remote."""

import logging
from pathlib import Path
from typing import Any

from vemsa.config import Settings
from vemsa.jobs.models import JobStage, Segment, SpeakerBounds, TranscriptionResult, Word
from vemsa.pipeline.align import build_segment_aligner, force_align_segments
from vemsa.pipeline.base import StageReporter, report_stage
from vemsa.pipeline.diarize import Diarizer, resolve_segments
from vemsa.pipeline.label import alignment_input, label_speakers, words_plausible
from vemsa.pipeline.whisper_api import build_result, parse_verbose_json, request_transcription

logger = logging.getLogger(__name__)


class HybridEngine:
    """Remote whisper text + local forced alignment; runs in a worker thread."""

    def __init__(self, settings: Settings, diarizer: Diarizer | None = None) -> None:
        if not settings.whisper_api_base:
            raise ValueError(
                "VEMSA_WHISPER_API_BASE is required for the hybrid engine (or set"
                " VEMSA_ENGINE=local to run whisper in-process)"
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
        vocabulary: list[str] | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        report_stage(on_stage, JobStage.TRANSCRIBING)
        payload = request_transcription(
            self._settings, audio_path, language=language, model=model, vocabulary=vocabulary
        )
        provider_words, plain_segments = parse_verbose_json(payload)
        if not provider_words and not plain_segments:
            raise RuntimeError("whisper API returned neither words nor segments")

        # provider words are never used as timestamps on this tier; they only
        # judge whether the provider's segment windows can anchor the alignment
        # (windows from an implausible timeline cannot — align the whole audio)
        windows_trusted = not (provider_words and not words_plausible(provider_words))
        if not windows_trusted:
            logger.warning(
                "provider word timestamps are implausible; aligning the whole audio in one window"
            )
        align_segments = alignment_input(
            plain_segments,
            windows_trusted=windows_trusted,
            total_duration=float(payload.get("duration") or 0.0)
            or max((segment.end for segment in plain_segments), default=0.0),
        )
        report_stage(on_stage, JobStage.ALIGNING)
        words = self._force_align(audio_path, payload, align_segments, language)
        if not words:
            raise RuntimeError("forced alignment produced no words for the provider transcript")

        if diarize:
            report_stage(on_stage, JobStage.DIARIZING)
        turns = self._diarizer.diarize(audio_path, speakers=speakers) if diarize else None
        segments = resolve_segments(words, [], turns, tuning=self._settings.attribution_tuning())
        return build_result(payload, segments, model=model, language=language, alignment="forced")

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
            prefer_alignment=self._settings.diarize_prefer_align,
            tuning=self._settings.attribution_tuning(),
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
