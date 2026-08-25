"""Hybrid engine: remote whisper produces the transcript (the GPU-heavy stage,
offloaded), then the text is force-aligned locally with easyaligner (wav2vec2 CTC
emissions + Viterbi — far lighter than whisper inference) for precise word-level
timestamps. Diarization runs locally via pyannote.

If the alignment stack is not installed or alignment fails at runtime, the engine
degrades to the provider's own timestamps (same behavior as TOLKA_ENGINE=remote),
so offloading never hard-fails on a machine without the align extra."""

import logging
import threading
from pathlib import Path
from typing import Any

from tolka.config import Settings
from tolka.jobs.models import Segment, TranscriptionResult, Word
from tolka.pipeline.diarize import Diarizer, resolve_segments
from tolka.pipeline.label import label_speakers
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
        self._lock = threading.Lock()

    def transcribe(
        self, audio_path: Path, *, language: str, model: str, diarize: bool
    ) -> TranscriptionResult:
        payload = request_transcription(self._settings, audio_path, language=language, model=model)
        provider_words, plain_segments = parse_verbose_json(payload)
        if not provider_words and not plain_segments:
            raise RuntimeError("whisper API returned neither words nor segments")

        words = provider_words
        try:
            aligned = self._force_align(audio_path, payload, plain_segments, language)
            if aligned:
                words = aligned
        except Exception:
            logger.warning(
                "local forced alignment unavailable or failed; falling back to provider"
                " timestamps (install the 'align' extra for word-precise output)",
                exc_info=True,
            )

        turns = self._diarizer.diarize(audio_path) if diarize else None
        segments = resolve_segments(words, plain_segments if words is provider_words else [], turns)
        return build_result(payload, segments, model=model, language=language)

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

    def _force_align(
        self,
        audio_path: Path,
        payload: dict[str, Any],
        plain_segments: list[Segment],
        language: str,
    ) -> list[Word]:
        """Align the provider's transcript against the audio with easyaligner.

        GPU-VERIFY(milestone-2): easyaligner's exact API surface (SpeechSegment
        construction, pipeline parameters, Swedish tokenizer availability) must be
        validated against a real installation; this is a best-effort invocation kept
        behind the runtime fallback above."""
        from easyaligner.pipelines import pipeline as align_pipeline

        transcript_segments = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in plain_segments
        ] or [{"start": 0.0, "end": 0.0, "text": str(payload.get("text", "")).strip()}]

        with self._lock:
            aligned = align_pipeline(
                audio_paths=[str(audio_path)],
                transcripts=[transcript_segments],
                emissions_model=self._settings.emissions_model,
                language=None if language == "auto" else language,
                cache_dir=str(self._settings.model_cache_dir),
                save_json=False,
                return_alignments=True,
            )

        words: list[Word] = []
        for segment in aligned[0]:
            nested = getattr(segment, "words", None) or [segment]
            for item in nested:
                text = getattr(item, "word", None) or getattr(item, "text", None)
                start = getattr(item, "start", None)
                end = getattr(item, "end", None)
                if text is None or start is None or end is None:
                    raise ValueError(f"unrecognized alignment shape: {item!r}")
                words.append(Word(word=str(text).strip(), start=float(start), end=float(end)))
        words.sort(key=lambda word: word.start)
        return words
