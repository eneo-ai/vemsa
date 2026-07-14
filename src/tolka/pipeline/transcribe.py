"""Real transcription engine wrapping easytranscriber + pyannote diarization.

Everything ML-flavoured is imported lazily inside methods: this module is importable
without the `ml` extra, but transcribe() requires it. Spots that could not be verified
without a GPU box are marked GPU-VERIFY(milestone-2).
"""

import logging
import threading
from pathlib import Path
from typing import Any

from tolka.config import Settings
from tolka.jobs.models import TranscriptionResult, Word
from tolka.pipeline.diarize import Diarizer, assign_speakers, segments_without_speakers
from tolka.pipeline.render import render_text

logger = logging.getLogger(__name__)


def words_from_alignments(aligned_segments: list[Any]) -> list[Word]:
    """Normalize easytranscriber's per-file alignment output into flat Word lists.

    Handles both shapes we may get back: segments carrying a word list attribute, and
    flat word-level segments. GPU-VERIFY(milestone-2): confirm the exact SpeechSegment
    field names against real pipeline output and drop the fallbacks.
    """
    words: list[Word] = []
    for segment in aligned_segments:
        nested = getattr(segment, "words", None) or getattr(segment, "word_alignments", None)
        for item in nested if nested is not None else [segment]:
            text = getattr(item, "word", None) or getattr(item, "text", None)
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            if text is None or start is None or end is None:
                raise ValueError(f"unrecognized alignment shape: {item!r}")
            words.append(Word(word=str(text).strip(), start=float(start), end=float(end)))
    return sorted(words, key=lambda word: word.start)


class EasyTranscriberEngine:
    """Blocking easytranscriber pipeline + pyannote diarization; runs in a worker thread."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._diarizer = Diarizer(settings)

    def transcribe(
        self, audio_path: Path, *, language: str, model: str, diarize: bool
    ) -> TranscriptionResult:
        from easytranscriber.pipelines import pipeline

        # GPU-VERIFY(milestone-2): confirm easytranscriber accepts language=None for
        # whisper auto-detect; otherwise require an explicit language in the API.
        lang = None if language == "auto" else language

        with self._lock:
            # GPU-VERIFY(milestone-2): model instances are cached on disk via cache_dir,
            # but confirm in-memory reuse across pipeline() calls; if models reload per
            # call, hold them here instead.
            #
            # GPU-VERIFY(milestone-2): tokenizer for forced alignment — easyaligner's
            # load_tokenizer() Swedish support is unverified; the emissions model default
            # is KBLab/wav2vec2-large-voxrex-swedish (see Settings.emissions_model).
            aligned = pipeline(
                transcription_model=model,
                emissions_model=self._settings.emissions_model,
                audio_paths=[str(audio_path)],
                language=lang,
                cache_dir=str(self._settings.model_cache_dir),
                save_json=False,
                return_alignments=True,
            )

        words = words_from_alignments(aligned[0])
        duration = self._audio_duration(audio_path, words)

        if diarize:
            turns = self._diarizer.diarize(audio_path)
            segments = assign_speakers(words, turns)
        else:
            segments = segments_without_speakers(words)

        return TranscriptionResult(
            # GPU-VERIFY(milestone-2): surface whisper's detected language when
            # auto-detect was used, instead of echoing the request.
            language=lang or "auto",
            duration_seconds=duration,
            model=model,
            text=render_text(segments),
            segments=segments,
        )

    def warm_up(self) -> None:
        logger.info("warming up diarization pipeline")
        self._diarizer.load()
        # GPU-VERIFY(milestone-2): also pre-download the whisper + emissions models
        # (first pipeline() call does it today, making the first job slow).

    @staticmethod
    def _audio_duration(audio_path: Path, words: list[Word]) -> float:
        try:
            import soundfile

            info = soundfile.info(str(audio_path))
            return float(info.frames) / info.samplerate
        except Exception:
            return words[-1].end if words else 0.0
