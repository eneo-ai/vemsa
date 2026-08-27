"""Fully local engine (TOLKA_ENGINE=local): easytranscriber whisper + forced alignment
+ pyannote diarization, all in-process. For deployments with their own GPU and no
external whisper endpoint.

Everything ML-flavoured is imported lazily inside methods: this module is importable
without the `local` extra, but transcribe() requires it. Spots that could not be
verified without a GPU box are marked GPU-VERIFY(milestone-2).
"""

import logging
import threading
from pathlib import Path
from typing import Any

from tolka.config import Settings
from tolka.jobs.models import JobStage, Segment, SpeakerBounds, TranscriptionResult, Word
from tolka.pipeline.align import build_segment_aligner
from tolka.pipeline.base import StageReporter, report_stage
from tolka.pipeline.diarize import (
    Diarizer,
    assign_speakers,
    audio_duration,
    segments_without_speakers,
)
from tolka.pipeline.label import label_speakers
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
        from easytranscriber.pipelines import pipeline

        # GPU-VERIFY(milestone-2): confirm easytranscriber accepts language=None for
        # whisper auto-detect; otherwise require an explicit language in the API.
        lang = None if language == "auto" else language

        report_stage(on_stage, JobStage.TRANSCRIBING)
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
        duration = audio_duration(audio_path, fallback=words[-1].end if words else 0.0)

        if diarize:
            report_stage(on_stage, JobStage.DIARIZING)
            turns = self._diarizer.diarize(audio_path, speakers=speakers)
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
            # easytranscriber's word timestamps come from its own forced alignment
            alignment="forced",
        )

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
        logger.info("warming up diarization pipeline")
        self._diarizer.load()
        # GPU-VERIFY(milestone-2): also pre-download the whisper + emissions models
        # (first pipeline() call does it today, making the first job slow).
