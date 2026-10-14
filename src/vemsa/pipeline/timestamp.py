"""task=timestamp: word timestamps for a caller transcript, without speakers.

The consumer transcribed the audio itself and measured only coarse windows (one
segment per transcription chunk). It wants every word's time, to play from a word
and follow the audio, but no speaker identification. The text is force-aligned
against the audio by the same aligner and window-trust rule task=diarize uses
(implausible windows give way to one whole-audio window), and the words come back
as reader lines grouped like the diarized output, speaker left empty. No ASR, no
diarization: one alignment pass, and a failed or empty alignment fails the job."""

import logging
from pathlib import Path

from vemsa.jobs.models import EXTERNAL_MODEL, Segment, TranscriptionResult
from vemsa.pipeline.align import SegmentAligner
from vemsa.pipeline.diarize import AttributionTuning, audio_duration, segments_without_speakers
from vemsa.pipeline.label import alignment_input
from vemsa.pipeline.render import render_text

logger = logging.getLogger(__name__)


def timestamp_transcript(
    aligner: SegmentAligner,
    audio_path: Path,
    *,
    segments: list[Segment],
    language: str,
    model: str = EXTERNAL_MODEL,
    tuning: AttributionTuning | None = None,
) -> TranscriptionResult:
    """Force-align a speakerless transcript against its audio and return its
    words as bounded reader lines (rung ``forced``, no speakers, no review)."""
    duration = audio_duration(
        audio_path, fallback=max((segment.end for segment in segments), default=0.0)
    )
    windows = alignment_input(segments, windows_trusted=True, total_duration=duration)
    if windows is not segments:
        logger.info("segment windows are implausible; aligning the whole audio in one window")
    words = aligner(audio_path, windows, language)
    if not words:
        raise RuntimeError("forced alignment produced no words for the supplied transcript")
    lines = segments_without_speakers(words, tuning=tuning)
    logger.info(
        "transcript timed: alignment=forced words=%d lines=%d",
        len(words),
        len(lines),
        extra={"event": "timestamp.align", "words": len(words), "lines": len(lines)},
    )
    return TranscriptionResult(
        language=language if language != "auto" else "unknown",
        duration_seconds=max(duration, lines[-1].end),
        model=model,
        text=render_text(lines),
        segments=lines,
        alignment="forced",
    )
