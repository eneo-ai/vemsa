"""Speaker labelling of an externally produced transcript (task=diarize).

The ASR already happened somewhere else; this runs diarization on the audio and
merges the turns into the caller's words or segments with the same heuristics the
transcribe task uses, so both tasks render identical output."""

import logging
import subprocess
from pathlib import Path
from typing import Protocol

from tolka.jobs.models import Segment, TranscriptionResult, Word
from tolka.pipeline.diarize import Turn, resolve_segments
from tolka.pipeline.render import render_text

logger = logging.getLogger(__name__)


class SpeakerDiarizer(Protocol):
    def diarize(self, audio_path: Path) -> list[Turn]: ...

    def load(self) -> None: ...


def audio_duration(audio_path: Path, *, fallback: float) -> float:
    """Decoded length of the audio, or ``fallback`` when it cannot be read.

    soundfile first (cheap, no subprocess), then ffprobe for the formats libsndfile
    cannot open (e.g. m4a) — the duration feeds consumers' usage accounting."""
    try:
        import soundfile

        info = soundfile.info(str(audio_path))
        return float(info.frames) / info.samplerate
    except Exception:
        pass
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(audio_path),
            ],
            capture_output=True,
            timeout=60,
        )
        return float(completed.stdout.strip())
    except Exception:
        return fallback


def label_speakers(
    diarizer: SpeakerDiarizer,
    audio_path: Path,
    *,
    words: list[Word],
    segments: list[Segment],
    language: str,
    model: str,
) -> TranscriptionResult:
    plain_segments = [segment.model_copy(update={"speaker": None}) for segment in segments]
    if not words:
        words = [word for segment in plain_segments for word in segment.words]
    if not words:
        logger.info("no word timestamps supplied; merging speakers per segment")
    turns = diarizer.diarize(audio_path)
    labelled = resolve_segments(words, plain_segments, turns)
    last_end = max((segment.end for segment in labelled), default=0.0)
    return TranscriptionResult(
        language=language if language != "auto" else "unknown",
        duration_seconds=audio_duration(audio_path, fallback=last_end),
        model=model,
        text=render_text(labelled),
        segments=labelled,
    )
