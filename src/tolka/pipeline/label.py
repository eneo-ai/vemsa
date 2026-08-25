"""Speaker labelling of an externally produced transcript (task=diarize).

The ASR already happened somewhere else; this runs diarization on the audio and
merges the turns into the caller's words or segments with the same heuristics the
transcribe task uses, so both tasks render identical output. Segment-only input is
force-aligned into words first when an aligner is available, so speaker changes
inside one segment survive."""

import logging
from pathlib import Path
from typing import Protocol

from tolka.jobs.models import Alignment, Segment, SpeakerBounds, TranscriptionResult, Word
from tolka.pipeline.align import SegmentAligner
from tolka.pipeline.diarize import Turn, audio_duration, resolve_segments
from tolka.pipeline.render import render_text

logger = logging.getLogger(__name__)


class SpeakerDiarizer(Protocol):
    def diarize(self, audio_path: Path, *, speakers: SpeakerBounds | None = None) -> list[Turn]: ...

    def load(self) -> None: ...


def segment_merge_alignment(inputs: list[Segment], outputs: list[Segment]) -> Alignment:
    """How a wordless merge labelled speakers: proportional splits happened iff the
    merge produced more segments than it was given."""
    return "segment_split" if len(outputs) > len(inputs) else "segment_only"


# Providers deriving word timestamps from decoder heuristics sometimes emit a
# compressed timeline (observed from a vLLM-hosted kb-whisper-large: ~7 words/s
# with a 19.5 s hole in a 56 s recording). Merging speakers against such a
# timeline puts every turn in the wrong place, so segment-level fallbacks beat it.
PLAUSIBLE_MAX_WORDS_PER_SECOND = 4.5
PLAUSIBLE_PAUSE_GAP_S = 1.0
PLAUSIBLE_MIN_WORDS = 5


def words_plausible(words: list[Word]) -> bool:
    """Whether a word timeline shows a humanly possible speaking rate.

    The rate is words over speaking time — the overall span minus pauses longer
    than PLAUSIBLE_PAUSE_GAP_S, so long silences do not mask a compressed
    timeline. Timelines too short to judge are accepted."""
    if len(words) < PLAUSIBLE_MIN_WORDS:
        return True
    ordered = sorted(words, key=lambda word: word.start)
    speaking_time = ordered[-1].end - ordered[0].start
    for previous, current in zip(ordered, ordered[1:], strict=False):
        gap = current.start - previous.end
        if gap > PLAUSIBLE_PAUSE_GAP_S:
            speaking_time -= gap
    if speaking_time <= 0:
        return False
    return len(ordered) / speaking_time <= PLAUSIBLE_MAX_WORDS_PER_SECOND


def segment_windows_plausible(segments: list[Segment]) -> bool:
    """Whether segment windows show a humanly possible speaking rate.

    Whitespace-words per second of total window time; windows produced by the
    same rejected decoder timeline as implausible words are compressed the same
    way, so they must not anchor forced alignment."""
    word_count = sum(len(segment.text.split()) for segment in segments)
    if word_count < PLAUSIBLE_MIN_WORDS:
        return True
    window_time = sum(max(0.0, segment.end - segment.start) for segment in segments)
    if window_time <= 0:
        return False
    return word_count / window_time <= PLAUSIBLE_MAX_WORDS_PER_SECOND


def alignment_input(
    plain_segments: list[Segment], *, windows_trusted: bool, total_duration: float
) -> list[Segment]:
    """Segments to hand the aligner: verbatim when their windows can anchor the
    alignment, otherwise the full transcript as one whole-audio window."""
    if windows_trusted and segment_windows_plausible(plain_segments):
        return plain_segments
    text = " ".join(segment.text.strip() for segment in plain_segments if segment.text.strip())
    return [Segment(start=0.0, end=total_duration, text=text)]


def label_speakers(
    diarizer: SpeakerDiarizer,
    audio_path: Path,
    *,
    words: list[Word],
    segments: list[Segment],
    language: str,
    model: str,
    aligner: SegmentAligner | None = None,
    speakers: SpeakerBounds | None = None,
) -> TranscriptionResult:
    plain_segments = [segment.model_copy(update={"speaker": None}) for segment in segments]
    if not words:
        words = [word for segment in plain_segments for word in segment.words]
    words_discarded = False
    if words and plain_segments and not words_plausible(words):
        logger.warning(
            "supplied word timestamps are implausible (%d words); ignoring them in"
            " favour of alignment or segment-level merging",
            len(words),
        )
        words = []
        words_discarded = True
    alignment: Alignment | None = "provider_words" if words else None
    fallback_reason = "no word timestamps supplied"
    if not words and plain_segments and aligner is None:
        fallback_reason = (
            "no aligner configured (align extra not installed or TOLKA_DIARIZE_FORCE_ALIGN=false)"
        )
    if not words and plain_segments and aligner is not None:
        # windows from the same timeline as discarded words cannot anchor alignment
        align_segments = alignment_input(
            plain_segments,
            windows_trusted=not words_discarded,
            total_duration=audio_duration(
                audio_path, fallback=max(segment.end for segment in plain_segments)
            ),
        )
        if align_segments is not plain_segments:
            logger.info("segment windows are untrusted; aligning the whole audio in one window")
        try:
            words = aligner(audio_path, align_segments, language)
            alignment = "forced" if words else None
            fallback_reason = "aligner returned no words"
        except Exception as exc:
            logger.warning(
                "forced alignment of the supplied segments failed; falling back to"
                " segment-level speaker labelling",
                exc_info=True,
            )
            words = []
            alignment = None
            fallback_reason = f"alignment failed: {exc!r}"
    turns = diarizer.diarize(audio_path, speakers=speakers)
    labelled = resolve_segments(words, plain_segments, turns)
    if alignment is None:
        alignment = segment_merge_alignment(plain_segments, labelled)
    if alignment == "forced":
        logger.info(
            "speaker labels merged per word: alignment=forced words=%d",
            len(words),
            extra={"event": "label.align", "alignment": "forced", "words": len(words)},
        )
    elif alignment != "provider_words":
        logger.info(
            "speaker labels merged per segment: alignment=%s reason=%s",
            alignment,
            fallback_reason,
            extra={"event": "label.align", "alignment": alignment, "reason": fallback_reason},
        )
    last_end = max((segment.end for segment in labelled), default=0.0)
    return TranscriptionResult(
        language=language if language != "auto" else "unknown",
        duration_seconds=audio_duration(audio_path, fallback=last_end),
        model=model,
        text=render_text(labelled),
        segments=labelled,
        alignment=alignment,
    )
