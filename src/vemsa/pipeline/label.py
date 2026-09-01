"""Speaker labelling of an externally produced transcript (task=diarize).

The ASR already happened somewhere else; this runs diarization on the audio and
merges the turns into the caller's words or segments with the same heuristics the
transcribe task uses, so both tasks render identical output. Whenever the caller's
words are not trusted (absent, implausible, or set aside by
VEMSA_DIARIZE_PREFER_ALIGN), the transcript is force-aligned into words — a failed
or unavailable alignment fails the job rather than degrading to a segment-level
merge: the service always runs with a GPU, and quality beats completing coarsely."""

import logging
from pathlib import Path
from typing import Protocol

from vemsa.jobs.models import Alignment, Segment, SpeakerBounds, TranscriptionResult, Word
from vemsa.pipeline.align import SegmentAligner
from vemsa.pipeline.diarize import AttributionTuning, Turn, audio_duration, resolve_segments
from vemsa.pipeline.render import render_text

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

    Whitespace-words per second, checked over the total window time and per
    segment: a broken timeline can average out plausible while individual
    windows cram a sentence into a second (observed from decoder-heuristic
    timestamps: sentence-sized clumps separated by silence). Windows produced
    by such a timeline must not anchor forced alignment."""
    word_count = sum(len(segment.text.split()) for segment in segments)
    if word_count < PLAUSIBLE_MIN_WORDS:
        return True
    window_time = sum(max(0.0, segment.end - segment.start) for segment in segments)
    if window_time <= 0:
        return False
    if word_count / window_time > PLAUSIBLE_MAX_WORDS_PER_SECOND:
        return False
    return all(_window_plausible(segment) for segment in segments)


def _window_plausible(segment: Segment) -> bool:
    words = len(segment.text.split())
    if words < PLAUSIBLE_MIN_WORDS:
        return True
    duration = segment.end - segment.start
    if duration <= 0:
        return False
    return words / duration <= PLAUSIBLE_MAX_WORDS_PER_SECOND


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
    prefer_alignment: bool = False,
    tuning: AttributionTuning | None = None,
) -> TranscriptionResult:
    plain_segments = [segment.model_copy(update={"speaker": None}) for segment in segments]
    if not words:
        words = [word for segment in plain_segments for word in segment.words]
    # the caller's text in order, kept before any word discard: alignment can
    # recover timestamps from the audio, but the text itself only exists here
    caller_text = " ".join(word.word for word in words)
    words_discarded = False
    if words and not words_plausible(words):
        logger.warning(
            "supplied word timestamps are implausible (%d words); discarding them"
            " in favour of forced alignment",
            len(words),
        )
        words = []
        words_discarded = True
    if words and prefer_alignment:
        # Timestamps derived from the audio beat whatever the caller's provider
        # decoded.
        logger.info(
            "caller-supplied words set aside: forced alignment is preferred"
            " (VEMSA_DIARIZE_PREFER_ALIGN)",
            extra={"event": "label.prefer_align", "words": len(words)},
        )
        words = []
    alignment: Alignment = "provider_words"
    if not words:
        # doctrine: word timestamps must be derived from the audio; a missing
        # aligner or a failed alignment fails the job instead of degrading to
        # a segment-level merge
        if aligner is None:
            raise RuntimeError(
                "forced alignment is required to label this transcript but no"
                " aligner is available (install the 'align' extra)"
            )
        total_duration = audio_duration(
            audio_path, fallback=max((segment.end for segment in plain_segments), default=0.0)
        )
        if plain_segments:
            # windows from the same timeline as discarded words cannot anchor alignment
            align_segments = alignment_input(
                plain_segments,
                windows_trusted=not words_discarded,
                total_duration=total_duration,
            )
            if align_segments is not plain_segments:
                logger.info("segment windows are untrusted; aligning the whole audio in one window")
        else:
            # words-only input whose timestamps were discarded: the text survives
            align_segments = [Segment(start=0.0, end=total_duration, text=caller_text)]
        words = aligner(audio_path, align_segments, language)
        if not words:
            raise RuntimeError("forced alignment produced no words for the supplied transcript")
        alignment = "forced"
    turns = diarizer.diarize(audio_path, speakers=speakers)
    labelled = resolve_segments(words, plain_segments, turns, tuning=tuning)
    logger.info(
        "speaker labels merged per word: alignment=%s words=%d",
        alignment,
        len(words),
        extra={"event": "label.align", "alignment": alignment, "words": len(words)},
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
