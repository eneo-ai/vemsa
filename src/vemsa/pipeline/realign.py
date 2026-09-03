"""task=align: re-derive word timestamps for a speaker-labelled transcript.

The consumer's transcript came back from Vemsa once, a human then corrected words
and moved sentences between speakers, and now the timestamps must follow the text
again. Nothing here decides *who* spoke — the caller's labels are ground truth and
kept verbatim — only *when*: the corrected text is force-aligned against the audio
(the same CTC aligner every tier uses) and the segment windows are tightened to the
words that came out. No ASR, no diarization, so a round-trip costs one alignment
pass instead of a full pipeline run, and the speaker names a human fixed cannot be
scrambled by a fresh clustering.

The pure parts (window grouping, token counting, redistribution) are CI-tested
without the ML stack; only ``align_transcript`` touches the aligner."""

import logging
import re
import unicodedata
from pathlib import Path

from vemsa.jobs.models import EXTERNAL_MODEL, Segment, TranscriptionResult, Word
from vemsa.pipeline.align import SegmentAligner
from vemsa.pipeline.diarize import AttributionTuning, audio_duration
from vemsa.pipeline.render import render_text

logger = logging.getLogger(__name__)

_NON_ALIGNABLE = re.compile(r"[^\w\s]")


def alignable_tokens(text: str) -> int:
    """How many words the aligner will produce for ``text``.

    Mirrors easyaligner's default normalizer (NFKC, lowercase, drop everything
    that is neither a word character nor whitespace, split on whitespace) so a
    window's words can be handed back to the segments it was built from. A
    punctuation-only token ("—", "...") vanishes under that normalization and
    therefore counts as no word."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return len(_NON_ALIGNABLE.sub("", normalized).split())


def group_windows(segments: list[Segment], *, merge_gap_s: float) -> list[list[int]]:
    """Indices of the (time-ordered) segments that share one alignment window.

    Consecutive segments whose windows overlap, or sit closer than merge_gap_s,
    are aligned together: a sentence a human split between two speakers arrives
    as two segments over one window, and only aligning the whole sentence at
    once lets the audio decide where the split lands. Segments without any
    alignable text never start a window of their own."""
    order = sorted(range(len(segments)), key=lambda index: (segments[index].start, index))
    groups: list[list[int]] = []
    window_end: float | None = None
    for index in order:
        segment = segments[index]
        if not alignable_tokens(segment.text):
            continue
        if window_end is not None and segment.start - window_end <= merge_gap_s:
            groups[-1].append(index)
            window_end = max(window_end, segment.end)
        else:
            groups.append([index])
            window_end = segment.end
    return groups


def window_for(segments: list[Segment], group: list[int]) -> Segment:
    """One alignment window for a group: the union of the members' windows and
    their texts in time order."""
    members = [segments[index] for index in group]
    return Segment(
        start=min(member.start for member in members),
        end=max(member.end for member in members),
        text=" ".join(member.text.strip() for member in members),
    )


def pad_windows(windows: list[Segment], *, pad_s: float, duration: float) -> list[Segment]:
    """Give every (time-ordered, disjoint) window ``pad_s`` of slack on each side.

    Windows built from segment bounds hug the words, and the aligner needs a
    little audio around a word to place its edges — a word ending exactly at its
    window's end comes back with a garbage end time. The slack is clamped to the
    audio and, between neighbours, to the midpoint of their gap so windows stay
    disjoint (the words are handed back per window by count, in time order)."""
    padded: list[Segment] = []
    for index, window in enumerate(windows):
        start = max(0.0, window.start - pad_s)
        end = window.end + pad_s
        if duration > 0:
            end = min(duration, end)
        if index > 0:
            midpoint = (windows[index - 1].end + window.start) / 2
            start = max(start, min(midpoint, window.start))
        if index + 1 < len(windows):
            midpoint = (window.end + windows[index + 1].start) / 2
            end = min(end, max(midpoint, window.end))
        padded.append(window.model_copy(update={"start": start, "end": max(end, window.end)}))
    return padded


def distribute_words(
    segments: list[Segment], group: list[int], words: list[Word]
) -> dict[int, list[Word]]:
    """Hand a window's aligned words back to its member segments, in order,
    each taking as many words as its text has alignable tokens.

    A single-member window takes everything it got (nothing to split, and
    easyaligner's exact token count is then irrelevant). A multi-member window
    whose word count disagrees with the members' token count is a contract
    violation between the two normalizers and fails loudly: guessing which
    speaker a word belongs to is exactly the mistake this task must not make."""
    if len(group) == 1:
        return {group[0]: list(words)}
    counts = {index: alignable_tokens(segments[index].text) for index in group}
    expected = sum(counts.values())
    if expected != len(words):
        raise RuntimeError(
            f"forced alignment returned {len(words)} words for a window whose"
            f" {len(group)} segments hold {expected} alignable tokens; cannot"
            " attribute the words to segments"
        )
    distributed: dict[int, list[Word]] = {}
    cursor = 0
    for index in group:
        distributed[index] = words[cursor : cursor + counts[index]]
        cursor += counts[index]
    return distributed


def retime_segments(
    segments: list[Segment], words_by_segment: dict[int, list[Word]]
) -> list[Segment]:
    """The caller's segments with windows tightened to their aligned words.

    Speaker and text stay verbatim. A segment that got no words (no alignable
    text) keeps the window it arrived with. Output order is time order of the
    new windows, ties broken by input order, so a caller that re-sorted its
    segments still gets a chronological transcript back."""
    retimed: list[tuple[float, int, Segment]] = []
    for index, segment in enumerate(segments):
        words = words_by_segment.get(index, [])
        if words:
            updated = segment.model_copy(
                update={"start": words[0].start, "end": words[-1].end, "words": words}
            )
        else:
            updated = segment.model_copy(update={"words": []})
        retimed.append((updated.start, index, updated))
    retimed.sort(key=lambda item: (item[0], item[1]))
    return [segment for _, _, segment in retimed]


def align_transcript(
    aligner: SegmentAligner,
    audio_path: Path,
    *,
    segments: list[Segment],
    language: str,
    model: str = EXTERNAL_MODEL,
    tuning: AttributionTuning | None = None,
) -> TranscriptionResult:
    """Force-align a speaker-labelled transcript against its audio and return
    it with word timestamps and tightened segment windows (rung ``forced``).

    The aligner is called once with every window; ``SegmentAligner`` returns a
    flat word list sorted by time, and windows never overlap (overlapping
    segments were merged into one, padding stops at the midpoint between
    neighbours), so the words are split back per window by each window's token
    count and then per segment inside the window."""
    tuning = tuning or AttributionTuning()
    groups = group_windows(segments, merge_gap_s=tuning.align_merge_gap_s)
    if not groups:
        raise RuntimeError("the transcript has no alignable text")
    duration = audio_duration(
        audio_path, fallback=max((segment.end for segment in segments), default=0.0)
    )
    windows = pad_windows(
        [window_for(segments, group) for group in groups],
        pad_s=tuning.align_window_pad_s,
        duration=duration,
    )
    logger.info(
        "aligning corrected transcript: %d segments in %d windows",
        len(segments),
        len(windows),
        extra={"event": "align.windows", "segments": len(segments), "windows": len(windows)},
    )
    words = aligner(audio_path, windows, language)
    if not words:
        raise RuntimeError("forced alignment produced no words for the supplied transcript")
    words = sorted(words, key=lambda word: word.start)
    window_counts = [alignable_tokens(window.text) for window in windows]
    if sum(window_counts) != len(words):
        raise RuntimeError(
            f"forced alignment returned {len(words)} words for {sum(window_counts)}"
            " alignable tokens; cannot attribute the words to segments"
        )
    words_by_segment: dict[int, list[Word]] = {}
    cursor = 0
    for group, count in zip(groups, window_counts, strict=True):
        words_by_segment.update(distribute_words(segments, group, words[cursor : cursor + count]))
        cursor += count
    retimed = retime_segments(segments, words_by_segment)
    last_end = max((segment.end for segment in retimed), default=0.0)
    return TranscriptionResult(
        language=language if language != "auto" else "unknown",
        duration_seconds=max(duration, last_end),
        model=model,
        text=render_text(retimed),
        segments=retimed,
        alignment="forced",
    )
