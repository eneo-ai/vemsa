"""Overlap evidence and conservative attribution flags, independent of ML libraries.

Review is derived from regular diarization before caller labels are remapped.
It describes model-detected overlap, not intelligibility or speaker confidence.
"""

import logging
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from vemsa.jobs.models import Segment, SpeakerReview, SpeechOverlap, TranscriptionResult
from vemsa.pipeline.render import render_text

if TYPE_CHECKING:
    from vemsa.pipeline.diarize import Turn

logger = logging.getLogger(__name__)


def detect_overlaps(turns: list["Turn"]) -> SpeakerReview:
    """Sweep half-open turns; repeated tracks of one cluster count as one voice."""
    events: dict[float, Counter[str]] = defaultdict(Counter)
    for turn in turns:
        if not (math.isfinite(turn.start) and math.isfinite(turn.end)):
            raise ValueError("diarization timestamps must be finite")
        start = max(0.0, turn.start)
        if turn.end <= start:
            continue
        events[start][turn.speaker] += 1
        events[turn.end][turn.speaker] -= 1
    active: Counter[str] = Counter()
    intervals: list[SpeechOverlap] = []
    previous: float | None = None
    previous_speakers: frozenset[str] = frozenset()
    for time in sorted(events):
        speakers = frozenset(speaker for speaker, count in active.items() if count > 0)
        if previous is not None and time > previous and len(speakers) >= 2:
            if intervals and intervals[-1].end == previous and speakers == previous_speakers:
                intervals[-1] = intervals[-1].model_copy(update={"end": time})
            else:
                intervals.append(
                    SpeechOverlap(
                        id=f"overlap_{len(intervals):04d}",
                        start=previous,
                        end=time,
                        detected_speaker_count=len(speakers),
                    )
                )
        previous_speakers = speakers
        active.update(events[time])
        # Remove expired labels; keep this sweep proportional to simultaneous voices.
        active = +active
        previous = time
    return SpeakerReview(overlap_detection="available", overlaps=intervals)


def _ids(start: float, end: float, overlaps: list[SpeechOverlap]) -> tuple[str, ...]:
    return tuple(o.id for o in overlaps if start < o.end and end > o.start and end > start)


def _annotate(segment: Segment, ids: tuple[str, ...]) -> Segment:
    return segment.model_copy(
        update={
            "speaker_attribution": (
                "provisional" if ids else "assigned" if segment.speaker else "unassigned"
            ),
            "overlap_ids": list(ids),
        }
    )


def _split_segment(segment: Segment, overlaps: list[SpeechOverlap]) -> list[Segment]:
    if not segment.words:
        return [_annotate(segment, _ids(segment.start, segment.end, overlaps))]
    word_ids = [_ids(word.start, word.end, overlaps) for word in segment.words]
    # Preserve provider punctuation/spacing, rather than rebuilding text from tokens.
    starts: list[int] = []
    cursor = 0
    for word in segment.words:
        at = segment.text.find(word.word, cursor) if word.word else -1
        if at < 0:
            # Without a reliable text anchor, retain the complete segment and words.
            affected = set(_ids(segment.start, segment.end, overlaps))
            affected.update(item for ids in word_ids for item in ids)
            return [_annotate(segment, tuple(o.id for o in overlaps if o.id in affected))]
        starts.append(at)
        cursor = at + len(word.word)
    boundaries = [0] + [i for i in range(1, len(word_ids)) if word_ids[i] != word_ids[i - 1]]
    boundaries.append(len(word_ids))
    if len(boundaries) == 2:
        return [_annotate(segment, word_ids[0])]
    result = []
    for first, last in zip(boundaries, boundaries[1:], strict=False):
        words = segment.words[first:last]
        text_start = starts[first] if first else 0
        text_end = starts[last] if last < len(starts) else len(segment.text)
        part = segment.model_copy(
            update={
                "start": min(word.start for word in words),
                "end": max(word.end for word in words),
                "text": segment.text[text_start:text_end].strip(),
                "words": words,
            }
        )
        result.append(_annotate(part, word_ids[first]))
    return result


def apply_speaker_review(
    result: TranscriptionResult, review: SpeakerReview | None
) -> TranscriptionResult:
    """Apply after attribution/smoothing so those heuristics cannot erase flags."""
    if review is None:
        return result
    segments = []
    # Keep interval lookups local to each segment, including outlying word timestamps.
    ordered = review.overlaps
    ends = [o.end for o in ordered]
    for segment in result.segments:
        start = min([segment.start, *(w.start for w in segment.words)])
        end = max([segment.end, *(w.end for w in segment.words)])
        candidates = []
        index = bisect_right(ends, start)
        while index < len(ordered) and ordered[index].start < end:
            candidates.append(ordered[index])
            index += 1
        segments.extend(_split_segment(segment, candidates))
    logger.info(
        "speaker review prepared",
        extra={
            "event": "diarize.review",
            "overlap_detection": review.overlap_detection,
            "overlap_seconds": round(sum(o.end - o.start for o in ordered), 3),
            "overlap_intervals": len(ordered),
            "provisional_words": sum(
                len(s.words) for s in segments if s.speaker_attribution == "provisional"
            ),
        },
    )
    return result.model_copy(
        update={"segments": segments, "text": render_text(segments), "speaker_review": review}
    )
