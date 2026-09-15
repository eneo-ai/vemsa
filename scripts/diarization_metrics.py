"""Synthetic benchmark metrics. No inference, audio decoding, or service calls.

DER uses pyannote's optimal one-to-one mapping, zero collar, and an explicit
whole-recording evaluation window. It includes missed speech and false alarms.
The old pairwise, many-to-one 'mis-attributed speech' score was not valid DER.
"""

from vemsa.pipeline.diarize import Turn
from vemsa.pipeline.speaker_review import detect_overlaps


def overlap_intervals(turns: list[tuple[float, float, str]]) -> list[tuple[float, float]]:
    return [(o.start, o.end) for o in detect_overlaps([Turn(*t) for t in turns]).overlaps]


def overlap_scores(reference, hypothesis):
    """Wall-clock precision/recall; three voices do not triple-count time."""
    truth = overlap_intervals(reference)
    found = overlap_intervals(hypothesis)
    ref_seconds = sum(e - s for s, e in truth)
    hyp_seconds = sum(e - s for s, e in found)
    matched = sum(max(0, min(e, b) - max(s, a)) for s, e in truth for a, b in found)
    return {
        "reference_seconds": ref_seconds,
        "detected_seconds": hyp_seconds,
        "matched_seconds": matched,
        "precision": matched / hyp_seconds if hyp_seconds else None,
        "recall": matched / ref_seconds if ref_seconds else None,
    }


def score_diarization(reference, hypothesis, *, duration: float, skip_overlap=False):
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    if duration <= 0:
        raise ValueError("evaluation duration must be positive")

    def annotation(turns):
        result = Annotation()
        for i, (start, end, label) in enumerate(turns):
            if end > start:
                result[Segment(start, end), i] = label
        # Merge duplicate/adjacent tracks from the same identity before scoring.
        return result.support()

    metric = DiarizationErrorRate(collar=0, skip_overlap=skip_overlap)
    components = metric(
        annotation(reference),
        annotation(hypothesis),
        uem=Timeline([Segment(0, duration)]),
        detailed=True,
    )
    return {
        "der": float(components["diarization error rate"]) if components["total"] else None,
        "reference_speaker_seconds": float(components["total"]),
        "missed_speaker_seconds": float(components["missed detection"]),
        "false_alarm_speaker_seconds": float(components["false alarm"]),
        "confused_speaker_seconds": float(components["confusion"]),
        "collar_seconds": 0,
        "skip_overlap": skip_overlap,
        "mapping": "optimal_one_to_one",
    }
