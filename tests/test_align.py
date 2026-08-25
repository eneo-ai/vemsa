"""Transcript-window selection for forced alignment."""

from tolka.jobs.models import Segment
from tolka.pipeline.align import alignment_transcript


def segment(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


def test_segment_windows_are_passed_verbatim():
    segments = [segment(0.0, 4.0, "hej och välkomna"), segment(5.0, 8.0, "tack så mycket")]
    assert alignment_transcript(segments) == [
        {"start": 0.0, "end": 4.0, "text": "hej och välkomna"},
        {"start": 5.0, "end": 8.0, "text": "tack så mycket"},
    ]


def test_whole_file_segment_window_is_kept():
    # a consumer that rejected the provider timeline sends one segment covering
    # the whole audio; its window is the whole file, which aligns everything
    segments = [segment(0.0, 56.6, "en hel fil med text")]
    assert alignment_transcript(segments) == [
        {"start": 0.0, "end": 56.6, "text": "en hel fil med text"}
    ]


def test_consecutive_chunk_windows_are_kept():
    segments = [
        segment(0.0, 300.0, "första fem minuterna"),
        segment(300.0, 600.0, "andra fem minuterna"),
    ]
    assert alignment_transcript(segments) == [
        {"start": 0.0, "end": 300.0, "text": "första fem minuterna"},
        {"start": 300.0, "end": 600.0, "text": "andra fem minuterna"},
    ]


def test_empty_segments_fall_back_to_the_payload_text():
    segments = [segment(0.0, 1.0, "   ")]
    assert alignment_transcript(segments, fallback_text=" hela ") == [
        {"start": 0.0, "end": 0.0, "text": "hela"}
    ]
