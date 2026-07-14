from conftest import make_result
from tolka.jobs.models import Segment, Word
from tolka.pipeline.render import render_text


def test_render_with_speakers():
    segments = make_result(diarize=True).segments
    assert render_text(segments) == (
        "[00:00:00 - 00:00:01] SPEAKER_00: hej och välkomna\n"
        "[00:00:02 - 00:00:02] SPEAKER_01: tack så mycket"
    )


def test_render_without_speakers():
    segments = make_result(diarize=False).segments
    assert render_text(segments) == (
        "[00:00:00 - 00:00:01] hej och välkomna\n[00:00:02 - 00:00:02] tack så mycket"
    )


def test_render_formats_hours():
    segment = Segment(
        start=3725.2,
        end=7326.9,
        speaker="SPEAKER_03",
        text="ord",
        words=[Word(word="ord", start=3725.2, end=7326.9)],
    )
    assert render_text([segment]) == "[01:02:05 - 02:02:06] SPEAKER_03: ord"


def test_render_empty():
    assert render_text([]) == ""
