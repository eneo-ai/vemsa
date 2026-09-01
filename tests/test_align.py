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


def test_emissions_model_resolution(tmp_path):
    from tolka.config import Settings
    from tolka.pipeline.align import _resolve_emissions_model

    settings = Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        emissions_model="fallback/model",
        emissions_models="sv=KBLab/wav2vec2-large-voxrex-swedish,en=facebook/wav2vec2-base-960h",
    )
    # explicit matches, case-insensitively
    assert settings.emissions_model_for("sv") == ("KBLab/wav2vec2-large-voxrex-swedish", True)
    assert settings.emissions_model_for("EN") == ("facebook/wav2vec2-base-960h", True)
    # unmapped languages fall back to the default model
    assert settings.emissions_model_for("fi") == ("fallback/model", False)
    assert _resolve_emissions_model(settings, "fi") == "fallback/model"
    assert _resolve_emissions_model(settings, "auto") == "fallback/model"


def test_emissions_models_entries_are_validated(tmp_path):
    import pytest

    from tolka.config import Settings

    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            database_url="postgresql://unused/unused",
            emissions_models="not-a-pair",
        )


class FakeAlignedWord:
    def __init__(self, text: str, start: float, end: float, score: float | None = None):
        self.text = text
        self.start = start
        self.end = end
        self.score = score


class FakeAlignmentSegment:
    def __init__(self, words):
        self.words = words


class FakeSpeechSegment:
    def __init__(self, words):
        self.alignments = [FakeAlignmentSegment(words)]


def test_alignment_scores_become_word_probabilities():
    from tolka.pipeline.align import _words_from_alignments

    speech = FakeSpeechSegment(
        [FakeAlignedWord("hej", 0.0, 0.4, score=0.91), FakeAlignedWord("då", 0.5, 0.7)]
    )
    words = _words_from_alignments([speech])
    assert [word.probability for word in words] == [0.91, None]


def test_transcribe_alignment_scores_become_word_probabilities():
    from tolka.pipeline.transcribe import words_from_alignments

    segment = FakeAlignmentSegment(
        [FakeAlignedWord("hej", 0.0, 0.4, score=0.55), FakeAlignedWord("då", 0.5, 0.7)]
    )
    words = words_from_alignments([segment])
    assert [word.probability for word in words] == [0.55, None]


def test_word_validates_without_probability():
    # results stored before the probability field existed must still deserialize
    from tolka.jobs.models import Word

    word = Word.model_validate({"word": "hej", "start": 0.0, "end": 0.4})
    assert word.probability is None
