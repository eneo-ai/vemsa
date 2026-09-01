from tolka.config import Settings
from tolka.jobs.models import Segment, Word
from tolka.pipeline.diarize import (
    Diarizer,
    Turn,
    assign_speakers,
    assign_speakers_to_segments,
    segments_without_speakers,
)


def word(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end)


def test_word_inside_single_turn():
    segments = assign_speakers([word("hej", 1.0, 1.5)], [Turn(0.0, 5.0, "SPEAKER_00")])
    assert len(segments) == 1
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[0].text == "hej"


def test_word_overlapping_two_turns_picks_larger_overlap():
    turns = [Turn(0.0, 1.1, "SPEAKER_00"), Turn(1.1, 5.0, "SPEAKER_01")]
    # word spans 1.0-2.0: 0.1s in turn 0, 0.9s in turn 1
    segments = assign_speakers([word("x", 1.0, 2.0)], turns)
    assert segments[0].speaker == "SPEAKER_01"


def test_word_in_gap_inherits_previous_speaker():
    turns = [Turn(0.0, 1.0, "SPEAKER_00"), Turn(3.0, 4.0, "SPEAKER_01")]
    words = [word("a", 0.2, 0.8), word("b", 1.5, 2.0), word("c", 3.1, 3.5)]
    segments = assign_speakers(words, turns)
    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert segments[0].text == "a b"  # gap word b inherited SPEAKER_00


def test_first_word_in_gap_uses_nearest_turn():
    turns = [Turn(2.0, 3.0, "SPEAKER_00"), Turn(10.0, 11.0, "SPEAKER_01")]
    segments = assign_speakers([word("a", 0.0, 0.5)], turns)
    assert segments[0].speaker == "SPEAKER_00"


def test_grouping_splits_on_speaker_change_and_sentence_pause():
    turns = [Turn(0.0, 2.0, "SPEAKER_00"), Turn(2.0, 10.0, "SPEAKER_01")]
    words = [
        word("a", 0.0, 0.5),
        word("b", 0.6, 1.0),
        word("c.", 2.5, 3.0),  # speaker change
        word("d", 5.0, 5.5),  # same speaker, >1s pause after a sentence end
    ]
    segments = assign_speakers(words, turns)
    assert [(s.speaker, s.text) for s in segments] == [
        ("SPEAKER_00", "a b"),
        ("SPEAKER_01", "c."),
        ("SPEAKER_01", "d"),
    ]
    assert segments[0].start == 0.0
    assert segments[0].end == 1.0
    assert [w.word for w in segments[0].words] == ["a", "b"]


def test_pause_mid_sentence_keeps_the_sentence_together():
    # >1s pauses without sentence-final punctuation (and after a heading colon)
    # do not split: a human transcriber keeps the sentence on one line
    turns = [Turn(0.0, 12.0, "SPEAKER_00")]
    words = [
        word("Rubrik:", 0.0, 0.5),
        word("hej", 3.0, 3.3),  # pause after a colon: keeps going
        word("och", 6.0, 6.3),  # pause mid-sentence: keeps going
        word("välkomna!", 6.4, 7.0),
        word("Tack.", 10.0, 10.4),  # pause after "!": new segment
    ]
    segments = assign_speakers(words, turns)
    assert [s.text for s in segments] == ["Rubrik: hej och välkomna!", "Tack."]


def test_very_long_silence_splits_even_mid_sentence():
    # unpunctuated transcripts must not collapse into one endless segment
    turns = [Turn(0.0, 40.0, "SPEAKER_00")]
    words = [word("hej", 0.0, 0.5), word("där", 20.0, 20.5)]
    segments = assign_speakers(words, turns)
    assert [s.text for s in segments] == ["hej", "där"]


def test_single_word_island_mid_sentence_is_absorbed():
    # the observed eneo case: an untranscribed backchannel flips the exclusive
    # diarization track to SPEAKER_02 for a moment and the mid-sentence word
    # "som" lands inside that window — it must stay with the surrounding speaker
    turns = [
        Turn(0.0, 2.1, "SPEAKER_01"),
        Turn(2.1, 2.6, "SPEAKER_02"),
        Turn(2.6, 5.0, "SPEAKER_01"),
    ]
    words = [
        word("min", 0.0, 0.3),
        word("mamma", 0.4, 0.8),
        word("jobbade", 0.9, 1.4),
        word("natt", 1.5, 1.9),
        word("som", 2.15, 2.5),
        word("sjuksköterska.", 2.7, 3.6),
    ]
    segments = assign_speakers(words, turns)
    assert [(s.speaker, s.text) for s in segments] == [
        ("SPEAKER_01", "min mamma jobbade natt som sjuksköterska.")
    ]


def test_island_after_sentence_end_is_kept():
    # a short interjection right after sentence-final punctuation is legitimate
    turns = [
        Turn(0.0, 1.1, "SPEAKER_00"),
        Turn(1.1, 1.9, "SPEAKER_01"),
        Turn(1.9, 4.0, "SPEAKER_00"),
    ]
    words = [
        word("Hej", 0.0, 0.4),
        word("där.", 0.5, 1.0),
        word("Ja.", 1.2, 1.7),
        word("Och", 2.0, 2.4),
        word("sedan", 2.5, 3.0),
        word("vidare", 3.1, 3.6),
    ]
    segments = assign_speakers(words, turns)
    assert [(s.speaker, s.text) for s in segments] == [
        ("SPEAKER_00", "Hej där."),
        ("SPEAKER_01", "Ja."),
        ("SPEAKER_00", "Och sedan vidare"),
    ]


def test_long_island_is_not_absorbed():
    # four words over well more than a second is a real speaker turn
    turns = [
        Turn(0.0, 1.0, "SPEAKER_00"),
        Turn(1.0, 3.5, "SPEAKER_01"),
        Turn(3.5, 5.0, "SPEAKER_00"),
    ]
    words = [
        word("a", 0.0, 0.4),
        word("b", 0.5, 0.9),
        word("c", 1.1, 1.6),
        word("d", 1.7, 2.2),
        word("e", 2.3, 2.8),
        word("f", 2.9, 3.4),
        word("g", 3.6, 4.0),
    ]
    segments = assign_speakers(words, turns)
    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_island_at_the_edges_is_kept():
    # no neighbour on one side: nothing to absorb into
    turns = [Turn(0.0, 0.6, "SPEAKER_01"), Turn(0.6, 3.0, "SPEAKER_00")]
    words = [word("a", 0.0, 0.5), word("b", 0.7, 1.2), word("c", 1.3, 1.8)]
    segments = assign_speakers(words, turns)
    assert [s.speaker for s in segments] == ["SPEAKER_01", "SPEAKER_00"]


def test_sliver_overlap_inherits_previous_speaker():
    # word b only clips the SPEAKER_01 turn by 0.05s (10% of its duration):
    # below the coverage floor, so it inherits SPEAKER_00 from the word before
    turns = [Turn(0.0, 0.9, "SPEAKER_00"), Turn(1.45, 3.0, "SPEAKER_01")]
    words = [word("a", 0.2, 0.8), word("b", 1.0, 1.5), word("c", 2.0, 2.5)]
    segments = assign_speakers(words, turns)
    assert [(s.speaker, s.text) for s in segments] == [
        ("SPEAKER_00", "a b"),
        ("SPEAKER_01", "c"),
    ]


def test_no_turns_falls_back_to_speakerless_segments():
    words = [word("a", 0.0, 0.5), word("b", 0.6, 1.0)]
    segments = assign_speakers(words, [])
    assert len(segments) == 1
    assert segments[0].speaker is None


def test_segments_without_speakers_splits_on_sentence_pause():
    words = [word("a.", 0.0, 0.5), word("b", 3.0, 3.5)]
    segments = segments_without_speakers(words)
    assert [s.text for s in segments] == ["a.", "b"]
    assert all(s.speaker is None for s in segments)


def test_empty_words():
    assert assign_speakers([], [Turn(0.0, 1.0, "SPEAKER_00")]) == []
    assert segments_without_speakers([]) == []


def plain_segment(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text, words=[])


def test_segment_level_assignment_picks_max_overlap():
    turns = [Turn(0.0, 1.5, "SPEAKER_00"), Turn(1.5, 5.0, "SPEAKER_01")]
    segments = [plain_segment(0.0, 1.4, "a"), plain_segment(1.4, 3.0, "b")]
    labelled = assign_speakers_to_segments(segments, turns)
    assert [s.speaker for s in labelled] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s.text for s in labelled] == ["a", "b"]


def test_segment_level_assignment_gap_inherits_previous():
    turns = [Turn(0.0, 1.0, "SPEAKER_00")]
    segments = [plain_segment(0.0, 0.9, "a"), plain_segment(5.0, 6.0, "b")]
    labelled = assign_speakers_to_segments(segments, turns)
    assert [s.speaker for s in labelled] == ["SPEAKER_00", "SPEAKER_00"]


def test_segment_level_assignment_without_turns_is_unchanged():
    segments = [plain_segment(0.0, 1.0, "a")]
    assert assign_speakers_to_segments(segments, []) == segments


def test_segment_spanning_two_turns_splits_proportionally_on_word_boundary():
    turns = [Turn(0.0, 6.0, "SPEAKER_00"), Turn(6.0, 10.0, "SPEAKER_01")]
    segments = [plain_segment(0.0, 10.0, "ett två tre fyra fem sex sju åtta nio tio")]
    labelled = assign_speakers_to_segments(segments, turns)
    assert [(s.speaker, s.text) for s in labelled] == [
        ("SPEAKER_00", "ett två tre fyra fem sex"),
        ("SPEAKER_01", "sju åtta nio tio"),
    ]
    assert [(s.start, s.end) for s in labelled] == [(0.0, 6.0), (6.0, 10.0)]
    assert all(s.words == [] for s in labelled)


def test_small_minority_overlap_does_not_split():
    turns = [Turn(0.0, 9.0, "SPEAKER_00"), Turn(9.0, 10.0, "SPEAKER_01")]
    segments = [plain_segment(0.0, 10.0, "ett två tre fyra fem sex sju åtta nio tio")]
    labelled = assign_speakers_to_segments(segments, turns)
    assert [(s.speaker, s.text) for s in labelled] == [
        ("SPEAKER_00", "ett två tre fyra fem sex sju åtta nio tio")
    ]


def test_short_segment_is_never_split():
    turns = [Turn(0.0, 0.75, "SPEAKER_00"), Turn(0.75, 1.5, "SPEAKER_01")]
    segments = [plain_segment(0.0, 1.5, "ett två tre fyra fem")]
    labelled = assign_speakers_to_segments(segments, turns)
    assert len(labelled) == 1


def test_few_words_are_never_split():
    turns = [Turn(0.0, 5.0, "SPEAKER_00"), Turn(5.0, 10.0, "SPEAKER_01")]
    segments = [plain_segment(0.0, 10.0, "ett två tre")]
    labelled = assign_speakers_to_segments(segments, turns)
    assert len(labelled) == 1


def test_split_prefers_sentence_punctuation_over_nearer_space():
    # the cut at 5.5s lands on the space at offset 22, but the sentence end at
    # offset 17 is within the 12-character punctuation window and wins
    turns = [Turn(0.0, 5.5, "SPEAKER_00"), Turn(5.5, 10.0, "SPEAKER_01")]
    segments = [plain_segment(0.0, 10.0, "Hej och välkomna. Tack så mycket för det")]
    labelled = assign_speakers_to_segments(segments, turns)
    assert [(s.speaker, s.text) for s in labelled] == [
        ("SPEAKER_00", "Hej och välkomna."),
        ("SPEAKER_01", "Tack så mycket för det"),
    ]


def test_silence_between_turns_joins_the_preceding_speaker():
    # gap 4-6s inside the segment: the cut sits where the next speaker starts
    turns = [Turn(0.0, 4.0, "SPEAKER_00"), Turn(6.0, 10.0, "SPEAKER_01")]
    segments = [plain_segment(0.0, 10.0, "ett två tre fyra fem sex sju åtta nio tio")]
    labelled = assign_speakers_to_segments(segments, turns)
    assert [s.speaker for s in labelled] == ["SPEAKER_00", "SPEAKER_01"]
    assert labelled[0].end == 6.0 and labelled[1].start == 6.0


def _diarizer(**overrides) -> "Diarizer":
    settings = Settings(_env_file=None, database_url="postgresql://unused/test", **overrides)
    return Diarizer(settings)


class _DiarizeOutput:
    def __init__(self, regular: object, exclusive: object | None):
        self.speaker_diarization = regular
        self.exclusive_speaker_diarization = exclusive


def test_pick_annotation_prefers_exclusive():
    annotation, exclusive = _diarizer()._pick_annotation(_DiarizeOutput("regular", "exclusive"))
    assert (annotation, exclusive) == ("exclusive", True)


def test_pick_annotation_respects_disabled_exclusive():
    diarizer = _diarizer(diarize_exclusive=False)
    annotation, exclusive = diarizer._pick_annotation(_DiarizeOutput("regular", "exclusive"))
    assert (annotation, exclusive) == ("regular", False)


def test_pick_annotation_falls_back_to_bare_annotation():
    annotation, exclusive = _diarizer()._pick_annotation("bare-annotation")
    assert (annotation, exclusive) == ("bare-annotation", False)
