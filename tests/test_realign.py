"""task=align: re-timing a corrected, speaker-labelled transcript against the audio."""

import json
from pathlib import Path

import pytest

from conftest import FakeEngine
from test_diarize_task import AUTH, api_client, poll_until, submit
from vemsa.config import Settings
from vemsa.jobs.models import JobRequest, Segment, Word
from vemsa.pipeline.diarize import AttributionTuning
from vemsa.pipeline.fake import CannedEngine
from vemsa.pipeline.realign import (
    align_transcript,
    alignable_tokens,
    distribute_words,
    group_windows,
    pad_windows,
    retime_segments,
    window_for,
)


def segment(start: float, end: float, text: str, speaker: str | None = None) -> Segment:
    return Segment(start=start, end=end, text=text, speaker=speaker)


# ---- alignable_tokens: the contract with easyaligner's normalizer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hej och välkomna", 3),
        ("Hej, och välkomna!", 3),
        ("hej — och … välkomna", 3),  # punctuation-only tokens vanish
        ("  hej\n\toch  ", 2),
        ("...", 0),
        ("", 0),
        ("ﬁka", 1),  # NFKC folds the ligature; still one word
        ("d.v.s. imorgon", 2),  # abbreviation dots stripped, letters glued: one token
    ],
)
def test_alignable_tokens(text: str, expected: int):
    assert alignable_tokens(text) == expected


def test_alignable_tokens_matches_easyaligner_normalizer():
    normalization = pytest.importorskip("easyaligner.text.normalization")
    for text in [
        "Hej, och välkomna!",
        "hej — och … välkomna",
        "Vi ses kl. 14:30 på Sveavägen 12 (om det går).",
        "ﬁka? Ja! 100% säkert…",
        "d.v.s. imorgon",
    ]:
        tokens, _ = normalization.text_normalizer(text)
        assert alignable_tokens(text) == len(tokens), text


# ---- window grouping


def test_split_sentence_halves_share_one_window():
    # a human moved the second half of a sentence to another speaker: eneo sends
    # both halves over the original segment's window
    segments = [
        segment(10.0, 14.0, "vi ses imorgon", "SPEAKER_00"),
        segment(10.0, 14.0, "ja det gör vi.", "SPEAKER_01"),
        segment(15.2, 17.0, "tack så mycket.", "SPEAKER_00"),
    ]
    groups = group_windows(segments, merge_gap_s=0.5)
    assert groups == [[0, 1], [2]]
    assert window_for(segments, groups[0]) == Segment(
        start=10.0, end=14.0, text="vi ses imorgon ja det gör vi."
    )


def test_gap_knob_merges_adjacent_segments():
    segments = [segment(0.0, 2.0, "hej"), segment(2.3, 4.0, "då"), segment(6.0, 7.0, "tack")]
    assert group_windows(segments, merge_gap_s=0.5) == [[0, 1], [2]]
    assert group_windows(segments, merge_gap_s=0.0) == [[0], [1], [2]]
    assert group_windows(segments, merge_gap_s=5.0) == [[0, 1, 2]]


def test_grouping_follows_time_order_not_input_order():
    segments = [segment(5.0, 6.0, "sen"), segment(0.0, 1.0, "först")]
    assert group_windows(segments, merge_gap_s=0.5) == [[1], [0]]


def test_segments_without_alignable_text_are_left_out_of_windows():
    segments = [segment(0.0, 1.0, "hej"), segment(1.0, 1.5, "…"), segment(1.2, 2.0, "då")]
    assert group_windows(segments, merge_gap_s=0.5) == [[0, 2]]


def test_windows_get_slack_clamped_to_audio_and_neighbours():
    windows = [segment(0.2, 4.0, "a"), segment(4.6, 9.0, "b"), segment(12.0, 14.9, "c")]
    padded = pad_windows(windows, pad_s=0.5, duration=15.0)
    assert [(w.start, w.end) for w in padded] == [
        (0.0, 4.3),  # clamped to the audio start; end stops at the gap's midpoint
        (4.3, 9.5),
        (11.5, 15.0),  # clamped to the audio end
    ]
    assert [w.text for w in padded] == ["a", "b", "c"]


def test_window_padding_never_shrinks_a_window():
    windows = [segment(0.0, 4.0, "a"), segment(4.0, 8.0, "b")]
    assert [(w.start, w.end) for w in pad_windows(windows, pad_s=0.5, duration=8.0)] == [
        (0.0, 4.0),
        (4.0, 8.0),
    ]
    assert pad_windows(windows, pad_s=0.0, duration=0.0) == windows


# ---- redistribution


def words(*specs: tuple[str, float, float]) -> list[Word]:
    return [Word(word=text, start=start, end=end, probability=0.9) for text, start, end in specs]


def test_single_segment_window_takes_all_its_words():
    segments = [segment(0.0, 3.0, "hej, och välkomna")]
    got = words(("hej,", 0.1, 0.4), ("och", 0.5, 0.7), ("välkomna", 0.8, 1.6))
    assert distribute_words(segments, [0], got) == {0: got}


def test_multi_segment_window_splits_by_token_count():
    segments = [
        segment(0.0, 3.0, "vi ses imorgon", "SPEAKER_00"),
        segment(0.0, 3.0, "ja, det gör vi.", "SPEAKER_01"),
    ]
    got = words(
        ("vi", 0.0, 0.2),
        ("ses", 0.3, 0.5),
        ("imorgon", 0.6, 1.1),
        ("ja,", 1.5, 1.7),
        ("det", 1.8, 1.9),
        ("gör", 2.0, 2.2),
        ("vi.", 2.3, 2.5),
    )
    split = distribute_words(segments, [0, 1], got)
    assert [w.word for w in split[0]] == ["vi", "ses", "imorgon"]
    assert [w.word for w in split[1]] == ["ja,", "det", "gör", "vi."]


def test_token_count_mismatch_fails_loudly():
    segments = [segment(0.0, 3.0, "vi ses", "A"), segment(0.0, 3.0, "ja det", "B")]
    with pytest.raises(RuntimeError, match="cannot attribute"):
        distribute_words(segments, [0, 1], words(("vi", 0.0, 0.2), ("ses", 0.3, 0.5)))


def test_retime_keeps_speakers_and_text_and_tightens_windows():
    segments = [
        segment(0.0, 5.0, "hej där.", "Anna"),
        segment(4.0, 9.0, "hej själv.", "Björn"),
        segment(9.0, 9.5, "…", "Anna"),
    ]
    retimed = retime_segments(
        segments,
        {
            0: words(("hej", 0.4, 0.6), ("där.", 0.7, 1.1)),
            1: words(("hej", 4.9, 5.1), ("själv.", 5.2, 5.8)),
        },
    )
    assert [(s.start, s.end, s.speaker, s.text) for s in retimed] == [
        (0.4, 1.1, "Anna", "hej där."),
        (4.9, 5.8, "Björn", "hej själv."),
        (9.0, 9.5, "Anna", "…"),  # no alignable text: window kept, no words
    ]
    assert [len(s.words) for s in retimed] == [2, 2, 0]


# ---- align_transcript end to end with a stubbed aligner


class RecordingAligner:
    """Returns evenly spread words for each window it is given, like the real
    aligner would (in time order), and records the windows."""

    def __init__(self) -> None:
        self.windows: list[list[Segment]] = []

    def __call__(self, audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        self.windows.append(segments)
        out: list[Word] = []
        for window in segments:
            tokens = window.text.split()
            step = (window.end - window.start) / len(tokens)
            for index, token in enumerate(tokens):
                out.append(
                    Word(
                        word=token,
                        start=round(window.start + index * step + 0.05, 3),
                        end=round(window.start + (index + 1) * step - 0.05, 3),
                        probability=0.8,
                    )
                )
        return out


def test_align_transcript_retimes_a_corrected_transcript(tmp_path: Path):
    aligner = RecordingAligner()
    segments = [
        segment(0.0, 4.0, "hej och välkomna.", "SPEAKER_00"),
        # one original segment a human split between two speakers
        segment(5.0, 9.0, "vi ses imorgon", "SPEAKER_01"),
        segment(5.0, 9.0, "ja det gör vi.", "SPEAKER_00"),
        segment(20.0, 22.0, "tack.", "SPEAKER_01"),
    ]
    result = align_transcript(
        aligner,
        tmp_path / "missing.wav",
        segments=segments,
        language="sv",
        model="external",
        tuning=AttributionTuning(align_merge_gap_s=0.5),
    )
    # windows: the split sentence aligned as one, the rest on their own, each
    # with half a second of slack (audio unreadable: no clamp at the end)
    assert [(w.start, w.end, w.text) for w in aligner.windows[0]] == [
        (0.0, 4.5, "hej och välkomna."),
        (4.5, 9.5, "vi ses imorgon ja det gör vi."),
        (19.5, 22.0, "tack."),
    ]
    assert result.alignment == "forced"
    assert result.model == "external"
    assert [(s.speaker, s.text) for s in result.segments] == [
        ("SPEAKER_00", "hej och välkomna."),
        ("SPEAKER_01", "vi ses imorgon"),
        ("SPEAKER_00", "ja det gör vi."),
        ("SPEAKER_01", "tack."),
    ]
    first, second, third, fourth = result.segments
    # the split now has two distinct, non-overlapping windows inside the (padded) original
    assert 4.5 <= second.start < second.end <= third.start < third.end <= 9.5
    assert [w.word for w in second.words] == ["vi", "ses", "imorgon"]
    assert [w.word for w in third.words] == ["ja", "det", "gör", "vi."]
    assert first.start == first.words[0].start and first.end == first.words[-1].end
    assert all(w.probability == 0.8 for s in result.segments for w in s.words)
    # audio unreadable: duration falls back to the last input window's end
    assert result.duration_seconds == 22.0
    assert result.text.count("\n") == 3 and "SPEAKER_01: vi ses imorgon" in result.text


def test_align_transcript_language_auto_is_reported_unknown(tmp_path: Path):
    result = align_transcript(
        RecordingAligner(), tmp_path / "a.wav", segments=[segment(0.0, 1.0, "hej")], language="auto"
    )
    assert result.language == "unknown"


def test_align_transcript_without_text_fails(tmp_path: Path):
    with pytest.raises(RuntimeError, match="no alignable text"):
        align_transcript(
            RecordingAligner(), tmp_path / "a.wav", segments=[segment(0.0, 1.0, "…")], language="sv"
        )


def test_align_transcript_with_empty_alignment_fails(tmp_path: Path):
    def silent_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        return []

    with pytest.raises(RuntimeError, match="no words"):
        align_transcript(
            silent_aligner, tmp_path / "a.wav", segments=[segment(0.0, 1.0, "hej")], language="sv"
        )


def test_align_transcript_with_wrong_word_count_fails(tmp_path: Path):
    def lossy_aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        return [Word(word="hej", start=0.0, end=0.3)]

    with pytest.raises(RuntimeError, match="cannot attribute"):
        align_transcript(
            lossy_aligner,
            tmp_path / "a.wav",
            segments=[segment(0.0, 1.0, "hej"), segment(3.0, 4.0, "då")],
            language="sv",
        )


def test_canned_engine_aligns_with_speakers_verbatim(tmp_path: Path):
    result = CannedEngine().align_transcript(
        tmp_path / "a.wav",
        segments=[segment(0.0, 2.0, "hej och tack", "Anna"), segment(3.0, 4.0, "hej", "Björn")],
        language="sv",
        model="external",
    )
    assert result.alignment == "forced"
    assert [(s.speaker, s.text, len(s.words)) for s in result.segments] == [
        ("Anna", "hej och tack", 3),
        ("Björn", "hej", 1),
    ]
    assert "Anna: hej och tack" in result.text


# ---- request validation


def test_align_request_validation():
    ok = JobRequest(task="align", segments=[{"start": 0.0, "end": 1.0, "text": "hej"}])
    assert ok.task == "align" and ok.speaker_bounds() is None
    with pytest.raises(ValueError, match="segments list with text"):
        JobRequest(task="align")
    with pytest.raises(ValueError, match="segments list with text"):
        JobRequest(task="align", segments=[{"start": 0.0, "end": 1.0, "text": "  "}])
    with pytest.raises(ValueError, match="words are not accepted"):
        JobRequest(task="align", words=[{"word": "hej", "start": 0.0, "end": 0.4}])
    with pytest.raises(ValueError, match="speaker bounds"):
        JobRequest(
            task="align", segments=[{"start": 0.0, "end": 1.0, "text": "hej"}], num_speakers=2
        )
    with pytest.raises(ValueError, match="vocabulary"):
        JobRequest(
            task="align", segments=[{"start": 0.0, "end": 1.0, "text": "hej"}], vocabulary=["x"]
        )
    with pytest.raises(ValueError, match="diarize=false"):
        JobRequest(
            task="align", segments=[{"start": 0.0, "end": 1.0, "text": "hej"}], diarize=False
        )
    with pytest.raises(ValueError, match="timestamps"):
        JobRequest(task="align", segments=[{"start": 2.0, "end": 1.0, "text": "hej"}])


# ---- job lifecycle over the API (PostgreSQL-backed)

SEGMENTS = [
    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "hej och välkomna."},
    {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_01", "text": "tack."},
]


async def test_multipart_align_lifecycle(settings: Settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        response = await submit(client, task="align", language="sv", segments=json.dumps(SEGMENTS))
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        await poll_until(client, job_id, "completed")
        result = (await client.get(f"/v1/jobs/{job_id}/result", headers=AUTH)).json()

    assert result["model"] == "external" and result["alignment"] == "forced"
    assert [(s["speaker"], s["text"]) for s in result["segments"]] == [
        ("SPEAKER_00", "hej och välkomna."),
        ("SPEAKER_01", "tack."),
    ]
    assert all(s["words"] for s in result["segments"])
    call = engine.calls[-1]
    assert call["task"] == "align" and call["language"] == "sv" and call["model"] == "external"
    assert [s.speaker for s in call["segments"]] == ["SPEAKER_00", "SPEAKER_01"]


async def test_json_align_submission_and_diarize_tier(settings: Settings):
    # align needs only the aligner, so the diarize-only tier accepts it
    settings.engine = "diarize"
    async with api_client(settings) as (client, _):
        response = await client.post(
            "/v1/jobs",
            json={"task": "align", "source_url": "https://example.org/m.mp3", "segments": SEGMENTS},
            headers=AUTH,
        )
        assert response.status_code == 202


async def test_align_submission_validation_failures(settings: Settings):
    async with api_client(settings) as (client, _):
        assert (await submit(client, task="align")).status_code == 422
        response = await submit(
            client, task="align", words=json.dumps([{"word": "hej", "start": 0, "end": 1}])
        )
        assert response.status_code == 422
        response = await submit(
            client, task="align", segments=json.dumps(SEGMENTS), num_speakers="2"
        )
        assert response.status_code == 422
        response = await submit(
            client, task="align", segments=json.dumps(SEGMENTS), vocabulary='["x"]'
        )
        assert response.status_code == 422
