"""Transcript-window selection for forced alignment."""

from pathlib import Path

import pytest

from vemsa.jobs.models import Segment
from vemsa.pipeline.align import alignment_available, alignment_transcript


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
    from vemsa.config import Settings
    from vemsa.pipeline.align import _resolve_emissions_model

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

    from vemsa.config import Settings

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
    def __init__(self, words, id=None):
        self.words = words
        # easyaligner names each AlignmentSegment "<speech id>-<n>"
        self.id = id


class FakeSpeechSegment:
    def __init__(self, words):
        self.alignments = [FakeAlignmentSegment(words)]


def test_alignment_scores_become_word_probabilities():
    from vemsa.pipeline.align import words_from_alignments

    speech = FakeSpeechSegment(
        [FakeAlignedWord("hej", 0.0, 0.4, score=0.91), FakeAlignedWord("då", 0.5, 0.7)]
    )
    words = words_from_alignments([speech])
    assert [word.probability for word in words] == [0.91, None]


def test_flat_alignment_shapes_are_tolerated():
    from vemsa.pipeline.align import words_from_alignments

    segment = FakeAlignmentSegment(
        [FakeAlignedWord("hej", 0.0, 0.4, score=0.55), FakeAlignedWord("då", 0.5, 0.7)]
    )
    words = words_from_alignments([segment])
    assert [word.probability for word in words] == [0.55, None]


def _spans(words):
    return [(word.word, word.start, word.end, word.probability) for word in words]


def _unheard_word_alignment():
    """easyaligner 0.3.3's real output for tests/fixtures/sv_numerals.wav aligned in two
    windows, 0-1.7 s "Hej. Jag är 57 år" and 1.7-3.8 s "gammal och min mamma är 102.".
    "år" is said at 1.81 s, after its window ends, so the aligner put it in the zero
    padding of the window's 30 s chunk: probability 0, start past the next word's,
    end 30 s after the window's start and past the 3.8 s audio."""
    first = FakeAlignmentSegment(
        [
            FakeAlignedWord("Hej.", 0.08095, 0.38452, 0.99227),
            FakeAlignedWord("Jag", 0.82976, 0.93095, 0.99533),
            FakeAlignedWord("är", 0.99167, 1.05238, 0.95553),
            FakeAlignedWord("57", 1.13333, 1.59881, 0.99882),
            FakeAlignedWord("år", 2.10476, 30.3369, 0.0),
        ],
        id="0-0",
    )
    second = FakeAlignmentSegment(
        [
            FakeAlignedWord("gammal", 2.02307, 2.32594, 0.9515),
            FakeAlignedWord("och", 2.38652, 2.44709, 0.94142),
            FakeAlignedWord("min", 2.50767, 2.58844, 0.99633),
            FakeAlignedWord("mamma", 2.6692, 2.9115, 0.99309),
            FakeAlignedWord("är", 2.99227, 3.03265, 0.05654),
            FakeAlignedWord("102.", 3.09323, 3.53745, 0.99512),
        ],
        id="1-0",
    )
    return [first, second]


def test_a_word_its_window_did_not_hear_stays_in_its_window():
    from vemsa.pipeline.align import words_from_alignments

    words = words_from_alignments(_unheard_word_alignment(), [(0.0, 1.7), (1.7, 3.8)])

    text = "Hej. Jag är 57 år gammal och min mamma är 102."
    assert [word.word for word in words] == text.split(" ")
    # the unplaced word sits at its window's end, before the next window's words
    assert _spans(words)[4] == ("år", 1.7, 1.7, 0.0)
    # placed words keep their times
    unbounded = _spans(words_from_alignments(_unheard_word_alignment()))
    assert _spans(words)[:4] + _spans(words)[5:] == [span for span in unbounded if span[0] != "år"]


def test_an_unplaced_word_ends_by_the_next_word_start():
    """Overlapping windows, spans as in a streamed meeting: the last word of the first
    window came back at probability 0 spanning past words of the next window. It ends
    where the next word starts; a placed word that overlaps it keeps its end, as
    overlapping speech does."""
    from vemsa.pipeline.align import words_from_alignments

    first = FakeAlignmentSegment(
        [
            FakeAlignedWord("men", 27.45, 27.65, 0.9),
            FakeAlignedWord("sen", 27.59, 48.3, 0.0),
        ],
        id="0-0",
    )
    second = FakeAlignmentSegment(
        [
            FakeAlignedWord("ja", 27.0, 27.8, 0.8),
            FakeAlignedWord("precis", 27.83, 28.39, 0.9),
        ],
        id="1-0",
    )
    words = words_from_alignments([first, second], [(18.3, 30.0), (26.9, 40.0)])

    assert _spans(words) == [
        ("ja", 27.0, 27.8, 0.8),
        ("men", 27.45, 27.65, 0.9),
        ("sen", 27.59, 27.83, 0.0),
        ("precis", 27.83, 28.39, 0.9),
    ]


def test_windows_are_found_by_speech_id_not_position():
    """easyaligner drops a window its VAD finds silent; the windows after it keep
    their speech ids, so the second alignment segment here is the third window's."""
    from vemsa.pipeline.align import words_from_alignments

    first = FakeAlignmentSegment([FakeAlignedWord("hej", 0.5, 0.9, 0.9)], id="0-0")
    third = FakeAlignmentSegment(
        [FakeAlignedWord("då", 4.0, 4.4, 0.9), FakeAlignedWord("igen", 4.5, 30.0, 0.0)],
        id="2-0",
    )
    words = words_from_alignments([first, third], [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)])

    assert _spans(words) == [
        ("hej", 0.5, 0.9, 0.9),
        ("då", 4.0, 4.4, 0.9),
        ("igen", 4.5, 5.0, 0.0),
    ]


def test_a_word_placed_before_its_window_starts_at_the_window_start():
    from vemsa.pipeline.align import words_from_alignments

    early = FakeAlignmentSegment(
        [FakeAlignedWord("ja", 9.7, 10.3, 0.6), FakeAlignedWord("visst", 10.4, 10.8, 0.9)],
        id="0-0",
    )
    words = words_from_alignments([early], [(10.0, 12.0)])

    assert _spans(words) == [("ja", 10.0, 10.3, 0.6), ("visst", 10.4, 10.8, 0.9)]


def test_a_placed_word_cut_by_its_window_ends_at_the_window_end():
    """Real aligner output for a word the window's end cut in half (synthesized
    meeting, window 18-24 s; the word is said 23.55-24.41 s): it scores just above 0
    and its end lands in the chunk's padding all the same."""
    from vemsa.pipeline.align import words_from_alignments

    cut = FakeAlignmentSegment(
        [
            FakeAlignedWord("om", 23.39799, 23.45819, 0.98154),
            FakeAlignedWord("skolskjutsarna?", 23.55853, 48.08027, 0.00424),
        ],
        id="0-0",
    )
    words = words_from_alignments([cut], [(18.0, 24.0)])

    assert _spans(words) == [
        ("om", 23.39799, 23.45819, 0.98154),
        ("skolskjutsarna?", 23.55853, 24.0, 0.00424),
    ]


def test_a_window_spread_evenly_keeps_its_spans():
    """easyaligner's fallback for a window it cannot align at all already ends every
    word where the next starts; the bounds leave it as it is."""
    from vemsa.pipeline.align import words_from_alignments

    spread = FakeAlignmentSegment(
        [
            FakeAlignedWord("ett", 10.0, 12.0, 0.0),
            FakeAlignedWord("två", 12.0, 14.0, 0.0),
            FakeAlignedWord("tre", 14.0, 16.0, 0.0),
        ],
        id="0-0",
    )
    words = words_from_alignments([spread], [(10.0, 16.0)])

    assert _spans(words) == [
        ("ett", 10.0, 12.0, 0.0),
        ("två", 12.0, 14.0, 0.0),
        ("tre", 14.0, 16.0, 0.0),
    ]


def test_without_windows_an_unplaced_word_still_ends_by_the_next_word_start():
    # the local engine: easytranscriber returns the alignments but not its windows
    from vemsa.pipeline.align import words_from_alignments

    words = words_from_alignments(_unheard_word_alignment())

    by_text = {word.word: word for word in words}
    assert (by_text["år"].start, by_text["år"].end) == (2.10476, 2.38652)
    assert [word.start for word in words] == sorted(word.start for word in words)


def test_force_align_keeps_each_word_in_the_window_it_was_aligned_in(monkeypatch, tmp_path):
    """force_align_segments hands words_from_alignments the windows it gave the
    aligner, so the windows are found by speech id."""
    import sys
    import types

    from vemsa.pipeline import align

    class FakeSpeechSegment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeProcessor:
        tokenizer = types.SimpleNamespace(pad_token_id=0, word_delimiter_token="|")

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        align, "_easyaligner", lambda: (FakeSpeechSegment, lambda **_: [_unheard_word_alignment()])
    )
    monkeypatch.setattr(align, "_load_alignment_stack", lambda *_: (object(), FakeProcessor()))
    monkeypatch.setattr(align, "_vad_model", lambda: object())
    monkeypatch.setattr(align, "_decodable_audio", lambda path: (path, False))
    monkeypatch.setattr(align, "audio_duration", lambda *_, **__: 3.8)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"not really audio")
    segments = [
        Segment(start=0.0, end=1.7, text="Hej. Jag är 57 år"),
        # a segment without text is not handed to the aligner and takes no speech id
        Segment(start=1.7, end=1.7, text=" "),
        Segment(start=1.7, end=3.8, text="gammal och min mamma är 102."),
    ]

    words = align.force_align_segments(_alignment_settings(tmp_path), audio, segments, "sv")

    assert [word.word for word in words][4:6] == ["år", "gammal"]
    assert _spans(words)[4] == ("år", 1.7, 1.7, 0.0)


def test_word_validates_without_probability():
    # results stored before the probability field existed must still deserialize
    from vemsa.jobs.models import Word

    word = Word.model_validate({"word": "hej", "start": 0.0, "end": 0.4})
    assert word.probability is None


# --- concurrency: shared CTC stack, per-thread VAD, GPU slot ------------------


def _alignment_settings(tmp_path):
    from vemsa.config import Settings

    return Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        work_dir=tmp_path / "work",
        model_cache_dir=tmp_path / "models",
    )


def test_vad_model_is_per_thread_and_ctc_stack_is_shared(monkeypatch, tmp_path):
    import threading

    from vemsa.pipeline import align

    created: list[object] = []

    def fake_vad():
        created.append(object())
        return created[-1]

    monkeypatch.setattr(align, "_new_vad_model", fake_vad)
    monkeypatch.setattr(align, "_loaded_stacks", {})
    stack = (object(), object())
    align._loaded_stacks[("m", "cpu")] = stack
    settings = _alignment_settings(tmp_path)

    seen: dict[str, tuple[object, object, object]] = {}

    def probe(name: str) -> None:
        first = align._vad_model()
        second = align._vad_model()
        seen[name] = (first, second, align._load_alignment_stack(settings, "m", "cpu"))

    threads = [threading.Thread(target=probe, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # one VAD instance per thread, reused within the thread
    assert seen["a"][0] is seen["a"][1] and seen["b"][0] is seen["b"][1]
    assert seen["a"][0] is not seen["b"][0]
    assert len(created) == 2
    # the CTC stack is the same object for both threads
    assert seen["a"][2] is stack and seen["b"][2] is stack


def _fake_alignment_run(monkeypatch, tmp_path, *, gpu_limit: int, threads: int) -> int:
    """Run `threads` concurrent force_align_segments calls whose fake pipeline meets
    at a barrier; returns the peak number of runs inside the pipeline at once."""
    import sys
    import threading
    import types

    from vemsa.pipeline import align, gpu

    barrier = threading.Barrier(threads, timeout=1.0)
    lock = threading.Lock()
    state = {"inside": 0, "peak": 0}

    def fake_pipeline(**kwargs):
        with lock:
            state["inside"] += 1
            state["peak"] = max(state["peak"], state["inside"])
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        finally:
            with lock:
                state["inside"] -= 1
        return [[]]

    class FakeSpeechSegment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeProcessor:
        tokenizer = types.SimpleNamespace(pad_token_id=0, word_delimiter_token="|")

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(align, "_easyaligner", lambda: (FakeSpeechSegment, fake_pipeline))
    monkeypatch.setattr(align, "_load_alignment_stack", lambda *_: (object(), FakeProcessor()))
    monkeypatch.setattr(align, "_vad_model", lambda: object())
    monkeypatch.setattr(align, "_decodable_audio", lambda path: (path, False))
    monkeypatch.setattr(align, "audio_duration", lambda *_, **__: 1.0)

    settings = _alignment_settings(tmp_path)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"not really audio")
    segments = [Segment(start=0.0, end=1.0, text="hej")]
    results: list[object] = []

    def run() -> None:
        results.append(align.force_align_segments(settings, audio, segments, "sv"))

    gpu.configure_gpu_slots(gpu_limit)
    try:
        workers = [threading.Thread(target=run) for _ in range(threads)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    finally:
        gpu.configure_gpu_slots(1)
    assert results == [[], []]
    return state["peak"]


def test_force_align_spells_numerals_out_for_the_job_language(monkeypatch, tmp_path):
    """The aligner is handed Vemsa's normalizer, picked per language: Swedish
    numerals reach the CTC model spelled out, other languages get the library
    default (digits as-is)."""
    import sys
    import types

    pytest.importorskip("easyaligner.text.normalization")
    from vemsa.pipeline import align

    seen: dict[str, object] = {}

    def fake_pipeline(**kwargs):
        seen.update(kwargs)
        return [[]]

    class FakeSpeechSegment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeProcessor:
        tokenizer = types.SimpleNamespace(pad_token_id=0, word_delimiter_token="|")

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(align, "_easyaligner", lambda: (FakeSpeechSegment, fake_pipeline))
    monkeypatch.setattr(align, "_load_alignment_stack", lambda *_: (object(), FakeProcessor()))
    monkeypatch.setattr(align, "_vad_model", lambda: object())
    monkeypatch.setattr(align, "_decodable_audio", lambda path: (path, False))
    monkeypatch.setattr(align, "audio_duration", lambda *_, **__: 1.0)
    settings = _alignment_settings(tmp_path)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"not really audio")
    segments = [Segment(start=0.0, end=1.0, text="Jag är 57 år")]

    align.force_align_segments(settings, audio, segments, "sv")
    tokens, mapping = seen["text_normalizer_fn"]("Jag är 57 år")
    assert tokens == ["jag", "är", "femtiosju", "år"]
    assert mapping[2]["text"] == "57"

    align.force_align_segments(settings, audio, segments, "en")
    tokens, _ = seen["text_normalizer_fn"]("Jag är 57 år")
    assert tokens == ["jag", "är", "57", "år"]


def _real_alignment_stack_available() -> bool:
    import shutil

    from vemsa.config import Settings

    cache = Settings(_env_file=None, database_url="postgresql://unused/unused").model_cache_dir
    return (
        alignment_available()
        and shutil.which("ffmpeg") is not None
        and any(cache.glob("models--KBLab--wav2vec2-large-voxrex-swedish/snapshots/*/vocab.json"))
    )


@pytest.mark.skipif(
    not _real_alignment_stack_available(),
    reason="needs easyaligner, ffmpeg and the cached Swedish CTC model (data/models)",
)
def test_numerals_are_aligned_not_interpolated():
    """Acceptance for the numeral fix on a synthetic Swedish clip (macOS `say -v Alva`:
    "Hej. Jag är femtiosju år gammal och min mamma är hundratvå."): the numerals
    written as digits come back timed by the audio with a real score, and are no
    longer counted as interpolated. Before the fix, "57" scored 3.6e-9 (0.0 after
    rounding) and "102" at the window's end got a garbage end time."""
    from vemsa.config import Settings
    from vemsa.pipeline.align import force_align_segments, interpolated_words

    settings = Settings(_env_file=None, database_url="postgresql://unused/unused")
    clip = Path(__file__).parent / "fixtures" / "sv_numerals.wav"
    text = "Hej. Jag är 57 år gammal och min mamma är 102."
    words = force_align_segments(settings, clip, [Segment(start=0.0, end=3.8, text=text)], "sv")
    assert [word.word for word in words] == text.split()
    assert interpolated_words(words) == 0
    by_text = {word.word: word for word in words}
    assert by_text["57"].probability > 0.5
    assert 1.0 < by_text["57"].start < by_text["57"].end < 1.8
    assert by_text["102."].probability > 0.5
    assert 2.9 < by_text["102."].start < by_text["102."].end <= 3.8


@pytest.mark.skipif(
    not _real_alignment_stack_available(),
    reason="needs easyaligner, ffmpeg and the cached Swedish CTC model (data/models)",
)
def test_a_word_said_after_its_window_stays_in_its_window():
    """Acceptance on the same clip in two windows that cut "år" (said at 1.81 s) into
    the first: the aligner leaves it unplaced (probability 0.0), and it comes back at
    its window's end instead of 30 s past the window's start."""
    from vemsa.config import Settings
    from vemsa.pipeline.align import force_align_segments

    settings = Settings(_env_file=None, database_url="postgresql://unused/unused")
    clip = Path(__file__).parent / "fixtures" / "sv_numerals.wav"
    windows = [
        Segment(start=0.0, end=1.7, text="Hej. Jag är 57 år"),
        Segment(start=1.7, end=3.8, text="gammal och min mamma är 102."),
    ]
    words = force_align_segments(settings, clip, windows, "sv")
    text = "Hej. Jag är 57 år gammal och min mamma är 102."
    assert [word.word for word in words] == text.split(" ")
    assert _spans(words)[4] == ("år", 1.7, 1.7, 0.0)
    assert all(word.end <= 3.8 for word in words)


def test_alignment_runs_overlap_under_gpu_concurrency_two(monkeypatch, tmp_path):
    assert _fake_alignment_run(monkeypatch, tmp_path, gpu_limit=2, threads=2) == 2


def test_alignment_runs_serialize_under_gpu_concurrency_one(monkeypatch, tmp_path):
    assert _fake_alignment_run(monkeypatch, tmp_path, gpu_limit=1, threads=2) == 1
