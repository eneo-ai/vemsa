"""Transcript-window selection for forced alignment."""

from vemsa.jobs.models import Segment
from vemsa.pipeline.align import alignment_transcript


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
    def __init__(self, words):
        self.words = words


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


def test_alignment_runs_overlap_under_gpu_concurrency_two(monkeypatch, tmp_path):
    assert _fake_alignment_run(monkeypatch, tmp_path, gpu_limit=2, threads=2) == 2


def test_alignment_runs_serialize_under_gpu_concurrency_one(monkeypatch, tmp_path):
    assert _fake_alignment_run(monkeypatch, tmp_path, gpu_limit=1, threads=2) == 1
