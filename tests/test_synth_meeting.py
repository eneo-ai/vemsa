"""Small known answers guard the benchmark before expensive model runs."""

import importlib
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="install the evaluation dependency group")
pytest.importorskip("pyannote.metrics", reason="install the evaluation dependency group")


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("scripts.diarization_metrics")


@pytest.fixture
def synth(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("scripts.synth_meeting")
    # Six seconds for full turns, 0.2s for backchannels; no macOS or model needed.
    monkeypatch.setattr(
        module,
        "synth",
        lambda v, text, **kw: (
            np.ones(int(module.SR * (0.2 if text in module.BACKCHANNELS else 6)), dtype=np.float32)
            * 0.02
        ),
    )
    return module


def test_perfect_simultaneous_voices_have_zero_error(metrics):
    truth = [(0, 2, "Anna"), (0, 2, "Bo")]
    score = metrics.score_diarization(truth, [(0, 2, "x"), (0, 2, "y")], duration=3)
    assert score["der"] == 0
    assert score["reference_speaker_seconds"] == 4
    overlap = metrics.overlap_scores(truth, [(0, 2, "x"), (0, 2, "y")])
    assert overlap["reference_seconds"] == 2
    assert overlap["precision"] == overlap["recall"] == 1


def test_exclusive_track_misses_second_voice_not_confusion(metrics):
    truth = [(0, 2, "Anna"), (0, 2, "Bo")]
    score = metrics.score_diarization(truth, [(0, 2, "x")], duration=3)
    assert score["der"] == 0.5
    assert score["missed_speaker_seconds"] == 2
    assert score["confused_speaker_seconds"] == 0
    assert (
        metrics.score_diarization(truth, [(0, 2, "x")], duration=3, skip_overlap=True)["der"]
        is None
    )


def test_missing_false_alarm_and_oversegmentation_are_counted(metrics):
    truth = [(0, 2, "Anna")]
    assert metrics.score_diarization(truth, [], duration=3)["der"] == 1
    false_alarm = metrics.score_diarization(truth, [(0, 3, "x")], duration=3)
    assert false_alarm["false_alarm_speaker_seconds"] == 1
    fragmented = metrics.score_diarization(truth, [(0, 1, "x"), (1, 2, "y")], duration=3)
    assert fragmented["der"] == 0.5
    assert fragmented["confused_speaker_seconds"] == 1


def test_speaker_swap_cannot_be_hidden_by_per_turn_mapping(metrics):
    truth = [(0, 2, "Anna"), (2, 4, "Bo"), (4, 6, "Anna"), (6, 8, "Bo")]
    mixed = [(0, 2, "x"), (2, 4, "y"), (4, 6, "y"), (6, 8, "x")]
    assert metrics.score_diarization(truth, mixed, duration=8)["der"] == 0.5


def test_triple_overlap_and_duplicate_tracks_do_not_double_count(metrics):
    truth = [(0, 3, "A"), (1, 2, "B"), (1, 2, "C"), (1, 2, "A")]
    hyp = [(0, 3, "x"), (1.5, 2.5, "y")]
    score = metrics.overlap_scores(truth, hyp)
    assert score["reference_seconds"] == score["detected_seconds"] == 1
    assert score["precision"] == score["recall"] == 0.5
    assert metrics.overlap_scores([], [])["precision"] is None
    assert metrics.score_diarization(truth, truth, duration=3)["der"] == 0


@pytest.mark.parametrize("seed", [3, 7, 19])
def test_generator_has_no_self_overlap_and_preserves_main_schedule(synth, seed):
    _, base = synth.build(synth.Scenario(seed=seed, crosstalk=True, repeats=2))
    mix, with_backchannels = synth.build(
        synth.Scenario(seed=seed, crosstalk=True, backchannels=True, repeats=2)
    )
    synth.validate_truth(with_backchannels)
    assert [p for p in with_backchannels if p.kind == "turn"] == base
    assert any(p.kind == "backchannel" for p in with_backchannels)
    again, same_truth = synth.build(
        synth.Scenario(seed=seed, crosstalk=True, backchannels=True, repeats=2)
    )
    assert same_truth == with_backchannels
    assert np.array_equal(mix, again)


def test_acoustic_toggles_preserve_utterance_windows(synth, monkeypatch):
    # Exercise the scheduling independently of filtering CPU cost.
    monkeypatch.setattr(synth, "SCRIPT", synth.SCRIPT[:6])
    _, base = synth.build(synth.Scenario(crosstalk=True, backchannels=True))
    _, acoustic = synth.build(
        synth.Scenario(crosstalk=True, backchannels=True, room=True, moving=True, noise=True)
    )
    assert base == acoustic


def test_invalid_ground_truth_rejected(synth):
    with pytest.raises(ValueError, match="self-overlap"):
        synth.validate_truth(
            [synth.Placed("A", 0, 2, "hej", "turn"), synth.Placed("A", 1, 3, "ja", "backchannel")]
        )


def test_service_replay_uploads_real_request_and_reads_review(metrics, tmp_path):
    import httpx

    from vemsa.jobs.models import JobRequest, TranscriptionResult

    replay = importlib.import_module("scripts.replay_synth_meeting")
    audio = tmp_path / "synthetic.wav"
    audio.write_bytes(b"synthetic fixture")
    fixture = json.loads((Path(__file__).parent / "fixtures/speaker_review.json").read_text())[
        "cases"
    ][2]["result"]
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "POST":
            assert b"include_speaker_review" in request.content
            assert b"true" in request.content
            assert b"file" in request.content
            return httpx.Response(202, json={"job_id": "test"})
        if request.url.path.endswith("/result"):
            return httpx.Response(200, json=fixture)
        return httpx.Response(200, json={"status": "completed"})

    with httpx.Client(
        transport=httpx.MockTransport(handle), base_url="https://vemsa.test/"
    ) as client:
        job_id = replay.submit(client, audio, JobRequest(include_speaker_review=True))
        result = replay.wait_result(client, job_id)
    assert result == TranscriptionResult.model_validate(fixture)
    assert len(calls) == 3


def test_service_replay_refuses_wrong_fixture_hash(metrics, tmp_path):
    replay = importlib.import_module("scripts.replay_synth_meeting")
    audio, truth = tmp_path / "a.wav", tmp_path / "truth.json"
    audio.write_bytes(b"different audio")
    truth.write_text(json.dumps({"generator_version": 2, "audio_sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash-matched"):
        replay.load_fixture(audio, truth)
