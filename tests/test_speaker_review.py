"""Overlap evidence is additive, conservative, and survives every service boundary."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import respx
from httpx import Response
from pydantic import ValidationError

from conftest import FakeEngine, make_result
from test_api_jobs import AUTH, api_client, poll_until
from test_local_engine import stub_easytranscriber
from test_whisper_api import API_BASE, ENDPOINT, WORD_PAYLOAD
from vemsa.config import Settings
from vemsa.jobs.models import (
    JobRequest,
    Segment,
    SpeakerBounds,
    SpeakerReview,
    TranscriptionResult,
    Word,
)
from vemsa.jobs.postgres_store import PostgresJobStore
from vemsa.pipeline.diarize import Diarizer, Turn
from vemsa.pipeline.diarize_only import DiarizeOnlyEngine
from vemsa.pipeline.fake import CannedEngine
from vemsa.pipeline.hybrid import HybridEngine
from vemsa.pipeline.label import collect_diarization, label_speakers
from vemsa.pipeline.speaker_review import apply_speaker_review, detect_overlaps
from vemsa.pipeline.transcribe import EasyTranscriberEngine
from vemsa.pipeline.whisper_api import OpenAIWhisperEngine


def review_between(start=0.5, end=0.7):
    return detect_overlaps([Turn(0, 3, "A"), Turn(start, end, "B")])


def result_of(*segments):
    return TranscriptionResult(
        language="sv", duration_seconds=4, model="test", text="original", segments=list(segments)
    )


def test_overlap_sweep_distinct_speakers_half_open_and_changing_sets():
    review = detect_overlaps(
        [
            Turn(0, 6, "A"),
            Turn(0.5, 2, "B"),
            Turn(1, 2, "B"),
            Turn(2, 3, "B"),
            Turn(3, 4, "C"),
            Turn(3.5, 5, "D"),
            Turn(5, 5, "E"),
        ]
    )
    assert [(o.start, o.end, o.detected_speaker_count) for o in review.overlaps] == [
        (0.5, 3, 2),
        (3, 3.5, 2),
        (3.5, 4, 3),
        (4, 5, 2),
    ]
    assert [o.id for o in review.overlaps] == [f"overlap_{i:04d}" for i in range(4)]
    assert detect_overlaps([Turn(0, 1, "A"), Turn(1, 2, "B")]).overlaps == []
    assert detect_overlaps([Turn(0, 2, "A"), Turn(1, 3, "A")]).overlaps == []
    assert detect_overlaps([]).overlap_detection == "available"


def test_overlap_sweep_gaps_and_invalid_timing():
    review = detect_overlaps([Turn(-1, 4, "A"), Turn(-0.5, 1, "B"), Turn(2, 3, "B")])
    assert [(o.start, o.end) for o in review.overlaps] == [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match="finite"):
        detect_overlaps([Turn(0, float("nan"), "A")])


def test_splits_after_smoothing_preserving_words_and_proposed_speaker():
    original = make_result()
    reviewed = apply_speaker_review(original, review_between())
    assert [s.text for s in reviewed.segments] == ["hej", "och", "välkomna", "tack så mycket"]
    assert [s.speaker_attribution for s in reviewed.segments] == [
        "assigned",
        "provisional",
        "assigned",
        "assigned",
    ]
    assert reviewed.segments[1].speaker == "SPEAKER_00"
    assert reviewed.segments[1].overlap_ids == ["overlap_0000"]
    assert [w for s in reviewed.segments for w in s.words] == [
        w for s in original.segments for w in s.words
    ]
    assert "[Överlappande tal – osäker talare]: och" in reviewed.text
    assert "SPEAKER_00: och" not in reviewed.text
    assert original.speaker_review is None
    assert original.segments[0].speaker_attribution is None
    assert apply_speaker_review(original, None) is original


@pytest.mark.parametrize(
    "text,words",
    [
        (
            "Hej, allihopa!",
            [Word(word="Hej", start=0, end=0.4), Word(word="allihopa", start=0.5, end=0.7)],
        ),
        (
            "Hej, allihopa!",
            [Word(word="hejsan", start=0, end=0.4), Word(word="allihopa", start=0.5, end=0.7)],
        ),
        ("Hej, allihopa!", []),
    ],
)
def test_punctuation_and_unanchored_or_wordless_fallback(text, words):
    result = apply_speaker_review(
        result_of(Segment(start=0, end=1, speaker="A", text=text, words=words)), review_between()
    )
    assert " ".join(s.text for s in result.segments) == text
    assert result.segments[-1].speaker_attribution == "provisional"
    if not words or words[0].word == "hejsan":
        assert len(result.segments) == 1


def test_zero_length_words_boundaries_and_unknown_are_not_invented_speakers():
    source = result_of(
        Segment(start=0, end=0.5, text="hej", words=[Word(word="hej", start=0, end=0.5)]),
        Segment(start=0.6, end=0.6, text="mm", words=[Word(word="mm", start=0.6, end=0.6)]),
        Segment(start=0.6, end=0.8, text="ja"),
    )
    result = apply_speaker_review(source, review_between())
    assert [s.speaker_attribution for s in result.segments] == [
        "unassigned",
        "unassigned",
        "provisional",
    ]
    assert all(s.speaker is None for s in result.segments)


def test_unavailable_and_legacy_results_are_distinct_from_clean():
    result = make_result()
    unavailable = apply_speaker_review(result, SpeakerReview(overlap_detection="unavailable"))
    clean = apply_speaker_review(result, detect_overlaps([]))
    assert unavailable.speaker_review.overlap_detection == "unavailable"
    assert clean.speaker_review.overlap_detection == "available"
    legacy = result.model_dump(exclude={"speaker_review"})
    for segment in legacy["segments"]:
        segment.pop("speaker_attribution")
        segment.pop("overlap_ids")
    assert TranscriptionResult.model_validate(legacy) == result


@pytest.mark.parametrize(
    "payload",
    [
        {"diarize": False, "include_speaker_review": True},
        {"task": "align", "include_speaker_review": True},
        {
            "task": "align",
            "segments": [{"start": 0, "end": 1, "text": "hej", "speaker_attribution": "assigned"}],
        },
        {
            "task": "align",
            "segments": [{"start": 0, "end": 1, "text": "hej", "overlap_ids": ["o1"]}],
        },
        {"task": "align", "speaker_review": {"version": 1}},
    ],
)
def test_incompatible_requests_rejected(payload):
    with pytest.raises(ValidationError, match="speaker.review"):
        JobRequest.model_validate(payload)


class Annotation:
    def __init__(self, turns):
        self.turns = turns

    def itertracks(self, yield_label):
        assert yield_label
        return iter(
            (SimpleNamespace(start=t.start, end=t.end), i, t.speaker)
            for i, t in enumerate(self.turns)
        )


def local_settings(tmp_path):
    return Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        work_dir=tmp_path / "work",
        model_cache_dir=tmp_path / "models",
        whisper_api_base=API_BASE,
        diarize_prefer_align=False,
    )


@pytest.mark.parametrize("exclusive", [True, False])
def test_single_inference_keeps_raw_evidence_and_cleans_decoded_file(
    tmp_path, monkeypatch, exclusive
):
    settings = local_settings(tmp_path)
    settings.diarize_exclusive = exclusive
    engine = Diarizer(settings)
    raw = [Turn(0, 3, "A"), Turn(0.5, 0.7, "B")]
    chosen = [Turn(0, 3, "A")]
    calls = []
    engine._pipeline = lambda *args, **kwargs: (
        calls.append((args, kwargs))
        or SimpleNamespace(
            speaker_diarization=Annotation(raw), exclusive_speaker_diarization=Annotation(chosen)
        )
    )
    decoded = tmp_path / "decoded.wav"
    decoded.write_bytes(b"test")
    monkeypatch.setattr("vemsa.pipeline.diarize._decodable_audio", lambda _: (decoded, True))
    turns, review = engine.diarize_with_review(
        tmp_path / "input.mp3", speakers=SpeakerBounds(max_speakers=4)
    )
    assert turns == (chosen if exclusive else raw)
    assert review == review_between()
    assert len(calls) == 1 and calls[0][1] == {"max_speakers": 4}
    assert not decoded.exists()


def test_bare_annotation_reports_unavailable(tmp_path, monkeypatch):
    diarizer = Diarizer(local_settings(tmp_path))
    diarizer._pipeline = lambda *a, **k: Annotation([Turn(0, 1, "A")])
    monkeypatch.setattr("vemsa.pipeline.diarize._decodable_audio", lambda p: (p, False))
    _, review = diarizer.diarize_with_review(tmp_path / "a.wav")
    assert review.overlap_detection == "unavailable"


class ReviewDiarizer:
    def __init__(self):
        self.calls = []

    def diarize(self, path, *, speakers=None):
        self.calls.append((False, speakers))
        return [Turn(0, 3, "SPEAKER_00")]

    def diarize_with_review(self, path, *, speakers=None):
        self.calls.append((True, speakers))
        return [Turn(0, 3, "SPEAKER_00")], review_between()


def test_custom_legacy_diarizer_does_not_claim_clean_audio(tmp_path):
    calls = []
    custom = SimpleNamespace(diarize=lambda *a, **k: calls.append(1) or [])
    _, review = collect_diarization(custom, tmp_path / "audio", include_speaker_review=True)
    assert calls == [1]
    assert review.overlap_detection == "unavailable"


@pytest.mark.parametrize(
    "engine_class", [OpenAIWhisperEngine, HybridEngine, EasyTranscriberEngine, DiarizeOnlyEngine]
)
def test_all_diarize_engines_keep_review_and_caller_label(tmp_path, engine_class):
    engine = engine_class(local_settings(tmp_path))
    diarizer = engine._diarizer = ReviewDiarizer()
    source = make_result().segments[:1]
    source[0].speaker = "Anna"
    result = engine.label_speakers(
        tmp_path / "a.wav",
        words=source[0].words,
        segments=source,
        language="sv",
        model="external",
        include_speaker_review=True,
        speakers=SpeakerBounds(max_speakers=4),
    )
    assert result.speaker_review == review_between()
    assert any(
        s.speaker_attribution == "provisional" and s.speaker == "Anna" for s in result.segments
    )
    assert diarizer.calls == [(True, SpeakerBounds(max_speakers=4))]


@pytest.mark.parametrize("engine_class", [OpenAIWhisperEngine, HybridEngine, EasyTranscriberEngine])
@respx.mock
def test_all_transcribe_engines_keep_review(tmp_path, monkeypatch, engine_class):
    engine = engine_class(local_settings(tmp_path))
    diarizer = engine._diarizer = ReviewDiarizer()
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"test")
    if engine_class is EasyTranscriberEngine:
        stub_easytranscriber(monkeypatch, {})
    else:
        respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
        if engine_class is HybridEngine:
            monkeypatch.setattr(engine, "_force_align", lambda *a: make_result().segments[0].words)
    result = engine.transcribe(
        audio, language="sv", model="test", diarize=True, include_speaker_review=True
    )
    assert result.speaker_review == review_between()
    assert any(s.speaker_attribution == "provisional" for s in result.segments)
    assert diarizer.calls == [(True, None)]


def test_review_recomputed_when_diarizing_a_tagged_transcript(tmp_path):
    source = apply_speaker_review(make_result(), review_between())
    result = label_speakers(
        ReviewDiarizer(),
        tmp_path / "audio",
        words=[w for s in source.segments for w in s.words],
        segments=source.segments,
        language="sv",
        model="external",
        include_speaker_review=False,
    )
    assert result.speaker_review is None
    assert all(s.speaker_attribution is None and not s.overlap_ids for s in result.segments)


@pytest.mark.parametrize("task", ["transcribe", "diarize"])
@pytest.mark.parametrize("transport", ["json", "multipart"])
@respx.mock
async def test_opt_in_survives_submission_queue_and_database(settings, task, transport):
    fields = {"task": task, "include_speaker_review": True}
    if task == "diarize":
        fields["segments"] = [{"start": 0, "end": 2, "text": "hej allihopa"}]
    respx.get("https://example.org/test.wav").mock(
        return_value=Response(200, content=b"fake audio")
    )
    async with api_client(settings, CannedEngine()) as (client, app):
        if transport == "json":
            response = await client.post(
                "/v1/jobs",
                json={**fields, "source_url": "https://example.org/test.wav"},
                headers=AUTH,
            )
        else:
            response = await client.post(
                "/v1/jobs",
                files={"file": ("a.wav", b"fake")},
                data={k: json.dumps(v) if not isinstance(v, str) else v for k, v in fields.items()},
                headers=AUTH,
            )
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        await poll_until(client, job_id, "completed")
        result = (await client.get(f"/v1/jobs/{job_id}/result", headers=AUTH)).json()
        assert result["speaker_review"]["overlap_detection"] == "available"
        assert any(s["speaker_attribution"] == "provisional" for s in result["segments"])
        assert (await app.state.deps.store.get(job_id)).request.include_speaker_review is True
        # A fresh storage instance reads the result, not the worker's in-memory object.
        store = PostgresJobStore(settings.database_url)
        await store.open()
        try:
            assert (await store.get_result(job_id)).model_dump(mode="json") == result
        finally:
            await store.close()


async def test_api_rejects_incompatible_review_before_work(settings):
    engine = FakeEngine()
    async with api_client(settings, engine) as (client, _):
        for fields in (
            {"diarize": "false"},
            {"task": "align", "segments": '[{"start":0,"end":1,"text":"hej"}]'},
        ):
            response = await client.post(
                "/v1/jobs",
                files={"file": ("a.wav", b"audio")},
                data={**fields, "include_speaker_review": "true"},
                headers=AUTH,
            )
            assert response.status_code == 422
        assert engine.calls == []


def test_shared_contract_fixtures():
    path = Path(__file__).parent / "fixtures" / "speaker_review.json"
    fixtures = json.loads(path.read_text())
    for case in fixtures["cases"]:
        source = TranscriptionResult.model_validate(case["input"])
        review = SpeakerReview.model_validate(case["review"]) if case["review"] else None
        result = apply_speaker_review(source, review)
        assert result.model_dump(mode="json") == case["result"], case["name"]
        known = {o.id for o in result.speaker_review.overlaps} if result.speaker_review else set()
        assert all(set(s.overlap_ids) <= known for s in result.segments)
