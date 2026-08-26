from pathlib import Path

import pytest
import respx
from httpx import Response

from test_whisper_api import API_BASE, ENDPOINT, WORD_PAYLOAD, FakeDiarizer
from tolka.config import Settings
from tolka.jobs.models import JobStage, Word
from tolka.pipeline.hybrid import HybridEngine

ALIGNED_WORDS = [
    Word(word="Hej", start=0.05, end=0.38),
    Word(word="och", start=0.52, end=0.68),
    Word(word="välkomna.", start=0.81, end=1.35),
    Word(word="Tack", start=2.02, end=2.19),
    Word(word="så", start=2.31, end=2.42),
    Word(word="mycket.", start=2.44, end=2.58),
]


@pytest.fixture
def hybrid_settings(settings: Settings) -> Settings:
    settings.whisper_api_base = API_BASE
    return settings


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake audio")
    return audio


@respx.mock
def test_uses_locally_aligned_words_when_alignment_succeeds(
    hybrid_settings: Settings, audio_file: Path, monkeypatch
):
    respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
    engine = HybridEngine(hybrid_settings, diarizer=FakeDiarizer())
    monkeypatch.setattr(engine, "_force_align", lambda *args, **kwargs: ALIGNED_WORDS)
    stages: list[JobStage] = []

    result = engine.transcribe(
        audio_file,
        language="sv",
        model="kb-whisper",
        diarize=True,
        on_stage=stages.append,
    )

    # word timestamps come from local alignment, not the provider payload
    assert result.segments[0].words[0].start == 0.05
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert stages == [
        JobStage.TRANSCRIBING,
        JobStage.ALIGNING,
        JobStage.DIARIZING,
    ]


@respx.mock
def test_falls_back_to_provider_timestamps_when_alignment_fails(
    hybrid_settings: Settings, audio_file: Path, monkeypatch
):
    respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
    engine = HybridEngine(hybrid_settings, diarizer=FakeDiarizer())

    def boom(*args, **kwargs):
        raise ImportError("No module named 'easyaligner'")

    monkeypatch.setattr(engine, "_force_align", boom)

    result = engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=True)

    # provider word timestamps survive the fallback
    assert result.segments[0].words[0].start == 0.0
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]


@respx.mock
def test_implausible_provider_words_fall_back_to_segments_not_words(
    hybrid_settings: Settings, audio_file: Path, monkeypatch
):
    # ~10 words/s decoder-heuristic timeline alongside segments; alignment also
    # fails — the segment merge must win over the garbage word timeline
    compressed = dict(WORD_PAYLOAD)
    compressed["words"] = [
        {"word": f"w{i}", "start": i * 0.1, "end": i * 0.1 + 0.08} for i in range(26)
    ]
    respx.post(ENDPOINT).mock(return_value=Response(200, json=compressed))
    engine = HybridEngine(hybrid_settings, diarizer=FakeDiarizer())

    def boom(*args, **kwargs):
        raise ImportError("No module named 'easyaligner'")

    monkeypatch.setattr(engine, "_force_align", boom)

    result = engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=True)

    assert result.alignment in ("segment_only", "segment_split")
    # output segments come from the provider segments, not the compressed words
    assert [s.text for s in result.segments] == ["Hej och välkomna.", "Tack så mycket."]
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]


@respx.mock
def test_no_diarize_with_alignment_groups_aligned_words(
    hybrid_settings: Settings, audio_file: Path, monkeypatch
):
    respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
    diarizer = FakeDiarizer()
    engine = HybridEngine(hybrid_settings, diarizer=diarizer)
    monkeypatch.setattr(engine, "_force_align", lambda *args, **kwargs: ALIGNED_WORDS)

    result = engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=False)

    assert diarizer.calls == []
    assert all(s.speaker is None for s in result.segments)
    assert result.segments[0].words[0].start == 0.05
