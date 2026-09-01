"""EasyTranscriberEngine.transcribe against a stubbed easytranscriber.

The real pipeline needs the `local` extra and a GPU box; these tests pin the
call contract of easytranscriber 0.3.0 (required vad_model/audio_dir, bare
audio names relative to audio_dir, save_json on for the inter-step JSON file
handoff) and the SpeechSegment -> AlignmentSegment -> WordSegment return shape.
"""

import sys
import types
from pathlib import Path

from test_align import FakeAlignedWord, FakeSpeechSegment
from vemsa.config import Settings
from vemsa.pipeline.transcribe import EasyTranscriberEngine


def stub_easytranscriber(monkeypatch, calls: dict) -> None:
    def fake_pipeline(**kwargs):
        calls.update(kwargs)
        return [
            [
                FakeSpeechSegment(
                    [
                        FakeAlignedWord("hej", 0.0, 0.4, score=0.9),
                        FakeAlignedWord("då", 0.5, 0.7),
                    ]
                )
            ]
        ]

    package = types.ModuleType("easytranscriber")
    pipelines = types.ModuleType("easytranscriber.pipelines")
    pipelines.pipeline = fake_pipeline
    package.pipelines = pipelines
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "easytranscriber", package)
    monkeypatch.setitem(sys.modules, "easytranscriber.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_transcribe_call_matches_easytranscriber_contract(tmp_path: Path, monkeypatch):
    calls: dict = {}
    stub_easytranscriber(monkeypatch, calls)
    settings = Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        work_dir=tmp_path / "work",
        model_cache_dir=tmp_path / "models",
    )
    engine = EasyTranscriberEngine(settings)
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake audio")

    result = engine.transcribe(audio, language="sv", model="kb-whisper", diarize=False)

    assert calls["vad_model"] == "silero"
    assert calls["transcription_model"] == "kb-whisper"
    assert calls["audio_paths"] == ["meeting.wav"]
    assert calls["audio_dir"] == str(tmp_path)
    assert calls["save_json"] is True
    assert calls["language"] == "sv"
    assert calls["device"] == "cpu"
    # intermediate files go to a throwaway directory under work_dir
    assert calls["output_vad_dir"].startswith(str(tmp_path / "work"))

    assert [w.word for w in result.segments[0].words] == ["hej", "då"]
    assert result.segments[0].words[0].probability == 0.9
    assert result.alignment == "forced"
    assert result.language == "sv"


def test_transcribe_auto_language_maps_to_none(tmp_path: Path, monkeypatch):
    calls: dict = {}
    stub_easytranscriber(monkeypatch, calls)
    settings = Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        work_dir=tmp_path / "work",
        model_cache_dir=tmp_path / "models",
    )
    engine = EasyTranscriberEngine(settings)
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake audio")

    engine.transcribe(audio, language="auto", model="kb-whisper", diarize=False)

    assert calls["language"] is None
