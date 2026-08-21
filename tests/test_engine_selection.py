import pytest

from tolka.config import Settings
from tolka.pipeline.factory import build_engine
from tolka.pipeline.fake import CannedEngine
from tolka.pipeline.hybrid import HybridEngine
from tolka.pipeline.transcribe import EasyTranscriberEngine
from tolka.pipeline.whisper_api import OpenAIWhisperEngine


def test_auto_resolves_by_whisper_endpoint(settings: Settings):
    settings.engine = "auto"
    settings.whisper_api_base = ""
    assert settings.resolve_engine() == "local"
    settings.whisper_api_base = "http://whisper.local/v1"
    assert settings.resolve_engine() == "hybrid"


def test_explicit_engine_passes_through(settings: Settings):
    for engine in ("local", "hybrid", "remote", "fake"):
        settings.engine = engine
        assert settings.resolve_engine() == engine


def test_build_engine_returns_expected_classes(settings: Settings):
    settings.whisper_api_base = "http://whisper.local/v1"

    settings.engine = "fake"
    assert isinstance(build_engine(settings), CannedEngine)
    settings.engine = "local"
    assert isinstance(build_engine(settings), EasyTranscriberEngine)
    settings.engine = "hybrid"
    assert isinstance(build_engine(settings), HybridEngine)
    settings.engine = "remote"
    assert isinstance(build_engine(settings), OpenAIWhisperEngine)


def test_auto_without_endpoint_builds_local_engine(settings: Settings):
    settings.engine = "auto"
    settings.whisper_api_base = ""
    assert isinstance(build_engine(settings), EasyTranscriberEngine)


@pytest.mark.parametrize("engine_cls", [HybridEngine, OpenAIWhisperEngine])
def test_remote_backed_engines_require_api_base(settings: Settings, engine_cls):
    settings.whisper_api_base = ""
    with pytest.raises(ValueError, match="TOLKA_WHISPER_API_BASE"):
        engine_cls(settings)
