from pathlib import Path

import pytest
import respx
from httpx import Response

from tolka.config import Settings
from tolka.pipeline.diarize import Turn
from tolka.pipeline.whisper_api import OpenAIWhisperEngine, parse_verbose_json

API_BASE = "http://whisper.local/v1"
ENDPOINT = f"{API_BASE}/audio/transcriptions"

WORD_PAYLOAD = {
    "task": "transcribe",
    "language": "sv",
    "duration": 2.6,
    "text": "Hej och välkomna. Tack så mycket.",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.4, "text": " Hej och välkomna."},
        {"id": 1, "start": 2.0, "end": 2.6, "text": " Tack så mycket."},
    ],
    "words": [
        {"word": "Hej", "start": 0.0, "end": 0.4},
        {"word": "och", "start": 0.5, "end": 0.7},
        {"word": "välkomna.", "start": 0.8, "end": 1.4},
        {"word": "Tack", "start": 2.0, "end": 2.2},
        {"word": "så", "start": 2.3, "end": 2.4},
        {"word": "mycket.", "start": 2.45, "end": 2.6},
    ],
}

SEGMENT_ONLY_PAYLOAD = {
    "language": "sv",
    "duration": 2.6,
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.4, "text": " Hej och välkomna."},
        {"id": 1, "start": 2.0, "end": 2.6, "text": " Tack så mycket."},
    ],
}

TURNS = [Turn(0.0, 1.5, "SPEAKER_00"), Turn(1.9, 2.7, "SPEAKER_01")]


class FakeDiarizer:
    def __init__(self, turns=TURNS):
        self.turns = turns
        self.calls: list[Path] = []
        self.speaker_bounds: list[object] = []

    def diarize(self, audio_path: Path, *, speakers=None):
        self.calls.append(audio_path)
        self.speaker_bounds.append(speakers)
        return self.turns

    def load(self):
        pass


@pytest.fixture
def whisper_settings(settings: Settings) -> Settings:
    settings.whisper_api_base = API_BASE
    return settings


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fake audio")
    return audio


def test_requires_api_base(settings: Settings):
    settings.whisper_api_base = ""
    with pytest.raises(ValueError, match="TOLKA_WHISPER_API_BASE"):
        OpenAIWhisperEngine(settings)


@respx.mock
def test_word_level_diarized_transcription(whisper_settings: Settings, audio_file: Path):
    route = respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
    engine = OpenAIWhisperEngine(whisper_settings, diarizer=FakeDiarizer())

    result = engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=True)

    assert result.language == "sv"
    assert result.duration_seconds == 2.6
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.segments[0].text == "Hej och välkomna."
    assert "SPEAKER_00" in result.text

    request = route.calls.last.request
    body = request.read()
    assert b'name="language"' in body and b"sv" in body
    assert body.count(b'name="timestamp_granularities[]"') == 2


@respx.mock
def test_auto_language_fallback_is_sent(whisper_settings: Settings, audio_file: Path):
    whisper_settings.whisper_auto_language = "sv"
    route = respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
    engine = OpenAIWhisperEngine(whisper_settings, diarizer=FakeDiarizer())

    engine.transcribe(audio_file, language="auto", model="kb-whisper", diarize=False)

    body = route.calls.last.request.read()
    assert b'name="language"' in body and b"sv" in body


@respx.mock
def test_extra_form_fields_are_sent(whisper_settings: Settings, audio_file: Path):
    whisper_settings.whisper_extra_form = ["align=true"]
    route = respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
    engine = OpenAIWhisperEngine(whisper_settings, diarizer=FakeDiarizer())

    engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=False)

    body = route.calls.last.request.read()
    assert b'name="align"' in body and b"true" in body


@respx.mock
def test_segment_only_fallback_assigns_speakers_per_segment(
    whisper_settings: Settings, audio_file: Path
):
    respx.post(ENDPOINT).mock(return_value=Response(200, json=SEGMENT_ONLY_PAYLOAD))
    engine = OpenAIWhisperEngine(whisper_settings, diarizer=FakeDiarizer())

    result = engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=True)

    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.segments[0].words == []
    assert result.segments[0].text == "Hej och välkomna."


@respx.mock
def test_no_diarize_keeps_provider_segments(whisper_settings: Settings, audio_file: Path):
    respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
    diarizer = FakeDiarizer()
    engine = OpenAIWhisperEngine(whisper_settings, diarizer=diarizer)

    result = engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=False)

    assert diarizer.calls == []
    assert all(s.speaker is None for s in result.segments)
    # top-level words are attached to their segment by midpoint
    assert [w.word for w in result.segments[0].words] == ["Hej", "och", "välkomna."]


@respx.mock
def test_auto_language_omits_language_field(whisper_settings: Settings, audio_file: Path):
    route = respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
    engine = OpenAIWhisperEngine(whisper_settings, diarizer=FakeDiarizer())

    engine.transcribe(audio_file, language="auto", model="kb-whisper", diarize=False)

    assert b'name="language"' not in route.calls.last.request.read()


@respx.mock
def test_api_key_sent_as_bearer(whisper_settings: Settings, audio_file: Path):
    whisper_settings.whisper_api_key = "whisper-secret"
    route = respx.post(ENDPOINT).mock(return_value=Response(200, json=WORD_PAYLOAD))
    engine = OpenAIWhisperEngine(whisper_settings, diarizer=FakeDiarizer())

    engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=False)

    assert route.calls.last.request.headers["authorization"] == "Bearer whisper-secret"


@respx.mock
def test_error_response_raises(whisper_settings: Settings, audio_file: Path):
    respx.post(ENDPOINT).mock(return_value=Response(500, text="model not loaded"))
    engine = OpenAIWhisperEngine(whisper_settings, diarizer=FakeDiarizer())

    with pytest.raises(RuntimeError, match="whisper API returned 500"):
        engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=True)


@respx.mock
def test_empty_response_raises(whisper_settings: Settings, audio_file: Path):
    respx.post(ENDPOINT).mock(return_value=Response(200, json={"text": "hej"}))
    engine = OpenAIWhisperEngine(whisper_settings, diarizer=FakeDiarizer())

    with pytest.raises(RuntimeError, match="neither words nor segments"):
        engine.transcribe(audio_file, language="sv", model="kb-whisper", diarize=True)


def test_parse_verbose_json_nests_words_by_midpoint():
    words, segments = parse_verbose_json(WORD_PAYLOAD)
    assert len(words) == 6
    assert [w.word for w in segments[1].words] == ["Tack", "så", "mycket."]
    assert segments[0].speaker is None
