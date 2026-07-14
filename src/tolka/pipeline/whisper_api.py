"""Transcription engine backed by a remote OpenAI-compatible /audio/transcriptions
endpoint (e.g. speaches / faster-whisper-server / vLLM hosting KBLab/kb-whisper-large).

Word-level timestamps are requested; when the serving stack only returns segments,
speaker assignment degrades gracefully to segment granularity. Diarization always
runs locally via pyannote."""

import logging
from pathlib import Path
from typing import Any

import httpx

from tolka.config import Settings
from tolka.jobs.models import Segment, TranscriptionResult, Word
from tolka.pipeline.diarize import (
    Diarizer,
    assign_speakers,
    assign_speakers_to_segments,
    segments_without_speakers,
)
from tolka.pipeline.render import render_text

logger = logging.getLogger(__name__)


def parse_verbose_json(payload: dict[str, Any]) -> tuple[list[Word], list[Segment]]:
    """Extract words and plain (speakerless) segments from a verbose_json response.

    Either list may be empty: some servers return only segments even when word
    granularity was requested, and vice versa."""
    words = [
        Word(word=str(item["word"]).strip(), start=float(item["start"]), end=float(item["end"]))
        for item in payload.get("words") or []
    ]
    words.sort(key=lambda word: word.start)

    segments = []
    for item in payload.get("segments") or []:
        start, end = float(item["start"]), float(item["end"])
        segments.append(
            Segment(
                start=start,
                end=end,
                text=str(item["text"]).strip(),
                words=[w for w in words if start <= (w.start + w.end) / 2 < end],
            )
        )
    segments.sort(key=lambda segment: segment.start)
    return words, segments


class OpenAIWhisperEngine:
    """Blocking engine calling the remote whisper endpoint; runs in a worker thread."""

    def __init__(self, settings: Settings, diarizer: Diarizer | None = None) -> None:
        if not settings.whisper_api_base:
            raise ValueError(
                "TOLKA_WHISPER_API_BASE is required (OpenAI-compatible base URL, e.g."
                " http://whisper-host:8000/v1)"
            )
        self._settings = settings
        self._diarizer = diarizer or Diarizer(settings)

    def transcribe(
        self, audio_path: Path, *, language: str, model: str, diarize: bool
    ) -> TranscriptionResult:
        payload = self._request_transcription(audio_path, language=language, model=model)
        words, plain_segments = parse_verbose_json(payload)
        if not words and not plain_segments:
            raise RuntimeError("whisper API returned neither words nor segments")

        if diarize:
            turns = self._diarizer.diarize(audio_path)
            if words:
                segments = assign_speakers(words, turns)
            else:
                logger.info("no word timestamps from whisper API; merging speakers per segment")
                segments = assign_speakers_to_segments(plain_segments, turns)
        elif plain_segments:
            segments = plain_segments
        else:
            segments = segments_without_speakers(words)

        duration = float(payload.get("duration") or 0.0) or (segments[-1].end if segments else 0.0)
        detected = payload.get("language") or (language if language != "auto" else "unknown")
        return TranscriptionResult(
            language=str(detected),
            duration_seconds=duration,
            model=model,
            text=render_text(segments),
            segments=segments,
        )

    def warm_up(self) -> None:
        self._diarizer.load()

    def _request_transcription(
        self, audio_path: Path, *, language: str, model: str
    ) -> dict[str, Any]:
        settings = self._settings
        data: dict[str, Any] = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        }
        if language != "auto":
            data["language"] = language
        headers = {}
        if settings.whisper_api_key:
            headers["Authorization"] = f"Bearer {settings.whisper_api_key}"

        url = f"{settings.whisper_api_base.rstrip('/')}/audio/transcriptions"
        with (
            audio_path.open("rb") as audio,
            httpx.Client(timeout=settings.whisper_timeout_s, headers=headers) as client,
        ):
            response = client.post(url, data=data, files={"file": (audio_path.name, audio)})
        if response.status_code != 200:
            raise RuntimeError(
                f"whisper API returned {response.status_code}: {response.text[:500]}"
            )
        return response.json()
