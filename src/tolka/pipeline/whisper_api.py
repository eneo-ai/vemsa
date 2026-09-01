"""Remote transcription via an OpenAI-compatible /audio/transcriptions endpoint
(e.g. speaches / faster-whisper-server / vLLM hosting KBLab/kb-whisper-large).

OpenAIWhisperEngine trusts the provider's timestamps (TOLKA_ENGINE=remote); the hybrid
engine reuses request_transcription/parse_verbose_json from here and force-aligns the
returned text locally instead. Diarization always runs locally via pyannote."""

import logging
import time
from pathlib import Path
from typing import Any

import httpx

from tolka.config import Settings
from tolka.jobs.models import Alignment, JobStage, Segment, SpeakerBounds, TranscriptionResult, Word
from tolka.pipeline.align import build_segment_aligner
from tolka.pipeline.base import StageReporter, report_stage
from tolka.pipeline.diarize import Diarizer, resolve_segments
from tolka.pipeline.label import label_speakers, segment_merge_alignment, words_plausible
from tolka.pipeline.render import render_text

logger = logging.getLogger(__name__)


def request_transcription(
    settings: Settings,
    audio_path: Path,
    *,
    language: str,
    model: str,
    vocabulary: list[str] | None = None,
) -> dict[str, Any]:
    """Blocking POST to the whisper endpoint; verbose_json with word+segment granularity."""
    if not settings.whisper_api_base:
        raise RuntimeError(
            "TOLKA_WHISPER_API_BASE is required (OpenAI-compatible base URL, e.g."
            " http://whisper-host:8000/v1)"
        )
    data: dict[str, Any] = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
    }
    if language != "auto":
        data["language"] = language
    elif settings.whisper_auto_language:
        data["language"] = settings.whisper_auto_language
    for item in settings.whisper_extra_form:
        key, _, value = item.partition("=")
        data[key] = value
    if vocabulary:
        # OpenAI-compatible decoder hint; a per-job vocabulary overrides a static
        # prompt from TOLKA_WHISPER_EXTRA_FORM.
        data["prompt"] = ", ".join(vocabulary)
    headers = {}
    if settings.whisper_api_key:
        headers["Authorization"] = f"Bearer {settings.whisper_api_key}"

    url = f"{settings.whisper_api_base.rstrip('/')}/audio/transcriptions"
    logger.info(
        "provider transcription requested",
        extra={
            "event": "provider.request",
            "provider": url,
            "model": model,
            "audio_bytes": audio_path.stat().st_size,
        },
    )
    started = time.perf_counter()
    with (
        audio_path.open("rb") as audio,
        httpx.Client(timeout=settings.whisper_timeout_s, headers=headers) as client,
    ):
        response = client.post(url, data=data, files={"file": (audio_path.name, audio)})
    if response.status_code != 200:
        raise RuntimeError(f"whisper API returned {response.status_code}: {response.text[:500]}")
    payload = response.json()
    logger.info(
        "provider transcription received",
        extra={
            "event": "provider.response",
            "provider": url,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "audio_seconds": payload.get("duration"),
            "segments": len(payload.get("segments") or []),
            "words": len(payload.get("words") or []),
        },
    )
    return payload


def parse_verbose_json(payload: dict[str, Any]) -> tuple[list[Word], list[Segment]]:
    """Extract words and plain (speakerless) segments from a verbose_json response.

    Either list may be empty: some servers return only segments even when word
    granularity was requested, and vice versa."""
    words = []
    for item in payload.get("words") or []:
        # faster-whisper-derived servers include a per-word probability; plain
        # OpenAI does not.
        probability = item.get("probability")
        words.append(
            Word(
                word=str(item["word"]).strip(),
                start=float(item["start"]),
                end=float(item["end"]),
                probability=float(probability) if probability is not None else None,
            )
        )
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


def build_result(
    payload: dict[str, Any],
    segments: list[Segment],
    *,
    model: str,
    language: str,
    alignment: Alignment | None = None,
) -> TranscriptionResult:
    duration = float(payload.get("duration") or 0.0) or (segments[-1].end if segments else 0.0)
    detected = payload.get("language") or (language if language != "auto" else "unknown")
    return TranscriptionResult(
        language=str(detected),
        duration_seconds=duration,
        model=model,
        text=render_text(segments),
        segments=segments,
        alignment=alignment,
    )


class OpenAIWhisperEngine:
    """Remote whisper, provider timestamps as-is; runs in a worker thread."""

    def __init__(self, settings: Settings, diarizer: Diarizer | None = None) -> None:
        if not settings.whisper_api_base:
            raise ValueError(
                "TOLKA_WHISPER_API_BASE is required (OpenAI-compatible base URL, e.g."
                " http://whisper-host:8000/v1)"
            )
        self._settings = settings
        self._diarizer = diarizer or Diarizer(settings)
        self._segment_aligner = build_segment_aligner(settings)

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        model: str,
        diarize: bool,
        speakers: SpeakerBounds | None = None,
        vocabulary: list[str] | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        report_stage(on_stage, JobStage.TRANSCRIBING)
        payload = request_transcription(
            self._settings, audio_path, language=language, model=model, vocabulary=vocabulary
        )
        words, plain_segments = parse_verbose_json(payload)
        if not words and not plain_segments:
            raise RuntimeError("whisper API returned neither words nor segments")
        if words and plain_segments and not words_plausible(words):
            logger.warning("provider word timestamps are implausible; merging speakers per segment")
            words = []
        if diarize and not words:
            logger.info("no word timestamps from whisper API; merging speakers per segment")

        if diarize:
            report_stage(on_stage, JobStage.DIARIZING)
        turns = self._diarizer.diarize(audio_path, speakers=speakers) if diarize else None
        segments = resolve_segments(words, plain_segments, turns)
        alignment: Alignment = (
            "provider_words" if words else segment_merge_alignment(plain_segments, segments)
        )
        return build_result(payload, segments, model=model, language=language, alignment=alignment)

    def label_speakers(
        self,
        audio_path: Path,
        *,
        words: list[Word],
        segments: list[Segment],
        language: str,
        model: str,
        speakers: SpeakerBounds | None = None,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        if not words:
            report_stage(on_stage, JobStage.ALIGNING)
        report_stage(on_stage, JobStage.DIARIZING)
        return label_speakers(
            self._diarizer,
            audio_path,
            words=words,
            segments=segments,
            language=language,
            model=model,
            aligner=self._segment_aligner,
            speakers=speakers,
            prefer_alignment=self._settings.diarize_prefer_align,
        )

    def warm_up(self) -> None:
        self._diarizer.load()
