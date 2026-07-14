"""Canned-output engine for local smoke runs (TOLKA_ENGINE=fake) — never for production."""

import time
from pathlib import Path

from tolka.jobs.models import Segment, TranscriptionResult, Word
from tolka.pipeline.render import render_text


class CannedEngine:
    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s

    def transcribe(
        self, audio_path: Path, *, language: str, model: str, diarize: bool
    ) -> TranscriptionResult:
        if self._delay_s:
            time.sleep(self._delay_s)
        words = [
            Word(word="detta", start=0.0, end=0.3),
            Word(word="är", start=0.35, end=0.5),
            Word(word="en", start=0.55, end=0.7),
            Word(word="fejkad", start=0.75, end=1.1),
            Word(word="transkribering", start=1.15, end=2.0),
        ]
        segment = Segment(
            start=0.0,
            end=2.0,
            speaker="SPEAKER_00" if diarize else None,
            text=" ".join(word.word for word in words),
            words=words,
        )
        return TranscriptionResult(
            language="sv" if language == "auto" else language,
            duration_seconds=2.0,
            model=model,
            text=render_text([segment]),
            segments=[segment],
        )

    def warm_up(self) -> None:
        pass
