"""Canned-output engine for local smoke runs (VEMSA_ENGINE=fake) — never for production."""

import time
from pathlib import Path

from vemsa.jobs.models import JobStage, Segment, SpeakerBounds, TranscriptionResult, Word
from vemsa.pipeline.base import StageReporter, report_stage
from vemsa.pipeline.diarize import resolve_segments
from vemsa.pipeline.render import render_text


class CannedEngine:
    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s

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
        if self._delay_s:
            time.sleep(self._delay_s)
        words = [
            Word(word="detta", start=0.0, end=0.3, probability=0.99),
            Word(word="är", start=0.35, end=0.5, probability=0.97),
            Word(word="en", start=0.55, end=0.7),
            Word(word="fejkad", start=0.75, end=1.1, probability=0.42),
            Word(word="transkribering", start=1.15, end=2.0, probability=0.88),
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
        """Alternate SPEAKER_00/SPEAKER_01 per segment so consumers can test labelling."""
        report_stage(on_stage, JobStage.DIARIZING)
        if self._delay_s:
            time.sleep(self._delay_s)
        plain = [segment.model_copy(update={"speaker": None}) for segment in segments]
        grouped = resolve_segments(words, plain, None)
        labelled = [
            segment.model_copy(update={"speaker": f"SPEAKER_{index % 2:02d}"})
            for index, segment in enumerate(grouped)
        ]
        return TranscriptionResult(
            language=language if language != "auto" else "unknown",
            duration_seconds=max((segment.end for segment in labelled), default=0.0),
            model=model,
            text=render_text(labelled),
            segments=labelled,
        )

    def align_transcript(
        self,
        audio_path: Path,
        *,
        segments: list[Segment],
        language: str,
        model: str,
        on_stage: StageReporter | None = None,
    ) -> TranscriptionResult:
        """Spread evenly spaced words over each segment's window, speakers and
        text verbatim, so consumers can test the align contract end to end."""
        report_stage(on_stage, JobStage.ALIGNING)
        if self._delay_s:
            time.sleep(self._delay_s)
        aligned: list[Segment] = []
        for segment in sorted(segments, key=lambda item: item.start):
            tokens = segment.text.split()
            if not tokens:
                aligned.append(segment.model_copy(update={"words": []}))
                continue
            step = (segment.end - segment.start) / len(tokens)
            words = [
                Word(
                    word=token,
                    start=round(segment.start + index * step, 3),
                    end=round(segment.start + (index + 1) * step, 3),
                    probability=0.9,
                )
                for index, token in enumerate(tokens)
            ]
            aligned.append(
                segment.model_copy(
                    update={"start": words[0].start, "end": words[-1].end, "words": words}
                )
            )
        return TranscriptionResult(
            language=language if language != "auto" else "unknown",
            duration_seconds=max((segment.end for segment in aligned), default=0.0),
            model=model,
            text=render_text(aligned),
            segments=aligned,
            alignment="forced",
        )

    def warm_up(self) -> None:
        pass
