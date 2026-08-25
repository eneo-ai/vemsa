"""Speaker diarization and word-to-speaker assignment.

The assignment logic is pure and CI-tested; the pyannote-backed Diarizer lives here
too but is only imported/constructed when the ML extra is installed.
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tolka.jobs.models import Segment, Word

if TYPE_CHECKING:
    from tolka.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def _overlap(word: Word, turn: Turn) -> float:
    return max(0.0, min(word.end, turn.end) - max(word.start, turn.start))


def _nearest_turn(word: Word | Segment, turns: list[Turn]) -> Turn:
    midpoint = (word.start + word.end) / 2
    return min(turns, key=lambda turn: abs((turn.start + turn.end) / 2 - midpoint))


def _label_words(words: list[Word], turns: list[Turn]) -> list[str]:
    """Per word, the speaker of the maximally overlapping turn; zero overlap inherits
    the previous word's speaker (nearest turn by midpoint for the first word)."""
    labels: list[str] = []
    first_candidate = 0
    for word in words:
        while first_candidate < len(turns) and turns[first_candidate].end <= word.start:
            first_candidate += 1
        best: str | None = None
        best_overlap = 0.0
        index = first_candidate
        while index < len(turns) and turns[index].start < word.end:
            overlap = _overlap(word, turns[index])
            if overlap > best_overlap:
                best_overlap = overlap
                best = turns[index].speaker
            index += 1
        if best is None:
            best = labels[-1] if labels else _nearest_turn(word, turns).speaker
        labels.append(best)
    return labels


def assign_speakers(
    words: list[Word], turns: list[Turn], *, gap_split_s: float = 1.0
) -> list[Segment]:
    """Assign speakers to aligned words by maximal temporal overlap with diarization
    turns, then group consecutive same-speaker words into segments (also splitting
    on inter-word gaps longer than gap_split_s)."""
    if not words:
        return []
    words = sorted(words, key=lambda word: word.start)
    if not turns:
        return segments_without_speakers(words, gap_split_s=gap_split_s)
    turns = sorted(turns, key=lambda turn: turn.start)
    labels = _label_words(words, turns)
    return _group_labelled(words, labels, gap_split_s)


def assign_speakers_to_segments(segments: list[Segment], turns: list[Turn]) -> list[Segment]:
    """Coarse fallback when only segment-level timestamps are available: each whole
    segment gets the speaker of the maximally overlapping diarization turn (a speaker
    change inside one whisper segment is lost at this granularity)."""
    if not segments or not turns:
        return segments
    ordered_turns = sorted(turns, key=lambda turn: turn.start)
    labelled: list[Segment] = []
    previous: str | None = None
    for segment in segments:
        best: str | None = None
        best_overlap = 0.0
        for turn in ordered_turns:
            overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best = turn.speaker
        if best is None:
            fallback = _nearest_turn(segment, ordered_turns).speaker
            best = previous if previous is not None else fallback
        labelled.append(segment.model_copy(update={"speaker": best}))
        previous = best
    return labelled


def resolve_segments(
    words: list[Word], plain_segments: list[Segment], turns: list[Turn] | None
) -> list[Segment]:
    """Best segment construction for the available inputs: word-level speaker merge
    when words exist, segment-level merge otherwise; without diarization (turns=None),
    provider segments verbatim or gap-grouped words."""
    if turns is not None:
        if words:
            return assign_speakers(words, turns)
        return assign_speakers_to_segments(plain_segments, turns)
    if plain_segments:
        return plain_segments
    return segments_without_speakers(words)


def segments_without_speakers(words: list[Word], *, gap_split_s: float = 1.0) -> list[Segment]:
    if not words:
        return []
    words = sorted(words, key=lambda word: word.start)
    return _group_labelled(words, [None] * len(words), gap_split_s)


def _decodable_audio(audio_path: Path) -> tuple[Path, bool]:
    """Path pyannote's soundfile backend can read, plus whether it is a temp file.

    libsndfile covers wav/flac/ogg/mp3 but not e.g. m4a/aac; the ingest contract is
    "any ffmpeg-decodable format", so fall back to an ffmpeg transcode next to the
    original (16 kHz mono wav — what the pipeline resamples to anyway)."""
    try:
        import soundfile

        soundfile.info(str(audio_path))
        return audio_path, False
    except Exception:
        pass
    converted = audio_path.with_name(audio_path.name + ".diarize.wav")
    logger.info("transcoding audio for diarization", extra={"event": "diarize.transcode"})
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(converted),
        ],
        capture_output=True,
    )
    if completed.returncode != 0:
        converted.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not decode the audio (exit {completed.returncode})")
    return converted, True


class Diarizer:
    """pyannote speaker diarization; models are HF-gated (accept the licenses for
    pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0, provide HF_TOKEN)."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._pipeline: Any = None

    def load(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                return
            import torch
            from pyannote.audio import Pipeline

            # torch >= 2.6 defaults torch.load to weights_only=True; the official
            # pyannote checkpoints pickle these classes, so allowlist them rather
            # than disabling the safety check wholesale
            from pyannote.audio.core.task import Problem, Resolution, Specifications
            from torch.torch_version import TorchVersion

            torch.serialization.add_safe_globals(
                [TorchVersion, Specifications, Problem, Resolution]
            )

            logger.info("loading diarization pipeline %s", self._settings.diarization_model)
            load_started = time.perf_counter()
            pipeline = Pipeline.from_pretrained(
                self._settings.diarization_model,
                use_auth_token=self._settings.hf_token,
                cache_dir=str(self._settings.model_cache_dir),
            )
            if pipeline is None:
                raise RuntimeError(
                    f"could not load {self._settings.diarization_model!r} — is the model"
                    " license accepted on Hugging Face and HF_TOKEN set?"
                )
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            self._pipeline = pipeline
            logger.info(
                "diarization pipeline loaded",
                extra={
                    "event": "diarize.loaded",
                    "duration_ms": round((time.perf_counter() - load_started) * 1000, 2),
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                },
            )

    def diarize(self, audio_path: Path) -> list[Turn]:
        self.load()
        logger.info("diarization started", extra={"event": "diarize.start"})
        started = time.perf_counter()
        decodable, is_temp = _decodable_audio(audio_path)
        try:
            # GPU-VERIFY(milestone-2): tune against real output; consider min_speakers /
            # max_speakers passthrough as API parameters in a later version.
            annotation = self._pipeline(str(decodable))
        finally:
            if is_temp:
                decodable.unlink(missing_ok=True)
        turns = [
            Turn(start=segment.start, end=segment.end, speaker=str(label))
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]
        logger.info(
            "diarization completed",
            extra={
                "event": "diarize.done",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "turns": len(turns),
                "speakers": len({turn.speaker for turn in turns}),
            },
        )
        return turns


def _group_labelled(
    words: list[Word], labels: list[str | None], gap_split_s: float
) -> list[Segment]:
    segments: list[Segment] = []
    start_index = 0
    for index in range(1, len(words) + 1):
        end_of_input = index == len(words)
        if not end_of_input:
            speaker_changed = labels[index] != labels[start_index]
            gap = words[index].start - words[index - 1].end
            if not speaker_changed and gap <= gap_split_s:
                continue
        group = words[start_index:index]
        segments.append(
            Segment(
                start=group[0].start,
                end=group[-1].end,
                speaker=labels[start_index],
                text=" ".join(word.word for word in group),
                words=group,
            )
        )
        start_index = index
    return segments
