"""Speaker diarization and word-to-speaker assignment.

The assignment logic is pure and CI-tested; the pyannote-backed Diarizer lives here
too but is only imported/constructed when the ML extra is installed.
"""

import logging
import threading
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


def _nearest_turn(word: Word, turns: list[Turn]) -> Turn:
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


def segments_without_speakers(words: list[Word], *, gap_split_s: float = 1.0) -> list[Segment]:
    if not words:
        return []
    words = sorted(words, key=lambda word: word.start)
    return _group_labelled(words, [None] * len(words), gap_split_s)


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

            logger.info("loading diarization pipeline %s", self._settings.diarization_model)
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

    def diarize(self, audio_path: Path) -> list[Turn]:
        self.load()
        # GPU-VERIFY(milestone-2): tune against real output; consider min_speakers /
        # max_speakers passthrough as API parameters in a later version.
        annotation = self._pipeline(str(audio_path))
        return [
            Turn(start=segment.start, end=segment.end, speaker=str(label))
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]


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
