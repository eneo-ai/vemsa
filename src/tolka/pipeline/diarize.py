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

from tolka.jobs.models import Segment, SpeakerBounds, Word

if TYPE_CHECKING:
    from tolka.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


# Sentence-final punctuation for word grouping; closing quotes/brackets after it
# are ignored. A colon deliberately does not end a sentence: a heading read
# aloud ("Styrkor och framgångar:") belongs with what it introduces.
_SENTENCE_END_CHARS = ".?!…"
_TRAILING_CLOSERS = "\"'”’»)]"

# A silence this long ends a segment even mid-sentence, so a transcript without
# punctuation cannot collapse into one endless segment.
HARD_GAP_SPLIT_S = 15.0


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip(_TRAILING_CLOSERS)
    return bool(stripped) and stripped[-1] in _SENTENCE_END_CHARS


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
    turns, then group consecutive same-speaker words into segments (splitting where
    a pause longer than gap_split_s coincides with a sentence end — see
    _group_labelled)."""
    if not words:
        return []
    words = sorted(words, key=lambda word: word.start)
    if not turns:
        return segments_without_speakers(words, gap_split_s=gap_split_s)
    turns = sorted(turns, key=lambda turn: turn.start)
    labels = _label_words(words, turns)
    return _group_labelled(words, labels, gap_split_s)


def assign_speakers_to_segments(
    segments: list[Segment],
    turns: list[Turn],
    *,
    split_threshold: float = 0.25,
    min_split_duration_s: float = 2.0,
    min_split_words: int = 4,
    punctuation_snap_chars: int = 12,
) -> list[Segment]:
    """Coarse fallback when only segment-level timestamps are available: each whole
    segment gets the speaker with the maximal total turn overlap. A segment whose
    minority speakers cover at least split_threshold of its overlapped time is
    instead split at the turn boundaries, slicing the text by time proportion
    (snapped to a word boundary, preferring sentence punctuation). Speech rate is
    not uniform, so a cut can land a word or two off — still far better than losing
    the speaker change entirely. Word-level input never reaches this path."""
    if not segments or not turns:
        return segments
    ordered_turns = sorted(turns, key=lambda turn: turn.start)
    labelled: list[Segment] = []
    previous: str | None = None
    split_count = 0
    for segment in segments:
        shares = _speaker_shares(segment, ordered_turns)
        if not shares:
            fallback = _nearest_turn(segment, ordered_turns).speaker
            best = previous if previous is not None else fallback
            labelled.append(segment.model_copy(update={"speaker": best}))
            previous = best
            continue
        majority = max(shares, key=lambda speaker: shares[speaker])
        splittable = (
            shares[majority] < (1 - split_threshold) * sum(shares.values())
            and segment.end - segment.start >= min_split_duration_s
            and len(segment.text.split()) >= min_split_words
        )
        pieces = (
            _split_segment(segment, ordered_turns, shares, punctuation_snap_chars)
            if splittable
            else []
        )
        if len(pieces) > 1:
            split_count += 1
            labelled.extend(pieces)
            previous = pieces[-1].speaker
        else:
            labelled.append(segment.model_copy(update={"speaker": majority}))
            previous = majority
    if split_count:
        logger.info(
            "split %d segment(s) at speaker turns by time proportion",
            split_count,
            extra={"event": "diarize.segment_split", "segments": split_count},
        )
    return labelled


def _speaker_shares(segment: Segment, turns: list[Turn]) -> dict[str, float]:
    """Total overlap duration per speaker across all turns touching the segment."""
    shares: dict[str, float] = {}
    for turn in turns:
        overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
        if overlap > 0.0:
            shares[turn.speaker] = shares.get(turn.speaker, 0.0) + overlap
    return shares


def _speaker_spans(
    segment: Segment, turns: list[Turn], shares: dict[str, float]
) -> list[tuple[float, float, str]]:
    """The segment partitioned into one span per contiguous speaker stretch.

    Overlapping turns are flattened to the speaker with the larger overall share;
    silence between turns joins the preceding span (the leading gap joins the
    following one), so span boundaries sit exactly at speaker changes."""
    bounds = {segment.start, segment.end}
    for turn in turns:
        if turn.end > segment.start and turn.start < segment.end:
            bounds.add(max(turn.start, segment.start))
            bounds.add(min(turn.end, segment.end))
    ordered = sorted(bounds)
    spans: list[list] = []
    for start, end in zip(ordered, ordered[1:], strict=False):
        midpoint = (start + end) / 2
        covering = [turn.speaker for turn in turns if turn.start < midpoint < turn.end]
        speaker = max(covering, key=lambda name: shares[name]) if covering else None
        spans.append([start, end, speaker])
    for index, span in enumerate(spans):
        if span[2] is None:
            following = next((other[2] for other in spans[index:] if other[2]), None)
            span[2] = spans[index - 1][2] if index > 0 else following
    merged: list[list] = []
    for start, end, speaker in spans:
        if merged and merged[-1][2] == speaker:
            merged[-1][1] = end
        else:
            merged.append([start, end, speaker])
    return [(start, end, speaker) for start, end, speaker in merged if speaker is not None]


def _snap_offset(text: str, offset: int, punctuation_window: int) -> int | None:
    """Nearest word boundary to a character offset: a cut just after sentence
    punctuation within the window wins, else the nearest whitespace; None when the
    text has no boundary at all."""
    punctuation = [
        index + 1
        for index, char in enumerate(text[:-1])
        if char in ".?!" and text[index + 1].isspace()
    ]
    nearest_punctuation = min(punctuation, key=lambda pos: abs(pos - offset), default=None)
    if nearest_punctuation is not None and abs(nearest_punctuation - offset) <= punctuation_window:
        return nearest_punctuation
    spaces = [index for index, char in enumerate(text) if char.isspace()]
    return min(spaces, key=lambda pos: abs(pos - offset), default=None)


def _split_segment(
    segment: Segment, turns: list[Turn], shares: dict[str, float], punctuation_window: int
) -> list[Segment]:
    """One piece per within-segment speaker stretch, text sliced by time proportion.

    Returns [] when no valid split remains (a single stretch, or every cut snapped
    onto the text edges), leaving the caller on the whole-segment path."""
    spans = _speaker_spans(segment, turns, shares)
    if len(spans) < 2:
        return []
    text = segment.text
    duration = segment.end - segment.start
    boundaries: list[tuple[float, int]] = [(segment.start, 0)]
    for span_start, _span_end, _speaker in spans[1:]:
        raw = round(len(text) * (span_start - segment.start) / duration)
        snapped = _snap_offset(text, raw, punctuation_window)
        if snapped is not None and boundaries[-1][1] < snapped < len(text):
            boundaries.append((span_start, snapped))
    boundaries.append((segment.end, len(text)))

    pieces: list[Segment] = []
    for (start, offset_start), (end, offset_end) in zip(boundaries, boundaries[1:], strict=False):
        piece_text = text[offset_start:offset_end].strip()
        if not piece_text:
            continue
        speaker = _dominant_speaker(spans, start, end)
        if pieces and pieces[-1].speaker == speaker:
            pieces[-1] = pieces[-1].model_copy(
                update={"end": end, "text": f"{pieces[-1].text} {piece_text}"}
            )
        else:
            pieces.append(Segment(start=start, end=end, speaker=speaker, text=piece_text))
    return pieces if len(pieces) > 1 else []


def _dominant_speaker(spans: list[tuple[float, float, str]], start: float, end: float) -> str:
    totals: dict[str, float] = {}
    for span_start, span_end, speaker in spans:
        overlap = max(0.0, min(end, span_end) - max(start, span_start))
        totals[speaker] = totals.get(speaker, 0.0) + overlap
    return max(totals, key=lambda name: totals[name])


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


def audio_duration(audio_path: Path, *, fallback: float) -> float:
    """Decoded length of the audio, or ``fallback`` when it cannot be read.

    soundfile first (cheap, no subprocess), then ffprobe for the formats libsndfile
    cannot open (e.g. m4a) — the duration feeds consumers' usage accounting."""
    try:
        import soundfile

        info = soundfile.info(str(audio_path))
        return float(info.frames) / info.samplerate
    except Exception:
        pass
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(audio_path),
            ],
            capture_output=True,
            timeout=60,
        )
        return float(completed.stdout.strip())
    except Exception:
        return fallback


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

    def diarize(self, audio_path: Path, *, speakers: SpeakerBounds | None = None) -> list[Turn]:
        self.load()
        bounds = speakers.pipeline_kwargs() if speakers else {}
        logger.info("diarization started", extra={"event": "diarize.start", **bounds})
        started = time.perf_counter()
        decodable, is_temp = _decodable_audio(audio_path)
        try:
            # GPU-VERIFY(milestone-2): tune against real output. pyannote over-splits
            # without a prior; callers can bound it via num/min/max_speakers.
            annotation = self._pipeline(str(decodable), **bounds)
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
    """Group consecutive same-speaker words into segments the way a human lines
    a transcript: a segment ends at a speaker change, at a pause longer than
    gap_split_s that coincides with sentence-final punctuation (a pause
    mid-sentence keeps the sentence together), or at a silence longer than
    HARD_GAP_SPLIT_S regardless of punctuation."""
    segments: list[Segment] = []
    start_index = 0
    for index in range(1, len(words) + 1):
        end_of_input = index == len(words)
        if not end_of_input:
            speaker_changed = labels[index] != labels[start_index]
            gap = words[index].start - words[index - 1].end
            sentence_break = gap > gap_split_s and _ends_sentence(words[index - 1].word)
            if not speaker_changed and not sentence_break and gap <= HARD_GAP_SPLIT_S:
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
