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


@dataclass(frozen=True)
class AttributionTuning:
    """Knobs for word→speaker attribution and segment shaping.

    Defaults are the tested behaviour; production overrides come from the
    TOLKA_ATTR_* settings (Settings.attribution_tuning) so tuning against real
    audio does not need a code change."""

    # a winning turn overlap below this fraction of the word's duration counts
    # as no overlap (forced-alignment jitter clips boundary words)
    min_coverage: float = 0.25
    # islands at most this many words AND spanning at most island_max_span_s
    # (or crammed into island_max_duration_s regardless of count) are absorbed
    island_max_words: int = 2
    island_max_duration_s: float = 1.0
    island_max_span_s: float = 3.0
    # a word without turn overlap inherits the previous word's speaker only
    # across a gap this short; beyond it the nearest turn wins
    inherit_max_gap_s: float = 2.0
    # a pause longer than this ends a segment when it coincides with a sentence end
    gap_split_s: float = 1.0
    # a silence longer than this ends a segment even mid-sentence
    hard_gap_split_s: float = HARD_GAP_SPLIT_S
    # a segment at most this many words and this short whose speaker appears on
    # neither side is merged into the neighbour it reads mid-sentence with
    min_segment_words: int = 1
    min_segment_duration_s: float = 0.6


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip(_TRAILING_CLOSERS)
    return bool(stripped) and stripped[-1] in _SENTENCE_END_CHARS


def _continues_sentence(text: str) -> bool:
    """Whether a word reads as a mid-sentence continuation: its first letter is
    lowercase. Words without letters (numbers, bare punctuation) give no signal
    and count as not continuing."""
    for char in text:
        if char.isalpha():
            return char.islower()
    return False


def _overlap(word: Word, turn: Turn) -> float:
    return max(0.0, min(word.end, turn.end) - max(word.start, turn.start))


def _nearest_turn(word: Word | Segment, turns: list[Turn]) -> Turn:
    midpoint = (word.start + word.end) / 2
    return min(turns, key=lambda turn: abs((turn.start + turn.end) / 2 - midpoint))


def _label_words(
    words: list[Word],
    turns: list[Turn],
    *,
    min_coverage: float,
    inherit_max_gap_s: float,
) -> list[str]:
    """Per word, the speaker of the maximally overlapping turn; a winning overlap
    below min_coverage of the word's duration counts as no overlap — forced-alignment
    jitter can land a sliver of a boundary word inside the neighbouring turn. Words
    without a (sufficient) overlap inherit the previous word's speaker across a gap
    of at most inherit_max_gap_s; beyond that (and for the first word) the nearest
    turn by midpoint wins — a speaker should not stretch across a long silence just
    because they spoke last."""
    labels: list[str] = []
    first_candidate = 0
    for position, word in enumerate(words):
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
        if best_overlap < min_coverage * (word.end - word.start):
            best = None
        if best is None:
            gap = word.start - words[position - 1].end if position > 0 else None
            if labels and gap is not None and gap <= inherit_max_gap_s:
                best = labels[-1]
            else:
                best = _nearest_turn(word, turns).speaker
        labels.append(best)
    return labels


def _smooth_islands(
    words: list[Word],
    labels: list[str],
    *,
    max_words: int,
    max_duration_s: float,
    max_span_s: float,
) -> list[str]:
    """Relabel a short run of words whose neighbours on both sides agree on a
    different speaker.

    An untranscribed backchannel ("mm") flips the exclusive diarization track to
    the other speaker for a moment, and alignment jitter can drop a mid-sentence
    word into that window — a human transcriber would never credit one word
    mid-sentence to another speaker. A run counts as short with at most max_words
    words inside max_span_s, or any word count inside max_duration_s; a few words
    stretched over a long span are a real turn. A run beginning right after
    sentence-final punctuation is kept — a short interjection ("Ja.") legitimately
    starts there — unless the run reads as glued into the following words
    mid-sentence (it carries no sentence-final punctuation of its own and the next
    word continues lowercase), where a human would keep the sentence on one line."""
    smoothed = list(labels)
    run_start = 0
    for index in range(1, len(smoothed) + 1):
        if index < len(smoothed) and smoothed[index] == smoothed[run_start]:
            continue
        if run_start > 0 and index < len(smoothed):
            neighbour = smoothed[run_start - 1]
            count = index - run_start
            duration = words[index - 1].end - words[run_start].start
            short_run = (
                count <= max_words and duration <= max_span_s
            ) or duration <= max_duration_s
            glued_to_following = not _ends_sentence(words[index - 1].word) and _continues_sentence(
                words[index].word
            )
            if (
                neighbour == smoothed[index]
                and neighbour != smoothed[run_start]
                and short_run
                and (not _ends_sentence(words[run_start - 1].word) or glued_to_following)
            ):
                smoothed[run_start:index] = [neighbour] * count
        run_start = index
    return smoothed


def assign_speakers(
    words: list[Word], turns: list[Turn], *, tuning: AttributionTuning | None = None
) -> list[Segment]:
    """Assign speakers to aligned words by maximal temporal overlap with diarization
    turns, smooth away single-word islands surrounded by another speaker, group
    consecutive same-speaker words into segments (splitting where a pause longer
    than tuning.gap_split_s coincides with a sentence end — see _group_labelled),
    then merge away tiny orphan segments glued mid-sentence to a neighbour."""
    if not words:
        return []
    tuning = tuning or AttributionTuning()
    words = sorted(words, key=lambda word: word.start)
    if not turns:
        return segments_without_speakers(words, gap_split_s=tuning.gap_split_s)
    turns = sorted(turns, key=lambda turn: turn.start)
    labels = _label_words(
        words,
        turns,
        min_coverage=tuning.min_coverage,
        inherit_max_gap_s=tuning.inherit_max_gap_s,
    )
    labels = _smooth_islands(
        words,
        labels,
        max_words=tuning.island_max_words,
        max_duration_s=tuning.island_max_duration_s,
        max_span_s=tuning.island_max_span_s,
    )
    segments = _group_labelled(
        words, labels, tuning.gap_split_s, hard_gap_split_s=tuning.hard_gap_split_s
    )
    return _merge_short_orphans(
        segments,
        max_words=tuning.min_segment_words,
        max_duration_s=tuning.min_segment_duration_s,
    )


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
    words: list[Word],
    plain_segments: list[Segment],
    turns: list[Turn] | None,
    *,
    tuning: AttributionTuning | None = None,
) -> list[Segment]:
    """Best segment construction for the available inputs: word-level speaker merge
    when words exist, segment-level merge otherwise; without diarization (turns=None),
    provider segments verbatim or gap-grouped words."""
    if turns is not None:
        if words:
            return assign_speakers(words, turns, tuning=tuning)
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


# Containers safe to hand to pyannote's decoder without transcoding. Compressed
# formats (mp3/ogg/...) are transcoded even though libsndfile could read them:
# pyannote 4 decodes via torchcodec, whose decoded frame count for e.g. VBR mp3
# can disagree with the container's duration estimate and fail the pipeline with
# a ValueError — while the same audio works once transcoded to wav.
_PCM_FORMATS = {"WAV", "WAVEX", "AIFF", "FLAC"}


def _decodable_audio(audio_path: Path) -> tuple[Path, bool]:
    """Path pyannote's decoder handles reliably, plus whether it is a temp file.

    PCM containers pass through; everything else — compressed formats libsndfile
    could read, and formats it cannot (m4a/aac) — gets one canonical ffmpeg
    transcode next to the original (16 kHz mono wav, what the pipeline resamples
    to anyway), which also keeps the aligner and the diarizer on the identical
    decoded timeline."""
    try:
        import soundfile

        if soundfile.info(str(audio_path)).format in _PCM_FORMATS:
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
    """pyannote speaker diarization; models are HF-gated (accept the license for
    pyannote/speaker-diarization-community-1, provide HF_TOKEN)."""

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

            # torch >= 2.6 defaults torch.load to weights_only=True; older pyannote
            # checkpoints (e.g. speaker-diarization-3.1, still configurable) pickle
            # these classes, so allowlist them rather than disabling the safety
            # check wholesale
            from pyannote.audio.core.task import Problem, Resolution, Specifications
            from torch.torch_version import TorchVersion

            torch.serialization.add_safe_globals(
                [TorchVersion, Specifications, Problem, Resolution]
            )

            logger.info("loading diarization pipeline %s", self._settings.diarization_model)
            load_started = time.perf_counter()
            pipeline = Pipeline.from_pretrained(
                self._settings.diarization_model,
                token=self._settings.hf_token,
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
            output = self._pipeline(str(decodable), **bounds)
        finally:
            if is_temp:
                decodable.unlink(missing_ok=True)
        annotation, exclusive = self._pick_annotation(output)
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
                "exclusive": exclusive,
            },
        )
        return turns

    def _pick_annotation(self, output: Any) -> tuple[Any, bool]:
        """The Annotation to derive turns from, and whether it is the exclusive one.

        pyannote 4.x pipelines return a DiarizeOutput whose exclusive variant keeps
        exactly one speaker active at a time — the one an ASR system would have
        transcribed — which is what word attribution wants during crosstalk. A
        custom pipeline may still return a bare Annotation; use it as-is."""
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        if self._settings.diarize_exclusive and exclusive is not None:
            return exclusive, True
        return getattr(output, "speaker_diarization", output), False


def _group_labelled(
    words: list[Word],
    labels: list[str | None],
    gap_split_s: float,
    *,
    hard_gap_split_s: float = HARD_GAP_SPLIT_S,
) -> list[Segment]:
    """Group consecutive same-speaker words into segments the way a human lines
    a transcript: a segment ends at a speaker change, at a pause longer than
    gap_split_s that coincides with sentence-final punctuation (a pause
    mid-sentence keeps the sentence together), or at a silence longer than
    hard_gap_split_s regardless of punctuation."""
    segments: list[Segment] = []
    start_index = 0
    for index in range(1, len(words) + 1):
        end_of_input = index == len(words)
        if not end_of_input:
            speaker_changed = labels[index] != labels[start_index]
            gap = words[index].start - words[index - 1].end
            sentence_break = gap > gap_split_s and _ends_sentence(words[index - 1].word)
            if not speaker_changed and not sentence_break and gap <= hard_gap_split_s:
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


def _merge_short_orphans(
    segments: list[Segment], *, max_words: int, max_duration_s: float
) -> list[Segment]:
    """Merge a tiny orphan segment into the neighbour it reads mid-sentence with.

    An orphan's speaker appears on neither side, so island smoothing never saw
    agreeing neighbours (three-speaker A/B/C jitter, or the transcript's edge).
    A one-word line credited to a third voice mid-sentence is attribution noise,
    not a turn: when the orphan carries no sentence-final punctuation and the
    following segment continues lowercase it joins that segment; when instead it
    finishes the previous segment's unfinished sentence (continues lowercase and
    carries the sentence-final punctuation) it joins that one. An orphan without
    such grammatical glue — a bounded interjection ("Ja."), or a bare word with
    no grammar signal at all — is left alone."""
    if len(segments) < 2:
        return segments
    merged: list[Segment] = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        previous = merged[-1] if merged else None
        following = segments[index + 1] if index + 1 < len(segments) else None
        if _orphan(segment, previous, following, max_words, max_duration_s):
            if following is not None and (
                not _ends_sentence(segment.words[-1].word)
                and _continues_sentence(following.words[0].word)
            ):
                segments[index + 1] = _join_segments(segment, following)
                index += 1
                continue
            if previous is not None and (
                not _ends_sentence(previous.words[-1].word)
                and _continues_sentence(segment.words[0].word)
                and _ends_sentence(segment.words[-1].word)
            ):
                merged[-1] = _join_segments(previous, segment, speaker=previous.speaker)
                index += 1
                continue
        merged.append(segment)
        index += 1
    return merged


def _orphan(
    segment: Segment,
    previous: Segment | None,
    following: Segment | None,
    max_words: int,
    max_duration_s: float,
) -> bool:
    if not segment.words or len(segment.words) > max_words:
        return False
    if segment.end - segment.start > max_duration_s:
        return False
    neighbours = [other for other in (previous, following) if other is not None]
    return bool(neighbours) and all(
        other.speaker != segment.speaker and other.words for other in neighbours
    )


def _join_segments(first: Segment, second: Segment, *, speaker: str | None = None) -> Segment:
    return Segment(
        start=first.start,
        end=second.end,
        speaker=speaker if speaker is not None else second.speaker,
        text=f"{first.text} {second.text}",
        words=[*first.words, *second.words],
    )
