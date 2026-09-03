"""Local forced alignment of a transcript against its audio via easyaligner.

Shared by the hybrid engine (transcribe jobs: provider text, local word timestamps)
and every engine's task=diarize path (caller-supplied segments without words). The
wav2vec2 CTC model + processor are loaded once per (model, device) and shared
read-only between concurrent jobs and engines; the silero VAD model keeps LSTM
state between calls, so every pipeline thread gets its own instance. Aligner runs
are bounded by the process-wide GPU slot (vemsa.pipeline.gpu), not by a module
lock. Verified against easyaligner 0.x on CPU (devcontainer)."""

import importlib.util
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

from vemsa.config import Settings
from vemsa.jobs.models import Segment, Word
from vemsa.pipeline.diarize import _decodable_audio, audio_duration
from vemsa.pipeline.gpu import gpu_slot

logger = logging.getLogger(__name__)

_stack_lock = threading.Lock()
# (emissions model, device) -> (ctc_model, processor); mutation guarded by _stack_lock
_loaded_stacks: dict[tuple[str, str], tuple[Any, Any]] = {}
# one silero VAD instance per pipeline thread (executor threads are reused, so the
# small JIT load is amortised); get_speech_timestamps resets its state on entry
_thread_vad = threading.local()

# request languages that carry no usable signal for picking an acoustic model
_UNSPECIFIC_LANGUAGES = {"", "auto", "unknown"}


class SegmentAligner(Protocol):
    def __call__(self, audio_path: Path, segments: list[Segment], language: str) -> list[Word]: ...


def alignment_available() -> bool:
    return importlib.util.find_spec("easyaligner") is not None


def build_segment_aligner(settings: Settings) -> SegmentAligner:
    """Aligner for task=diarize jobs. Forced alignment is mandatory — the service
    always runs with a GPU and the `align` extra — so there is no disabled state:
    without easyaligner installed, every job needing alignment fails loudly at
    run time (the error below flags the misconfiguration at engine build)."""
    if not alignment_available():
        logger.error(
            "easyaligner is not installed but forced alignment is mandatory;"
            " every job needing alignment will fail until the 'align' extra is"
            " installed"
        )

    def aligner(audio_path: Path, segments: list[Segment], language: str) -> list[Word]:
        return force_align_segments(settings, audio_path, segments, language)

    return aligner


def alignment_transcript(
    segments: list[Segment], *, fallback_text: str = "", fallback_end: float = 0.0
) -> list[dict[str, float | str]]:
    """Alignment windows for the segments: their start/end/text verbatim.

    A whole-file segment carries the whole audio as its window, which aligns the
    entire file — the intended behaviour when a consumer rejected the provider's
    timeline and sent one segment per chunk. Without any usable segment the text
    goes into a single window spanning the audio (fallback_end = its duration)."""
    windowed = [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in segments
        if segment.text.strip()
    ]
    if windowed:
        return windowed
    return [{"start": 0.0, "end": fallback_end, "text": fallback_text.strip()}]


def _load_alignment_stack(settings: Settings, emissions_model: str, device: str) -> tuple[Any, Any]:
    """wav2vec2 CTC model + processor, cached per (model, device).

    Shared read-only: easyaligner only runs the model under inference_mode."""
    key = (emissions_model, device)
    stack = _loaded_stacks.get(key)
    if stack is not None:
        return stack
    with _stack_lock:
        stack = _loaded_stacks.get(key)
        if stack is not None:
            return stack
        import torch
        from transformers import AutoModelForCTC, Wav2Vec2Processor

        logger.info("loading alignment stack %s on %s", emissions_model, device)
        model = (
            AutoModelForCTC.from_pretrained(
                emissions_model, cache_dir=str(settings.model_cache_dir)
            )
            .to(device)
            .to(torch.float16 if device == "cuda" else torch.float32)
        )
        processor = Wav2Vec2Processor.from_pretrained(
            emissions_model, cache_dir=str(settings.model_cache_dir)
        )
        stack = (model, processor)
        _loaded_stacks[key] = stack
        return stack


def _new_vad_model() -> Any:
    from easyaligner.vad.silero import load_vad_model

    return load_vad_model()


def _vad_model() -> Any:
    """This thread's silero VAD instance (stateful, so never shared across threads)."""
    model = getattr(_thread_vad, "model", None)
    if model is None:
        model = _new_vad_model()
        _thread_vad.model = model
    return model


def _easyaligner() -> tuple[Any, Any]:
    """(SpeechSegment, pipeline) from easyaligner, imported lazily so the module
    loads without the `align` extra."""
    from easyaligner.data.datamodel import SpeechSegment
    from easyaligner.pipelines import pipeline

    return SpeechSegment, pipeline


def _resolve_emissions_model(settings: Settings, language: str) -> str:
    """CTC model for the job's language, warning on the acoustic-model mismatch
    an unmapped explicit language implies (auto/unknown have no better choice)."""
    emissions_model, matched = settings.emissions_model_for(language)
    if not matched and language.strip().lower() not in _UNSPECIFIC_LANGUAGES:
        logger.warning(
            "no emissions model configured for language %s; aligning with %s"
            " (add a VEMSA_EMISSIONS_MODELS entry for word-precise quality)",
            language,
            emissions_model,
            extra={
                "event": "align.language_fallback",
                "language": language,
                "emissions_model": emissions_model,
            },
        )
    return emissions_model


def force_align_segments(
    settings: Settings,
    audio_path: Path,
    segments: list[Segment],
    language: str,
    *,
    fallback_text: str = "",
) -> list[Word]:
    """Align the transcript text against the audio, returning word timestamps.

    easyaligner's pipeline steps hand data to each other through JSON/npy files,
    so each run gets a throwaway directory under work_dir. `language` picks the
    CTC model via VEMSA_EMISSIONS_MODELS (falling back to VEMSA_EMISSIONS_MODEL
    with a warning — an acoustic-model mismatch degrades word precision).
    Alignment errors propagate and fail the job: quality doctrine forbids
    silently degrading to provider timestamps or segment-level merging."""
    import torch

    SpeechSegment, align_pipeline = _easyaligner()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emissions_model = _resolve_emissions_model(settings, language)

    # easyaligner is not guaranteed to decode everything ffmpeg can (the ingest
    # contract); transcode like the diarizer does when libsndfile cannot read it.
    decodable, is_temp = _decodable_audio(audio_path)
    try:
        transcript = alignment_transcript(
            segments,
            fallback_text=fallback_text,
            fallback_end=audio_duration(decodable, fallback=0.0),
        )
        speeches = [
            [
                SpeechSegment(
                    speech_id=index,
                    start=float(entry["start"]),
                    end=float(entry["end"]),
                    text=str(entry["text"]),
                )
                for index, entry in enumerate(transcript)
            ]
        ]
        ctc_model, processor = _load_alignment_stack(settings, emissions_model, device)
        vad_model = _vad_model()
        Path(settings.work_dir).mkdir(parents=True, exist_ok=True)
        # the whole run (VAD -> emissions -> Viterbi) holds one GPU slot; at
        # gpu_concurrency=1 aligner runs stay serialized exactly as before
        with (
            gpu_slot(),
            tempfile.TemporaryDirectory(prefix="align-", dir=str(settings.work_dir)) as tmp,
        ):
            aligned = align_pipeline(
                vad_model=vad_model,
                emissions_model=ctc_model,
                processor=processor,
                audio_paths=[decodable.name],
                audio_dir=str(decodable.parent),
                speeches=speeches,
                alignment_strategy="speech",
                blank_id=processor.tokenizer.pad_token_id,
                word_boundary=processor.tokenizer.word_delimiter_token,
                return_alignments=True,
                delete_emissions=True,
                output_vad_dir=f"{tmp}/vad",
                output_emissions_dir=f"{tmp}/emissions",
                output_alignments_dir=f"{tmp}/alignments",
                device=device,
            )
    finally:
        if is_temp:
            decodable.unlink(missing_ok=True)

    words = words_from_alignments(aligned[0])
    words.sort(key=lambda word: word.start)
    return words


def words_from_alignments(speeches: list[Any]) -> list[Word]:
    """Flatten easyaligner output (SpeechSegment -> AlignmentSegment -> WordSegment)
    into Words, tolerating flatter shapes from older/newer versions.

    Shared with the local engine: easytranscriber's pipeline returns the same
    easyaligner shapes."""
    return [word for speech in words_by_speech(speeches) for word in speech]


def words_by_speech(speeches: list[Any]) -> list[list[Word]]:
    """easyaligner output as one word list per input speech window, in order.

    Also the place where easyaligner's silent degradation is made visible: when a
    window's text cannot be aligned (too long for its audio, or characters the
    CTC vocabulary lacks) the library spreads the words evenly over the window
    with score 0.0 instead of failing. Those words are counted and logged here
    (`align.interpolated`); the job-level floor lives in the worker."""
    grouped: list[list[Word]] = []
    for speech in speeches:
        words: list[Word] = []
        containers = getattr(speech, "alignments", None) or [speech]
        for container in containers:
            nested = getattr(container, "words", None) or [container]
            for item in nested:
                text = getattr(item, "word", None) or getattr(item, "text", None)
                start = getattr(item, "start", None)
                end = getattr(item, "end", None)
                score = getattr(item, "score", None)
                if text is None or start is None or end is None:
                    raise ValueError(f"unrecognized alignment shape: {item!r}")
                words.append(
                    Word(
                        word=str(text).strip(),
                        start=float(start),
                        end=float(end),
                        probability=float(score) if score is not None else None,
                    )
                )
        interpolated = interpolated_words(words)
        if interpolated:
            logger.warning(
                "forced alignment interpolated %d of %d words in window %s-%s:"
                " the text could not be aligned against the audio",
                interpolated,
                len(words),
                getattr(speech, "start", None),
                getattr(speech, "end", None),
                extra={
                    "event": "align.interpolated",
                    "interpolated_words": interpolated,
                    "window_words": len(words),
                    "window_start": getattr(speech, "start", None),
                    "window_end": getattr(speech, "end", None),
                },
            )
        grouped.append(words)
    return grouped


def interpolated_words(words: list[Word]) -> int:
    """How many words carry easyaligner's fallback score (exactly 0.0): their
    timestamps were linearly interpolated, not derived from the audio."""
    return sum(1 for word in words if word.probability == 0.0)
