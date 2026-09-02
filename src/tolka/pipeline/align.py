"""Local forced alignment of a transcript against its audio via easyaligner.

Shared by the hybrid engine (transcribe jobs: provider text, local word timestamps)
and every engine's task=diarize path (caller-supplied segments without words). One
module-level lock serializes aligner runs, and the loaded model stack (silero VAD +
wav2vec2 CTC + processor) is cached so concurrent jobs and multiple engines never
load wav2vec2 twice. Verified against easyaligner 0.x on CPU (devcontainer)."""

import importlib.util
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

from tolka.config import Settings
from tolka.jobs.models import Segment, Word
from tolka.pipeline.diarize import _decodable_audio, audio_duration

logger = logging.getLogger(__name__)

_align_lock = threading.Lock()
# (emissions model, device) -> (vad_model, ctc_model, processor); guarded by _align_lock
_loaded_stacks: dict[tuple[str, str], tuple[Any, Any, Any]] = {}

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


def _load_alignment_stack(
    settings: Settings, emissions_model: str, device: str
) -> tuple[Any, Any, Any]:
    """Silero VAD + wav2vec2 CTC model + processor, cached per (model, device)."""
    key = (emissions_model, device)
    if key not in _loaded_stacks:
        import torch
        from easyaligner.vad.silero import load_vad_model
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
        _loaded_stacks[key] = (load_vad_model(), model, processor)
    return _loaded_stacks[key]


def _resolve_emissions_model(settings: Settings, language: str) -> str:
    """CTC model for the job's language, warning on the acoustic-model mismatch
    an unmapped explicit language implies (auto/unknown have no better choice)."""
    emissions_model, matched = settings.emissions_model_for(language)
    if not matched and language.strip().lower() not in _UNSPECIFIC_LANGUAGES:
        logger.warning(
            "no emissions model configured for language %s; aligning with %s"
            " (add a TOLKA_EMISSIONS_MODELS entry for word-precise quality)",
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
    CTC model via TOLKA_EMISSIONS_MODELS (falling back to TOLKA_EMISSIONS_MODEL
    with a warning — an acoustic-model mismatch degrades word precision).
    Alignment errors propagate and fail the job: quality doctrine forbids
    silently degrading to provider timestamps or segment-level merging."""
    import torch
    from easyaligner.data.datamodel import SpeechSegment
    from easyaligner.pipelines import pipeline as align_pipeline

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
        with _align_lock:
            vad_model, ctc_model, processor = _load_alignment_stack(
                settings, emissions_model, device
            )
            Path(settings.work_dir).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="align-", dir=str(settings.work_dir)) as tmp:
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
    words: list[Word] = []
    for speech in speeches:
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
    return words
