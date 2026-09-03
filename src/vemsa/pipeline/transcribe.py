"""Fully local engine (VEMSA_ENGINE=local): easytranscriber whisper + forced alignment
+ pyannote diarization, all in-process. For deployments with their own GPU and no
external whisper endpoint.

Everything ML-flavoured is imported lazily inside methods: this module is importable
without the `local` extra, but transcribe() requires it. Spots that could not be
verified without a GPU box are marked GPU-VERIFY(milestone-2).
"""

import logging
import tempfile
import threading
from pathlib import Path

from vemsa.config import Settings
from vemsa.jobs.models import JobStage, Segment, SpeakerBounds, TranscriptionResult, Word
from vemsa.pipeline.align import build_segment_aligner, words_from_alignments
from vemsa.pipeline.base import StageReporter, report_stage
from vemsa.pipeline.diarize import (
    Diarizer,
    assign_speakers,
    audio_duration,
    segments_without_speakers,
)
from vemsa.pipeline.gpu import gpu_slot
from vemsa.pipeline.label import label_speakers
from vemsa.pipeline.realign import align_transcript
from vemsa.pipeline.render import render_text

logger = logging.getLogger(__name__)


class EasyTranscriberEngine:
    """Blocking easytranscriber pipeline + pyannote diarization; runs in a worker thread."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._diarizer = Diarizer(settings)
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
        if vocabulary:
            logger.warning(
                "easytranscriber has no vocabulary/prompt support; ignoring %d terms",
                len(vocabulary),
            )

        import torch
        from easytranscriber.pipelines import pipeline

        # GPU-VERIFY(milestone-2): confirm easytranscriber accepts language=None for
        # whisper auto-detect; otherwise require an explicit language in the API.
        lang = None if language == "auto" else language
        device = "cuda" if torch.cuda.is_available() else "cpu"

        report_stage(on_stage, JobStage.TRANSCRIBING)
        with self._lock:
            # GPU-VERIFY(milestone-2): model instances are cached on disk via cache_dir,
            # but confirm in-memory reuse across pipeline() calls; if models reload per
            # call, hold them here instead.
            #
            # GPU-VERIFY(milestone-2): tokenizer for forced alignment — easyaligner's
            # load_tokenizer() Swedish support is unverified; the emissions model default
            # is KBLab/wav2vec2-large-voxrex-swedish (see Settings.emissions_model).
            Path(self._settings.work_dir).mkdir(parents=True, exist_ok=True)
            # engine lock outer, GPU slot inner (the only place they nest)
            with (
                gpu_slot(),
                tempfile.TemporaryDirectory(
                    prefix="transcribe-", dir=str(self._settings.work_dir)
                ) as tmp,
            ):
                aligned = pipeline(
                    # silero to match the alignment stack in align.py; pyannote VAD
                    # would additionally need the HF token
                    vad_model="silero",
                    emissions_model=self._settings.emissions_model,
                    transcription_model=model,
                    # names relative to audio_dir: the pipeline steps locate their
                    # intermediate files by these names
                    audio_paths=[audio_path.name],
                    audio_dir=str(audio_path.parent),
                    language=lang,
                    cache_dir=str(self._settings.model_cache_dir),
                    # the VAD -> transcribe -> emissions -> align steps hand data to
                    # each other through JSON files, so saving must stay on; the
                    # throwaway directory discards them
                    save_json=True,
                    return_alignments=True,
                    delete_emissions=True,
                    output_vad_dir=f"{tmp}/vad",
                    output_transcriptions_dir=f"{tmp}/transcriptions",
                    output_emissions_dir=f"{tmp}/emissions",
                    output_alignments_dir=f"{tmp}/alignments",
                    device=device,
                )

        words = words_from_alignments(aligned[0])
        duration = audio_duration(audio_path, fallback=words[-1].end if words else 0.0)

        if diarize:
            report_stage(on_stage, JobStage.DIARIZING)
            turns = self._diarizer.diarize(audio_path, speakers=speakers)
            segments = assign_speakers(words, turns, tuning=self._settings.attribution_tuning())
        else:
            segments = segments_without_speakers(words)

        return TranscriptionResult(
            # GPU-VERIFY(milestone-2): surface whisper's detected language when
            # auto-detect was used, instead of echoing the request.
            language=lang or "auto",
            duration_seconds=duration,
            model=model,
            text=render_text(segments),
            segments=segments,
            # easytranscriber's word timestamps come from its own forced alignment
            alignment="forced",
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
            tuning=self._settings.attribution_tuning(),
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
        report_stage(on_stage, JobStage.ALIGNING)
        return align_transcript(
            self._segment_aligner,
            audio_path,
            segments=segments,
            language=language,
            model=model,
            tuning=self._settings.attribution_tuning(),
        )

    def warm_up(self) -> None:
        logger.info("warming up diarization pipeline")
        self._diarizer.load()
        # GPU-VERIFY(milestone-2): also pre-download the whisper + emissions models
        # (first pipeline() call does it today, making the first job slow).
