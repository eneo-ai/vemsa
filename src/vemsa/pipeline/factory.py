import logging

from vemsa.config import Settings
from vemsa.pipeline.base import TranscriptionEngine
from vemsa.pipeline.gpu import configure_gpu_slots

logger = logging.getLogger(__name__)


def build_engine(settings: Settings) -> TranscriptionEngine:
    configure_gpu_slots(settings.gpu_concurrency)
    engine = settings.resolve_engine()
    logger.info(
        "transcription engine selected",
        extra={"event": "engine.selected", "engine": engine, "configured_engine": settings.engine},
    )
    if engine == "fake":
        logger.warning("fake engine is enabled — returning canned transcripts")
        from vemsa.pipeline.fake import CannedEngine

        return CannedEngine()
    if engine == "local":
        from vemsa.pipeline.transcribe import EasyTranscriberEngine

        return EasyTranscriberEngine(settings)
    if engine == "hybrid":
        from vemsa.pipeline.hybrid import HybridEngine

        return HybridEngine(settings)
    if engine == "diarize":
        from vemsa.pipeline.diarize_only import DiarizeOnlyEngine

        return DiarizeOnlyEngine(settings)
    from vemsa.pipeline.whisper_api import OpenAIWhisperEngine

    return OpenAIWhisperEngine(settings)
