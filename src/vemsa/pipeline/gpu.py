"""Process-wide GPU admission control shared by every pipeline stage that runs a model.

`worker_concurrency` lets one worker overlap the CPU/network parts of N jobs
(download, ffmpeg transcode, the remote whisper round trip, persistence);
`gpu_concurrency` says how many of those may be inside a GPU stage at once. At
the default of 1 the GPU stays serialized, so concurrency changes scheduling
but provably cannot change results. Raising it is a GPU-VERIFY item (see
docs/PRODUCTION.md)."""

import contextlib
import logging
import threading
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_slots = threading.BoundedSemaphore(1)
_limit = 1


def configure_gpu_slots(limit: int) -> None:
    """Set how many pipeline threads may run a GPU stage concurrently.

    Called once from `build_engine`; only safe while no stage is in flight."""
    global _slots, _limit
    if limit < 1:
        raise ValueError("gpu concurrency must be at least 1")
    _slots = threading.BoundedSemaphore(limit)
    _limit = limit


def gpu_slot_limit() -> int:
    return _limit


@contextlib.contextmanager
def gpu_slot() -> Iterator[None]:
    """Hold one GPU slot for the duration of a model run."""
    with _slots:
        yield


def is_out_of_memory(exc: BaseException) -> bool:
    """Whether `exc` is a device or host out-of-memory condition.

    pyannote converts CUDA OOM into MemoryError; easyaligner lets
    torch.cuda.OutOfMemoryError (a RuntimeError subclass) through; older torch
    raised a bare RuntimeError mentioning "out of memory". Matched by name so
    this module never imports torch."""
    if isinstance(exc, MemoryError):
        return True
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def release_cached_memory() -> None:
    """Return cached CUDA blocks to the device after an OOM so the retry starts clean."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # torch missing, no device, or the driver is wedged
        logger.debug("could not release cached GPU memory", exc_info=True)
