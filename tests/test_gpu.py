import threading
import time

from vemsa.pipeline import gpu


def _run_with_slots(limit: int, threads: int) -> int:
    """Peak number of threads inside gpu_slot() at once."""
    lock = threading.Lock()
    state = {"inside": 0, "peak": 0}

    def work() -> None:
        with gpu.gpu_slot():
            with lock:
                state["inside"] += 1
                state["peak"] = max(state["peak"], state["inside"])
            time.sleep(0.05)
            with lock:
                state["inside"] -= 1

    gpu.configure_gpu_slots(limit)
    try:
        workers = [threading.Thread(target=work) for _ in range(threads)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    finally:
        gpu.configure_gpu_slots(1)
    return state["peak"]


def test_gpu_slot_bounds_concurrency():
    assert _run_with_slots(1, 4) == 1
    assert _run_with_slots(2, 4) == 2
    assert gpu.gpu_slot_limit() == 1


def test_gpu_slot_limit_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        gpu.configure_gpu_slots(0)


def test_is_out_of_memory_classifies():
    class OutOfMemoryError(RuntimeError):  # noqa: N818 — same name as torch's
        pass

    assert gpu.is_out_of_memory(MemoryError("pyannote batch"))
    assert gpu.is_out_of_memory(OutOfMemoryError("CUDA"))
    assert gpu.is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not gpu.is_out_of_memory(RuntimeError("boom"))
    assert not gpu.is_out_of_memory(ValueError("out of memory"))


def test_release_cached_memory_never_raises():
    gpu.release_cached_memory()
