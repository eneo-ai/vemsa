"""Host and GPU readings a worker records on every heartbeat.

Worker-only: `sample_host` is called from the control thread, never by the API
process. The numbers are host-wide (psutil reads /proc), not scoped to the
container's cgroup. Nothing here raises: a field that cannot be read is None."""

import logging
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import psutil

from vemsa.jobs.models import WorkerSample

logger = logging.getLogger(__name__)

T = TypeVar("T")

_GpuReading = tuple[float | None, int | None, int | None, float | None]
_NO_GPU: _GpuReading = (None, None, None, None)


def _safe(read: Callable[[], T]) -> T | None:
    try:
        return read()
    except Exception:
        logger.debug("host sample field unavailable", exc_info=True)
        return None


class _GpuProbe:
    """NVML first (utilisation, memory, temperature); torch memory only as fallback.

    A failed NVML init is remembered so a GPU-less host does not retry the
    library load on every heartbeat."""

    def __init__(self) -> None:
        self._handle = None
        self._nvml_failed = False

    def read(self, device: str | None) -> _GpuReading:
        if device != "cuda":
            return _NO_GPU
        if not self._nvml_failed:
            try:
                return self._read_nvml()
            except Exception:
                self._nvml_failed = True
                logger.info(
                    "NVML unavailable; GPU utilisation and temperature will not be sampled",
                    exc_info=True,
                )
        return self._read_torch()

    def _read_nvml(self) -> _GpuReading:
        import pynvml

        if self._handle is None:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilisation = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        temperature = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
        return float(utilisation), int(memory.used), int(memory.total), float(temperature)

    @staticmethod
    def _read_torch() -> _GpuReading:
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return None, int(total - free), int(total), None
        except Exception:
            logger.debug("torch GPU memory unavailable", exc_info=True)
        return _NO_GPU


_gpu = _GpuProbe()
_cpu_primed = False


def sample_host(
    *,
    worker_id: str,
    engine: str,
    device: str | None,
    gpu_name: str | None,
    in_flight: int,
    concurrency: int,
    gpu_concurrency: int,
    work_dir: Path,
) -> WorkerSample:
    global _cpu_primed
    # psutil measures CPU between calls; the first call only starts the clock
    cpu_pct = _safe(lambda: psutil.cpu_percent(interval=None))
    if not _cpu_primed:
        _cpu_primed = True
        cpu_pct = None
    memory = _safe(psutil.virtual_memory)
    disk = _safe(lambda: psutil.disk_usage(str(_existing_dir(work_dir))))
    gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = _gpu.read(device)
    return WorkerSample(
        worker_id=worker_id,
        sampled_at=datetime.now(UTC),
        hostname=_safe(socket.gethostname) or "unknown",
        engine=engine,
        device=device,
        gpu_name=gpu_name,
        in_flight=in_flight,
        concurrency=concurrency,
        gpu_concurrency=gpu_concurrency,
        cpu_pct=cpu_pct,
        load1=_safe(lambda: psutil.getloadavg()[0]),
        mem_used_bytes=memory.total - memory.available if memory else None,
        mem_total_bytes=memory.total if memory else None,
        disk_used_bytes=disk.used if disk else None,
        disk_total_bytes=disk.total if disk else None,
        gpu_util_pct=gpu_util,
        gpu_mem_used_bytes=gpu_mem_used,
        gpu_mem_total_bytes=gpu_mem_total,
        gpu_temp_c=gpu_temp,
    )


def _existing_dir(path: Path) -> Path:
    """The work dir may not exist before the first job; measure its nearest parent."""
    for candidate in (path, *path.resolve().parents):
        if candidate.is_dir():
            return candidate
    return Path("/")
