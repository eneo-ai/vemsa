from pathlib import Path

from vemsa.ops import host
from vemsa.ops.host import sample_host


def _sample(tmp_path: Path, **overrides):
    return sample_host(
        **{
            "worker_id": "w1",
            "engine": "fake",
            "device": "cpu",
            "gpu_name": None,
            "in_flight": 0,
            "concurrency": 1,
            "gpu_concurrency": 1,
            "work_dir": tmp_path / "does-not-exist-yet",
            **overrides,
        }
    )


def test_sample_reports_host_numbers_and_no_gpu_on_cpu(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(host, "_cpu_primed", False)
    first = _sample(tmp_path)
    second = _sample(tmp_path)

    # the first psutil CPU reading only starts the clock
    assert first.cpu_pct is None
    assert second.cpu_pct is not None and 0.0 <= second.cpu_pct <= 100.0
    assert second.load1 is not None and second.load1 >= 0.0
    assert second.mem_total_bytes and second.mem_used_bytes
    assert 0 < second.mem_used_bytes <= second.mem_total_bytes
    # the work dir does not exist yet: its nearest existing parent is measured
    assert second.disk_total_bytes and second.disk_used_bytes is not None
    assert second.hostname
    assert second.gpu_util_pct is None and second.gpu_mem_total_bytes is None
    assert second.gpu_temp_c is None


def test_sample_never_raises_when_readers_fail(tmp_path: Path, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("no /proc here")

    monkeypatch.setattr(host.psutil, "virtual_memory", explode)
    monkeypatch.setattr(host.psutil, "getloadavg", explode)
    monkeypatch.setattr(host.psutil, "disk_usage", explode)
    monkeypatch.setattr(host.socket, "gethostname", explode)

    sample = _sample(tmp_path)

    assert sample.mem_total_bytes is None and sample.load1 is None
    assert sample.disk_total_bytes is None and sample.hostname == "unknown"


def test_gpu_probe_falls_back_when_nvml_is_missing(monkeypatch):
    probe = host._GpuProbe()
    monkeypatch.setattr(probe, "_read_nvml", lambda: (_ for _ in ()).throw(ImportError("pynvml")))
    monkeypatch.setattr(probe, "_read_torch", staticmethod(lambda: (None, 5, 10, None)))

    assert probe.read("cuda") == (None, 5, 10, None)
    # the failure is remembered: the second read skips NVML entirely
    assert probe._nvml_failed
    assert probe.read("cuda") == (None, 5, 10, None)
    assert probe.read("cpu") == (None, None, None, None)
