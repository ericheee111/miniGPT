"""Capture CPU controls and native process evidence for Benchmark v2 workers."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import psutil
import torch

if sys.platform.startswith("linux"):
    import resource as _resource
else:
    _resource = None

PeakRssMethod = Literal[
    "windows_peak_working_set",
    "linux_getrusage_ru_maxrss",
]

_MIB = 1024 * 1024


class _WindowsMemoryInfo(Protocol):
    """Describe the Windows peak field absent from cross-platform psutil typing."""

    rss: int
    peak_wset: int


class _AffinityProcess(Protocol):
    """Describe the psutil affinity API available on Windows and Linux."""

    def cpu_affinity(self, cpus: list[int] | None = None) -> list[int]:
        """Set affinity when provided and return the effective CPU IDs."""
        ...


@dataclass(frozen=True, slots=True)
class ProcessMemoryEvidence:
    """Record final RSS and the OS-native lifetime peak without sampling."""

    final_rss_mib: float
    peak_rss_mib: float
    peak_rss_method: PeakRssMethod
    peak_rss_sampling_interval_ms: None


@dataclass(frozen=True, slots=True)
class WorkerEnvironment:
    """Record the runtime and effective CPU controls of one fresh worker."""

    platform: str
    python_version: str
    torch_version: str
    torch_num_threads: int
    torch_num_interop_threads: int
    logical_cpu_count: int | None
    requested_cpu_affinity: tuple[int, ...] | None
    effective_cpu_affinity: tuple[int, ...] | None
    relevant_environment_variables: dict[str, str | None]


def apply_cpu_affinity(requested: tuple[int, ...] | None) -> tuple[int, ...] | None:
    """Apply and read back the requested logical CPU affinity."""
    if requested is None:
        return None
    process = cast("_AffinityProcess", cast("object", psutil.Process()))
    _ = process.cpu_affinity(list(requested))
    return tuple(process.cpu_affinity())


def read_process_memory() -> ProcessMemoryEvidence:
    """Return final RSS and a platform-native lifetime peak in mebibytes."""
    memory_info = psutil.Process().memory_info()
    final_rss_mib = memory_info.rss / _MIB
    if sys.platform == "win32":
        peak_rss_bytes = cast("_WindowsMemoryInfo", cast("object", memory_info)).peak_wset
        return ProcessMemoryEvidence(
            final_rss_mib=final_rss_mib,
            peak_rss_mib=peak_rss_bytes / _MIB,
            peak_rss_method="windows_peak_working_set",
            peak_rss_sampling_interval_ms=None,
        )
    if sys.platform.startswith("linux"):
        if _resource is None:
            msg = "resource module unavailable on Linux"
            raise RuntimeError(msg)
        peak_rss_kib = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        return ProcessMemoryEvidence(
            final_rss_mib=final_rss_mib,
            peak_rss_mib=peak_rss_kib / 1024,
            peak_rss_method="linux_getrusage_ru_maxrss",
            peak_rss_sampling_interval_ms=None,
        )
    msg = f"unsupported platform for native peak RSS: {sys.platform}"
    raise RuntimeError(msg)


def capture_worker_environment(
    *,
    requested_cpu_affinity: tuple[int, ...] | None,
    effective_cpu_affinity: tuple[int, ...] | None,
    relevant_environment_variables: tuple[str, ...],
) -> WorkerEnvironment:
    """Capture runtime identity and CPU controls after one measurement."""
    return WorkerEnvironment(
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        logical_cpu_count=psutil.cpu_count(logical=True),
        requested_cpu_affinity=requested_cpu_affinity,
        effective_cpu_affinity=effective_cpu_affinity,
        relevant_environment_variables={
            name: os.environ.get(name) for name in relevant_environment_variables
        },
    )
