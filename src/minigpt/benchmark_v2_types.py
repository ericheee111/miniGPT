"""Immutable records that define one CPU Benchmark v2 experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkV2Case:
    """Describe one explicit benchmark case without matrix expansion."""

    name: str
    model_name: str
    n_layer: int
    n_head: int
    n_embd: int
    torch_num_threads: int
    block_size: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class ProfileV2Settings:
    """Select an optional explicit case for a separate operator profile."""

    enabled: bool
    case_name: str
    warmup_steps: int
    active_steps: int


@dataclass(frozen=True, slots=True)
class PreconditioningV2Settings:
    """Define one explicit unmeasured training-step preconditioning phase."""

    enabled: bool
    case_name: str
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class BenchmarkV2Config:
    """Define strict, reproducible CPU Benchmark v2 settings."""

    schema_version: int
    experiment_name: str
    benchmark_seed: int
    vocab_size: int
    output_root: Path
    worker_timeout_seconds: float
    warmup_steps: int
    measurement_steps: int
    replicates: int
    torch_num_interop_threads: int
    cpu_affinity: tuple[int, ...] | None
    max_cv_percent: float
    minimum_replicates: int
    regression_threshold_percent: float
    relevant_environment_variables: tuple[str, ...]
    cases: tuple[BenchmarkV2Case, ...]
    profile: ProfileV2Settings
    preconditioning: PreconditioningV2Settings = field(
        default_factory=lambda: PreconditioningV2Settings(
            enabled=False,
            case_name="",
            duration_seconds=0.0,
        )
    )
