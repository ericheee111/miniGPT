"""Typed records shared by CPU benchmarking and reporting."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkAggregationError(ValueError):
    """Report raw measurements that cannot form one summary."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the aggregation failure reason."""
        return f"cannot aggregate benchmark measurements: {self.reason}"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Identify one model, thread, batch, and context configuration."""

    model_size: str
    n_layer: int
    n_head: int
    n_embd: int
    thread_count: int
    block_size: int
    batch_size: int

    @property
    def label(self) -> str:
        """Return a stable, filename-safe case identifier."""
        return f"{self.model_size}-t{self.thread_count}-b{self.batch_size}-s{self.block_size}"


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    """Preserve one raw repeat without hiding timing variability."""

    case: BenchmarkCase
    repeat_index: int
    step_time_ms: float
    tokens_per_sec: float
    cpu_memory_mb: float
    parameter_count: int


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Summarize repeat stability and central performance for one case."""

    case: BenchmarkCase
    parameter_count: int
    repeat_count: int
    median_step_time_ms: float
    min_step_time_ms: float
    max_step_time_ms: float
    step_time_stddev_ms: float
    step_time_mad_ms: float
    step_time_cv_percent: float
    median_tokens_per_sec: float
    median_cpu_memory_mb: float


@dataclass(frozen=True, slots=True)
class BenchmarkArtifacts:
    """Locate raw, summarized, and reader-facing benchmark outputs."""

    raw_csv: Path
    summary_csv: Path
    report_markdown: Path
