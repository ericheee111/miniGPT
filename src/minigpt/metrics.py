"""Record training telemetry as JSON Lines and process memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from pathlib import Path

type MetricValue = int | float | None


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Capture one step's losses, timing, throughput, and CPU memory."""

    step: int
    train_loss: float
    val_loss: float | None
    learning_rate: float
    step_time_ms: float
    tokens_per_sec: float
    data_time_ms: float
    forward_backward_time_ms: float
    optimizer_time_ms: float
    cpu_memory_mb: float


def append_jsonl(path: Path, metrics: TrainingMetrics) -> None:
    """Append one durable metrics record as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, MetricValue] = {
        "step": metrics.step,
        "train_loss": metrics.train_loss,
        "val_loss": metrics.val_loss,
        "learning_rate": metrics.learning_rate,
        "step_time_ms": metrics.step_time_ms,
        "tokens_per_sec": metrics.tokens_per_sec,
        "data_time_ms": metrics.data_time_ms,
        "forward_backward_time_ms": metrics.forward_backward_time_ms,
        "optimizer_time_ms": metrics.optimizer_time_ms,
        "cpu_memory_mb": metrics.cpu_memory_mb,
    }
    with path.open("a", encoding="utf-8") as stream:
        _ = stream.write(json.dumps(record, ensure_ascii=False))
        _ = stream.write("\n")


def cpu_memory_mb() -> float:
    """Return this process's resident memory in mebibytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)
