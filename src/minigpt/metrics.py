from dataclasses import asdict, dataclass
import json
from pathlib import Path

import psutil


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
    with path.open("a", encoding="utf-8") as stream:
        json.dump(asdict(metrics), stream, ensure_ascii=False)
        stream.write("\n")


def cpu_memory_mb() -> float:
    """Return this process's resident memory in mebibytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)
