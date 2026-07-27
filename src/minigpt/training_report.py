"""Validate and summarize source evidence for reference-training reports."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from minigpt.checkpoint import (
    CheckpointMetadata,
    compute_dataset_fingerprints,
    load_checkpoint_metadata,
)
from minigpt.config import load_experiment_config
from minigpt.data import CharTokenizer
from minigpt.optimization import learning_rate_at_step
from minigpt.run_provenance import (
    RunProvenance,
    find_repository_root,
    load_run_provenance,
    read_git_identity,
)

if TYPE_CHECKING:
    from minigpt.settings import ExperimentConfig

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None

_METRIC_FIELDS: Final = frozenset(
    {
        "step",
        "train_loss",
        "val_loss",
        "learning_rate",
        "step_time_ms",
        "tokens_per_sec",
        "data_time_ms",
        "forward_backward_time_ms",
        "optimizer_time_ms",
        "cpu_memory_mb",
    }
)
_SAMPLE_HEADER: Final = re.compile(r"(?m)^step=(\d+)\n")
_EMPTY_METRICS_REASON: Final = "metrics must contain at least one record"
_CONTIGUOUS_STEPS_REASON: Final = "metric steps must be contiguous from zero"
_EMPTY_SAMPLES_REASON: Final = "sample file must contain at least one step header"
_INCREASING_SAMPLES_REASON: Final = "sample steps must be strictly increasing"
_VALIDATION_REQUIRED_REASON: Final = "metrics must contain at least one validation loss"
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_MINIMUM_RESUME_SEGMENTS: Final = 2


@dataclass(slots=True)
class TrainingEvidenceError(ValueError):
    """Report invalid or mutually inconsistent training evidence."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the invalid evidence path and exact reason."""
        return f"invalid training evidence {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """Represent one strictly validated metrics JSONL record."""

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


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    """Represent generated text associated with one global training step."""

    step: int
    text: str


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Hold mechanically derived conclusions from validated metrics."""

    total_steps: int
    initial_train_loss: float
    final_train_loss: float
    best_train_loss: float
    initial_val_loss: float
    final_val_loss: float
    best_val_loss: float
    median_tokens_per_sec: float
    peak_cpu_memory_mb: float


@dataclass(frozen=True, slots=True)
class ReportInputs:
    """Locate every immutable source consumed by report generation."""

    repository_root: Path
    config_path: Path
    metrics_path: Path
    samples_path: Path
    checkpoint_path: Path
    provenance_path: Path


@dataclass(frozen=True, slots=True)
class ValidatedReportData:
    """Expose one internally consistent reference-training experiment."""

    config: ExperimentConfig
    tokenizer: CharTokenizer
    metrics: tuple[MetricRecord, ...]
    samples: tuple[GeneratedSample, ...]
    provenance: RunProvenance
    metadata: CheckpointMetadata
    summary: TrainingSummary
    train_token_count: int
    val_token_count: int
    checkpoint_sha256: str


def _mapping(value: JsonValue, path: Path, line_number: int) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        reason = f"line {line_number} JSON value must be an object"
        raise TrainingEvidenceError(path, reason)
    return value


def _step(document: dict[str, JsonValue], path: Path, line_number: int) -> int:
    value = document["step"]
    if isinstance(value, bool) or not isinstance(value, int):
        reason = f"line {line_number} step must be an integer"
        raise TrainingEvidenceError(path, reason)
    return value


def _finite_number(
    document: dict[str, JsonValue],
    key: str,
    path: Path,
    line_number: int,
) -> float:
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        reason = f"line {line_number} {key} must be numeric"
        raise TrainingEvidenceError(path, reason)
    result = float(value)
    if not math.isfinite(result):
        reason = f"line {line_number} {key} must be finite"
        raise TrainingEvidenceError(path, reason)
    return result


def _optional_finite_number(
    document: dict[str, JsonValue],
    key: str,
    path: Path,
    line_number: int,
) -> float | None:
    if document[key] is None:
        return None
    return _finite_number(document, key, path, line_number)


def _parse_metric_line(line: str, path: Path, line_number: int) -> MetricRecord:
    try:
        raw_document = cast("JsonValue", json.loads(line))
    except json.JSONDecodeError as error:
        reason = f"line {line_number} contains invalid JSON: {error.msg}"
        raise TrainingEvidenceError(path, reason) from error
    document = _mapping(raw_document, path, line_number)
    fields = frozenset(document)
    if fields != _METRIC_FIELDS:
        missing = sorted(_METRIC_FIELDS - fields)
        extra = sorted(fields - _METRIC_FIELDS)
        reason = f"line {line_number} has invalid fields; missing={missing}, extra={extra}"
        raise TrainingEvidenceError(path, reason)
    return MetricRecord(
        step=_step(document, path, line_number),
        train_loss=_finite_number(document, "train_loss", path, line_number),
        val_loss=_optional_finite_number(document, "val_loss", path, line_number),
        learning_rate=_finite_number(document, "learning_rate", path, line_number),
        step_time_ms=_finite_number(document, "step_time_ms", path, line_number),
        tokens_per_sec=_finite_number(document, "tokens_per_sec", path, line_number),
        data_time_ms=_finite_number(document, "data_time_ms", path, line_number),
        forward_backward_time_ms=_finite_number(
            document,
            "forward_backward_time_ms",
            path,
            line_number,
        ),
        optimizer_time_ms=_finite_number(document, "optimizer_time_ms", path, line_number),
        cpu_memory_mb=_finite_number(document, "cpu_memory_mb", path, line_number),
    )


def load_metric_records(path: Path) -> tuple[MetricRecord, ...]:
    """Parse strict, finite, contiguous training metrics from JSON Lines."""
    records = tuple(
        _parse_metric_line(line, path, line_number)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if line
    )
    if not records:
        raise TrainingEvidenceError(path, _EMPTY_METRICS_REASON)
    if [record.step for record in records] != list(range(len(records))):
        raise TrainingEvidenceError(path, _CONTIGUOUS_STEPS_REASON)
    return records


def load_generated_samples(path: Path) -> tuple[GeneratedSample, ...]:
    """Parse chronological samples written by the trainer."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    headers = tuple(_SAMPLE_HEADER.finditer(normalized))
    if not headers:
        raise TrainingEvidenceError(path, _EMPTY_SAMPLES_REASON)
    samples: list[GeneratedSample] = []
    for index, header in enumerate(headers):
        body_end = headers[index + 1].start() if index + 1 < len(headers) else len(normalized)
        body = normalized[header.end() : body_end]
        if body.endswith("\n\n"):
            body = body[:-2]
        elif body.endswith("\n"):
            body = body[:-1]
        samples.append(GeneratedSample(step=int(header.group(1)), text=body))
    steps = [sample.step for sample in samples]
    if any(current >= following for current, following in pairwise(steps)):
        raise TrainingEvidenceError(path, _INCREASING_SAMPLES_REASON)
    return tuple(samples)


def select_representative_samples(
    samples: tuple[GeneratedSample, ...],
) -> tuple[GeneratedSample, GeneratedSample, GeneratedSample]:
    """Select fixed first, middle, and final samples without editorial judgment."""
    if not samples:
        raise TrainingEvidenceError(Path("<memory>"), _EMPTY_SAMPLES_REASON)
    return samples[0], samples[len(samples) // 2], samples[-1]


def summarize_training(records: tuple[MetricRecord, ...]) -> TrainingSummary:
    """Reduce validated raw metrics into stable report conclusions."""
    if not records:
        raise TrainingEvidenceError(Path("<memory>"), _EMPTY_METRICS_REASON)
    validation_losses = tuple(record.val_loss for record in records if record.val_loss is not None)
    if not validation_losses:
        raise TrainingEvidenceError(Path("<memory>"), _VALIDATION_REQUIRED_REASON)
    return TrainingSummary(
        total_steps=len(records),
        initial_train_loss=records[0].train_loss,
        final_train_loss=records[-1].train_loss,
        best_train_loss=min(record.train_loss for record in records),
        initial_val_loss=validation_losses[0],
        final_val_loss=validation_losses[-1],
        best_val_loss=min(validation_losses),
        median_tokens_per_sec=statistics.median(record.tokens_per_sec for record in records),
        peak_cpu_memory_mb=max(record.cpu_memory_mb for record in records),
    )


def character_perplexity(loss: float) -> float:
    """Convert mean character cross-entropy to character-level perplexity."""
    perplexity = math.exp(loss)
    if not math.isfinite(perplexity):
        raise TrainingEvidenceError(Path("<memory>"), "character perplexity must be finite")
    return perplexity


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_data_directory(config: ExperimentConfig, repository_root: Path) -> Path:
    directory = config.data.directory
    return directory if directory.is_absolute() else repository_root / directory


def _validate_repository(inputs: ReportInputs, provenance: RunProvenance) -> None:
    repository_root = inputs.repository_root.resolve()
    discovered_root = find_repository_root(inputs.config_path)
    if repository_root != discovered_root:
        raise TrainingEvidenceError(
            inputs.config_path,
            "config and declared repository root differ",
        )
    git = read_git_identity(repository_root)
    if git.dirty:
        raise TrainingEvidenceError(repository_root, "Git tree must be clean")
    if git.commit_sha != provenance.git.commit_sha:
        raise TrainingEvidenceError(
            inputs.provenance_path,
            "Git commit differs from run provenance",
        )
    relative_config_path = inputs.config_path.resolve().relative_to(repository_root).as_posix()
    if relative_config_path != provenance.config_path:
        raise TrainingEvidenceError(
            inputs.config_path,
            "config path differs from run provenance",
        )


def _validate_config_and_data(
    inputs: ReportInputs,
    provenance: RunProvenance,
    metadata: CheckpointMetadata,
) -> tuple[ExperimentConfig, CharTokenizer, int, int]:
    source_config = load_experiment_config(inputs.config_path)
    data_directory = _resolved_data_directory(source_config, inputs.repository_root)
    tokenizer = CharTokenizer.load(data_directory / "tokenizer.json")
    resolved_config = source_config.resolve_vocab_size(tokenizer.vocab_size)
    if resolved_config != metadata.config:
        raise TrainingEvidenceError(
            inputs.checkpoint_path,
            "resolved config differs from checkpoint config",
        )
    source_config_sha256 = _sha256_file(inputs.config_path)
    if source_config_sha256 != provenance.source_config_sha256:
        raise TrainingEvidenceError(
            inputs.provenance_path,
            "source config SHA-256 differs from provenance",
        )
    resolved_config_sha256 = sha256(resolved_config.to_yaml().encode("utf-8")).hexdigest()
    if resolved_config_sha256 != provenance.resolved_config_sha256:
        raise TrainingEvidenceError(
            inputs.provenance_path,
            "resolved config hash differs from provenance",
        )
    absolute_data = replace(source_config.data, directory=data_directory)
    fingerprints = compute_dataset_fingerprints(absolute_data)
    if (
        fingerprints != metadata.dataset_fingerprints
        or fingerprints != provenance.dataset_fingerprints
    ):
        raise TrainingEvidenceError(
            data_directory,
            "dataset fingerprint differs across current data, checkpoint, and provenance",
        )
    train_tokens = cast(
        "npt.NDArray[np.uint16]",
        np.load(absolute_data.train_path, mmap_mode="r"),
    )
    val_tokens = cast(
        "npt.NDArray[np.uint16]",
        np.load(absolute_data.val_path, mmap_mode="r"),
    )
    return resolved_config, tokenizer, int(train_tokens.size), int(val_tokens.size)


def _validate_metrics(
    path: Path,
    config: ExperimentConfig,
    metrics: tuple[MetricRecord, ...],
) -> None:
    if len(metrics) != config.training.max_steps:
        raise TrainingEvidenceError(path, "metric count differs from configured max_steps")
    for record in metrics:
        expected = learning_rate_at_step(
            record.step,
            max_learning_rate=config.optimizer.learning_rate,
            min_learning_rate=config.optimizer.min_learning_rate,
            warmup_steps=config.training.warmup_steps,
            lr_decay_steps=config.training.lr_decay_steps,
        )
        if not math.isclose(record.learning_rate, expected, rel_tol=1e-12, abs_tol=0.0):
            reason = (
                f"step {record.step} learning rate {record.learning_rate} "
                f"differs from configured schedule {expected}"
            )
            raise TrainingEvidenceError(path, reason)


def _validate_samples(
    path: Path,
    config: ExperimentConfig,
    samples: tuple[GeneratedSample, ...],
) -> None:
    interval = config.training.sample_interval
    expected_steps = tuple(range(interval - 1, config.training.max_steps, interval))
    if tuple(sample.step for sample in samples) != expected_steps:
        raise TrainingEvidenceError(path, "sample steps differ from the configured schedule")
    if not expected_steps or expected_steps[-1] != config.training.max_steps - 1:
        raise TrainingEvidenceError(
            path, "sample interval does not produce a final scheduled sample"
        )


def _validate_provenance(
    inputs: ReportInputs,
    provenance: RunProvenance,
    metadata: CheckpointMetadata,
    checkpoint_sha256: str,
) -> None:
    if provenance.git.dirty:
        raise TrainingEvidenceError(
            inputs.provenance_path, "run provenance records a dirty Git tree"
        )
    if provenance.ended_at_utc is None:
        raise TrainingEvidenceError(inputs.provenance_path, "run provenance has no end time")
    if len(provenance.segments) < _MINIMUM_RESUME_SEGMENTS or not any(
        segment.resume_checkpoint_sha256 is not None for segment in provenance.segments[1:]
    ):
        raise TrainingEvidenceError(
            inputs.provenance_path,
            "run provenance must contain a completed resume segment",
        )
    if any(
        segment.status != "completed" or segment.ended_at_utc is None
        for segment in provenance.segments
    ):
        raise TrainingEvidenceError(
            inputs.provenance_path,
            "every run segment must be completed",
        )
    final_segment = provenance.segments[-1]
    if final_segment.final_completed_step != metadata.completed_step:
        raise TrainingEvidenceError(
            inputs.provenance_path,
            "final provenance step differs from checkpoint completed step",
        )
    if final_segment.checkpoint_sha256 != checkpoint_sha256:
        raise TrainingEvidenceError(
            inputs.checkpoint_path,
            "checkpoint SHA-256 differs from final provenance segment",
        )


def validate_reference_sources(inputs: ReportInputs) -> ValidatedReportData:
    """Validate all reference sources before generating any output."""
    provenance = load_run_provenance(inputs.provenance_path)
    metadata = load_checkpoint_metadata(inputs.checkpoint_path)
    expected_completed_step = metadata.config.training.max_steps - 1
    if metadata.completed_step != expected_completed_step:
        raise TrainingEvidenceError(
            inputs.checkpoint_path,
            "checkpoint completed step does not reach configured max_steps",
        )
    _validate_repository(inputs, provenance)
    config, tokenizer, train_token_count, val_token_count = _validate_config_and_data(
        inputs,
        provenance,
        metadata,
    )
    metrics = load_metric_records(inputs.metrics_path)
    _validate_metrics(inputs.metrics_path, config, metrics)
    samples = load_generated_samples(inputs.samples_path)
    _validate_samples(inputs.samples_path, config, samples)
    checkpoint_sha256 = _sha256_file(inputs.checkpoint_path)
    _validate_provenance(inputs, provenance, metadata, checkpoint_sha256)
    return ValidatedReportData(
        config=config,
        tokenizer=tokenizer,
        metrics=metrics,
        samples=samples,
        provenance=provenance,
        metadata=metadata,
        summary=summarize_training(metrics),
        train_token_count=train_token_count,
        val_token_count=val_token_count,
        checkpoint_sha256=checkpoint_sha256,
    )
