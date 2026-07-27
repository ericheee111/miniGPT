"""Validate and summarize source evidence for reference-training reports."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import import_module
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias, cast

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import torch
from typing_extensions import override

from minigpt.checkpoint import (
    CheckpointMetadata,
    compute_dataset_fingerprints,
    load_checkpoint_metadata,
)
from minigpt.config import load_experiment_config
from minigpt.data import CharTokenizer
from minigpt.model import GPT
from minigpt.optimization import create_sample_generator, learning_rate_at_step, seed_everything
from minigpt.run_provenance import (
    RunProvenance,
    find_repository_root,
    load_run_provenance,
    read_git_identity,
)

if TYPE_CHECKING:
    from minigpt.settings import ExperimentConfig


class _Figure(Protocol):
    def suptitle(self, title: str) -> None: ...

    def tight_layout(self) -> None: ...

    def savefig(
        self,
        path: Path,
        *,
        dpi: int,
        metadata: dict[str, str],
    ) -> None: ...


class _PlotModule(Protocol):
    def figure(self, *, figsize: tuple[float, float]) -> _Figure: ...

    def gcf(self) -> _Figure: ...

    def plot(
        self,
        steps: tuple[int, ...],
        values: tuple[float, ...],
        *,
        label: str,
        linewidth: float,
    ) -> None: ...

    def scatter(
        self,
        steps: tuple[int, ...],
        values: tuple[float, ...],
        *,
        label: str,
        marker: str,
        s: int,
    ) -> None: ...

    def xlabel(self, label: str) -> None: ...

    def ylabel(self, label: str) -> None: ...

    def grid(self, *, visible: bool, alpha: float) -> None: ...

    def legend(self) -> None: ...

    def axhline(
        self,
        y: float,
        *,
        color: str,
        linestyle: str,
        linewidth: float,
        label: str,
    ) -> None: ...

    def close(self, figure: _Figure) -> None: ...


mpl.use("Agg")
_PLOT = cast("_PlotModule", cast("object", import_module("matplotlib.pyplot")))

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
_SAMPLE_TEMPERATURE: Final = 0.8
_SAMPLE_TOP_K: Final = 20
_REPORT_SCHEMA_VERSION: Final = 1
_FIGURE_SIZE: Final = (12.0, 6.75)
_FIGURE_DPI: Final = 120
_MATERIAL_LOSS_RATIO: Final = 0.9
_CSV_FIELDS: Final = (
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
)


@dataclass(slots=True)
class TrainingEvidenceError(ValueError):
    """Report invalid or mutually inconsistent training evidence."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the invalid evidence path and exact reason."""
        return f"invalid training evidence {self.path}: {self.reason}"


@dataclass(slots=True)
class ReportOutputExistsError(FileExistsError):
    """Reject generation into a directory that may contain mixed evidence."""

    path: Path

    @override
    def __str__(self) -> str:
        """Render the destination that must remain untouched."""
        return f"training report output already exists: {self.path}"


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


@dataclass(frozen=True, slots=True)
class TrainingSeries:
    """Expose unsmoothed numerical series used by report figures."""

    train_loss: tuple[tuple[int, float], ...]
    validation_loss: tuple[tuple[int, float], ...]
    learning_rate: tuple[tuple[int, float], ...]
    tokens_per_sec: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    """Locate every generated compact reference-training artifact."""

    readme: Path
    environment: Path
    resolved_config: Path
    metrics_csv: Path
    loss_curve: Path
    learning_rate_curve: Path
    throughput_curve: Path
    generated_samples: Path
    manifest: Path


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
    if git.branch != provenance.git.branch:
        raise TrainingEvidenceError(
            inputs.provenance_path,
            "Git branch differs from run provenance",
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
        validation_due = (record.step + 1) % config.training.eval_interval == 0
        if (record.val_loss is not None) != validation_due:
            reason = f"step {record.step} validation schedule does not match eval_interval"
            raise TrainingEvidenceError(path, reason)
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
    if metrics[-1].train_loss > metrics[0].train_loss * _MATERIAL_LOSS_RATIO:
        reason = (
            "final train loss must be materially lower than initial loss "
            f"(at most {_MATERIAL_LOSS_RATIO:.0%})"
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
    if provenance.segments[0].resume_checkpoint_sha256 is not None:
        raise TrainingEvidenceError(
            inputs.provenance_path,
            "first run segment cannot have a resume checkpoint SHA-256",
        )
    for previous, current in pairwise(provenance.segments):
        if (
            previous.checkpoint_sha256 is None
            or current.resume_checkpoint_sha256 != previous.checkpoint_sha256
        ):
            raise TrainingEvidenceError(
                inputs.provenance_path,
                "resume checkpoint SHA-256 does not match the preceding segment output",
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


def build_training_series(records: tuple[MetricRecord, ...]) -> TrainingSeries:
    """Build exact observed figure points without smoothing or interpolation."""
    return TrainingSeries(
        train_loss=tuple((record.step, record.train_loss) for record in records),
        validation_loss=tuple(
            (record.step, record.val_loss) for record in records if record.val_loss is not None
        ),
        learning_rate=tuple((record.step, record.learning_rate) for record in records),
        tokens_per_sec=tuple((record.step, record.tokens_per_sec) for record in records),
    )


def figure_title(metric_name: str, experiment_name: str, commit_sha: str) -> str:
    """Build a stable title containing experiment and source identity."""
    return f"{metric_name} — {experiment_name} @ {commit_sha[:8]}"


def throughput_reference_line(series: TrainingSeries) -> tuple[str, float]:
    """Return the stable labeled median used by the throughput figure."""
    median = statistics.median(value for _, value in series.tokens_per_sec)
    return f"median tokens/s: {median:.2f}", median


def _write_metrics_csv(path: Path, records: tuple[MetricRecord, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "step": record.step,
                    "train_loss": record.train_loss,
                    "val_loss": record.val_loss,
                    "learning_rate": record.learning_rate,
                    "step_time_ms": record.step_time_ms,
                    "tokens_per_sec": record.tokens_per_sec,
                    "data_time_ms": record.data_time_ms,
                    "forward_backward_time_ms": record.forward_backward_time_ms,
                    "optimizer_time_ms": record.optimizer_time_ms,
                    "cpu_memory_mb": record.cpu_memory_mb,
                }
            )


def _save_figure(path: Path, title: str) -> None:
    figure = _PLOT.gcf()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=_FIGURE_DPI,
        metadata={"Software": f"miniGPT {title.rsplit(' ', maxsplit=1)[-1]}"},
    )
    _PLOT.close(figure)


def _plot_loss(path: Path, series: TrainingSeries, title: str) -> None:
    _ = _PLOT.figure(figsize=_FIGURE_SIZE)
    train_steps, train_values = zip(*series.train_loss, strict=True)
    validation_steps, validation_values = zip(*series.validation_loss, strict=True)
    _PLOT.plot(train_steps, train_values, label="train loss", linewidth=1.5)
    _PLOT.scatter(
        validation_steps,
        validation_values,
        label="validation loss (observed)",
        marker="o",
        s=24,
    )
    _PLOT.xlabel("global step")
    _PLOT.ylabel("cross-entropy loss")
    _PLOT.grid(visible=True, alpha=0.25)
    _PLOT.legend()
    _save_figure(path, title)


def _plot_single_series(
    path: Path,
    points: tuple[tuple[int, float], ...],
    *,
    title: str,
    label: str,
    y_axis_label: str,
) -> None:
    _ = _PLOT.figure(figsize=_FIGURE_SIZE)
    steps, values = zip(*points, strict=True)
    _PLOT.plot(steps, values, label=label, linewidth=1.5)
    _PLOT.xlabel("global step")
    _PLOT.ylabel(y_axis_label)
    _PLOT.grid(visible=True, alpha=0.25)
    _PLOT.legend()
    _save_figure(path, title)


def _plot_throughput(path: Path, series: TrainingSeries, title: str) -> None:
    _ = _PLOT.figure(figsize=_FIGURE_SIZE)
    steps, values = zip(*series.tokens_per_sec, strict=True)
    _PLOT.plot(steps, values, label="optimization-step tokens/s", linewidth=1.0)
    median_label, median = throughput_reference_line(series)
    _PLOT.axhline(
        median,
        color="black",
        linestyle="--",
        linewidth=1.25,
        label=median_label,
    )
    _PLOT.xlabel("global step")
    _PLOT.ylabel("tokens/s")
    _PLOT.grid(visible=True, alpha=0.25)
    _PLOT.legend()
    _save_figure(path, title)


def _baseline_sample(data: ValidatedReportData) -> tuple[str, int]:
    config = data.config
    seed_everything(config.runtime.seed, config.runtime.num_threads)
    model = GPT(config.model.to_gpt_config(config.data.block_size))
    _ = model.eval()
    prompt = torch.tensor(
        [data.tokenizer.encode(config.training.sample_prompt)],
        dtype=torch.long,
    )
    generated = model.generate(
        prompt,
        max_new_tokens=config.training.sample_tokens,
        temperature=_SAMPLE_TEMPERATURE,
        top_k=min(_SAMPLE_TOP_K, data.tokenizer.vocab_size),
        generator=create_sample_generator(config.runtime.seed),
    )
    token_ids = [int(generated[0, index]) for index in range(generated.shape[1])]
    return data.tokenizer.decode(token_ids), model.parameter_count()


def _environment_document(
    data: ValidatedReportData,
    model_parameter_count: int,
) -> dict[str, JsonValue]:
    provenance = data.provenance
    environment = provenance.environment
    fingerprints = data.metadata.dataset_fingerprints
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "experiment_name": provenance.experiment_name,
        "git_commit_sha": provenance.git.commit_sha,
        "git_branch": provenance.git.branch,
        "git_dirty": provenance.git.dirty,
        "run_started_at_utc": provenance.started_at_utc,
        "run_ended_at_utc": provenance.ended_at_utc,
        "operating_system": environment.operating_system,
        "machine": environment.machine,
        "python_version": environment.python_version,
        "pytorch_version": environment.pytorch_version,
        "numpy_version": environment.numpy_version,
        "cpu_name": environment.cpu_name,
        "physical_cores": environment.physical_cores,
        "logical_cores": environment.logical_cores,
        "torch_num_threads": environment.torch_num_threads,
        "cuda_available": environment.cuda_available,
        "dataset_fingerprints": {
            "tokenizer_sha256": fingerprints.tokenizer_sha256,
            "train_sha256": fingerprints.train_sha256,
            "val_sha256": fingerprints.val_sha256,
        },
        "checkpoint_format_version": data.metadata.format_version,
        "checkpoint_sha256": data.checkpoint_sha256,
        "model_parameter_count": model_parameter_count,
        "resolved_config_sha256": data.provenance.resolved_config_sha256,
    }


def _write_json(path: Path, document: dict[str, JsonValue]) -> None:
    _ = path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _command_lines(provenance: RunProvenance) -> str:
    return "\n".join(
        f"python train.py {subprocess.list2cmdline(list(segment.argv))}"
        for segment in provenance.segments
    )


def _sample_markdown(label: str, sample: GeneratedSample) -> str:
    return f"### {label} (step {sample.step})\n\n```text\n{sample.text}\n```"


def _word_like_run_count(text: str) -> int:
    return len(re.findall(r"(?<![A-Za-z])[A-Za-z]{2,}(?![A-Za-z])", text))


def _local_structure_observation(baseline: str, final: str) -> str:
    baseline_runs = _word_like_run_count(baseline)
    final_runs = _word_like_run_count(final)
    comparison = "more" if final_runs > baseline_runs else "no more"
    return (
        "**Deterministic local-structure observation:** the final sample contains "
        f"{final_runs} alphabetic runs of at least two characters versus {baseline_runs} "
        f"in the untrained baseline ({comparison} by this fixed heuristic). "
        "This is evidence about local word-like structure, not fluency or generalization."
    )


def _result_row(
    label: str,
    initial: float,
    final: float,
    best: float,
    *,
    precision: int,
) -> str:
    return f"| {label} | {initial:.{precision}f} | {final:.{precision}f} | {best:.{precision}f} |"


def _markdown_report(
    data: ValidatedReportData,
    *,
    baseline_sample: str,
    model_parameter_count: int,
) -> str:
    config = data.config
    summary = data.summary
    first, middle, final = select_representative_samples(data.samples)
    checkpoint_name = data.config.training.checkpoint_dir / "latest.pt"
    result_rows = "\n".join(
        (
            _result_row(
                "train cross-entropy",
                summary.initial_train_loss,
                summary.final_train_loss,
                summary.best_train_loss,
                precision=6,
            ),
            _result_row(
                "validation cross-entropy",
                summary.initial_val_loss,
                summary.final_val_loss,
                summary.best_val_loss,
                precision=6,
            ),
            _result_row(
                "train character perplexity",
                character_perplexity(summary.initial_train_loss),
                character_perplexity(summary.final_train_loss),
                character_perplexity(summary.best_train_loss),
                precision=4,
            ),
            _result_row(
                "validation character perplexity",
                character_perplexity(summary.initial_val_loss),
                character_perplexity(summary.final_val_loss),
                character_perplexity(summary.best_val_loss),
                precision=4,
            ),
        )
    )
    return f"""# CPU Reference Training Evidence

This report is generated from validated raw artifacts. Its purpose is to provide a reproducible,
reviewable training example—not a hardware benchmark.

## Experiment identity

- experiment: `{data.provenance.experiment_name}`
- Git SHA: `{data.provenance.git.commit_sha}`
- clean worktree at training time: `{str(not data.provenance.git.dirty).lower()}`
- checkpoint format: v{data.metadata.format_version}
- checkpoint: `{checkpoint_name.as_posix()}`
- checkpoint SHA-256: `{data.checkpoint_sha256}`
- checkpoint resume: yes ({len(data.provenance.segments)} completed process segments)
- resolved config SHA-256: `{data.provenance.resolved_config_sha256}`

## Model and data

- character-level GPT: {config.model.n_layer} layers, {config.model.n_head} heads,
  embedding width {config.model.n_embd}, block size {config.data.block_size}
- trainable parameters: {model_parameter_count:,}
- vocabulary size: {config.model.vocab_size}
- batch size: {config.data.batch_size}
- Tiny Shakespeare train tokens: {data.train_token_count:,}
- Tiny Shakespeare validation tokens: {data.val_token_count:,}

## Full training commands

```powershell
{_command_lines(data.provenance)}
```

Both processes used the same complete experiment config. The first process ended at its
`--run-until-step` boundary; the second restored checkpoint v2 and completed `max_steps`.

## Results

| metric | initial | final | best |
|---|---:|---:|---:|
{result_rows}

- total optimization steps: {summary.total_steps:,}
- median measured tokens/s: {summary.median_tokens_per_sec:.2f}
- peak process RSS: {summary.peak_cpu_memory_mb:.2f} MiB
- validation values are plotted only at their observed scheduled steps

![Train and validation loss](loss_curve.png)

![Learning rate](learning_rate_curve.png)

![Measured throughput](throughput_curve.png)

## Generated text

Selection rule: untrained baseline plus **first / middle / final scheduled samples**, using the
configured fixed prompt, sample seed, temperature, and top-k. No sample was selected for fluency.

### Untrained baseline (step -1)

```text
{baseline_sample}
```

{_sample_markdown("First scheduled sample", first)}

{_sample_markdown("Middle scheduled sample", middle)}

{_sample_markdown("Final scheduled sample", final)}

{_local_structure_observation(baseline_sample, final.text)}

The complete unedited scheduled sample file is [generated_samples.txt](generated_samples.txt).

## Limitations

This is one small CPU-only character-model training run on Tiny Shakespeare. Lower training loss
does not by itself demonstrate generalization, model quality, statistical significance, or
hardware performance. Reported throughput times batch acquisition, forward/backward, and the
optimizer update; it excludes validation, sampling, checkpoint, logging, and other event overhead.
It therefore must not be compared with a dedicated benchmark.

## Reproduce

1. Install `.[dev,report]` and prepare Tiny Shakespeare with
   `python prepare_data.py --data-dir data`.
2. Run the two commands recorded above from Git SHA `{data.provenance.git.commit_sha}`.
3. Run `python report_training.py` with the config, metrics, samples, checkpoint, provenance, and
   a new output directory.
4. Compare source and generated file hashes in
   [artifact_manifest.json](artifact_manifest.json). The checkpoint is intentionally not committed;
   its SHA-256 above binds the report to the retained local binary.
"""


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        reason = f"artifact path is outside repository: {path}"
        raise TrainingEvidenceError(path, reason) from error


def _file_identity_document(
    path: Path,
    *,
    repository_path: Path,
    repository_root: Path,
) -> dict[str, JsonValue]:
    return {
        "path": _repository_relative(repository_path, repository_root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _manifest_document(
    data: ValidatedReportData,
    inputs: ReportInputs,
    artifact_paths: tuple[Path, ...],
    temporary_output_dir: Path,
    published_output_dir: Path,
) -> dict[str, JsonValue]:
    repository_root = inputs.repository_root
    data_directory = _resolved_data_directory(data.config, repository_root)
    source_paths = {
        "checkpoint": inputs.checkpoint_path,
        "config": inputs.config_path,
        "metrics": inputs.metrics_path,
        "provenance": inputs.provenance_path,
        "samples": inputs.samples_path,
        "tokenizer": data_directory / "tokenizer.json",
        "train_data": data_directory / "train.npy",
        "validation_data": data_directory / "val.npy",
    }
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "generator": {
            "name": "minigpt.training_report",
            "version": _REPORT_SCHEMA_VERSION,
            "git_sha": data.provenance.git.commit_sha,
        },
        "checkpoint_format_version": data.metadata.format_version,
        "final_completed_step": data.metadata.completed_step,
        "resolved_config_sha256": data.provenance.resolved_config_sha256,
        "manifest_self_excluded": True,
        "artifacts": [
            _file_identity_document(
                path,
                repository_path=published_output_dir / path.relative_to(temporary_output_dir),
                repository_root=repository_root,
            )
            for path in sorted(artifact_paths, key=lambda item: item.name)
        ],
        "sources": {
            name: _file_identity_document(
                path,
                repository_path=path,
                repository_root=repository_root,
            )
            for name, path in sorted(source_paths.items())
        },
    }


def _temporary_artifact_paths(directory: Path) -> ReportArtifacts:
    return ReportArtifacts(
        readme=directory / "README.md",
        environment=directory / "environment.json",
        resolved_config=directory / "resolved_config.yaml",
        metrics_csv=directory / "metrics.csv",
        loss_curve=directory / "loss_curve.png",
        learning_rate_curve=directory / "learning_rate_curve.png",
        throughput_curve=directory / "throughput_curve.png",
        generated_samples=directory / "generated_samples.txt",
        manifest=directory / "artifact_manifest.json",
    )


def _generate_into_directory(
    data: ValidatedReportData,
    inputs: ReportInputs,
    directory: Path,
    published_output_dir: Path,
) -> None:
    paths = _temporary_artifact_paths(directory)
    series = build_training_series(data.metrics)
    baseline_sample, parameter_count = _baseline_sample(data)
    _ = paths.resolved_config.write_text(data.config.to_yaml(), encoding="utf-8")
    _write_metrics_csv(paths.metrics_csv, data.metrics)
    _ = shutil.copyfile(inputs.samples_path, paths.generated_samples)
    _write_json(paths.environment, _environment_document(data, parameter_count))
    _ = paths.readme.write_text(
        _markdown_report(
            data,
            baseline_sample=baseline_sample,
            model_parameter_count=parameter_count,
        ),
        encoding="utf-8",
    )
    experiment = data.provenance.experiment_name
    commit_sha = data.provenance.git.commit_sha
    _plot_loss(
        paths.loss_curve,
        series,
        figure_title("Train and validation loss", experiment, commit_sha),
    )
    _plot_single_series(
        paths.learning_rate_curve,
        series.learning_rate,
        title=figure_title("Learning rate", experiment, commit_sha),
        label="learning rate",
        y_axis_label="learning rate",
    )
    _plot_throughput(
        paths.throughput_curve,
        series,
        figure_title("Measured throughput", experiment, commit_sha),
    )
    artifact_paths = (
        paths.readme,
        paths.environment,
        paths.resolved_config,
        paths.metrics_csv,
        paths.loss_curve,
        paths.learning_rate_curve,
        paths.throughput_curve,
        paths.generated_samples,
    )
    _write_json(
        paths.manifest,
        _manifest_document(
            data,
            inputs,
            artifact_paths,
            directory,
            published_output_dir,
        ),
    )


def generate_training_report(inputs: ReportInputs, output_dir: Path) -> ReportArtifacts:
    """Validate sources and atomically generate the complete evidence package."""
    if output_dir.exists():
        raise ReportOutputExistsError(output_dir)
    data = validate_reference_sources(inputs)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}-",
            dir=output_dir.parent,
        )
    )
    try:
        _generate_into_directory(data, inputs, temporary, output_dir)
        _ = temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _temporary_artifact_paths(output_dir)
