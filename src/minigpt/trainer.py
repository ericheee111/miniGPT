"""Orchestrate reproducible training, artifacts, and checkpoint events."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch
from torch.utils.tensorboard import SummaryWriter

from minigpt.checkpoint import load_checkpoint, save_checkpoint
from minigpt.metrics import TrainingMetrics, append_jsonl
from minigpt.training_runtime import (
    InvalidTrainingStateError,
    ScalarWriter,
    TrainingComponents,
    as_scalar_writer,
    build_training_components,
    evaluate,
    run_training_step,
)

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.settings import ExperimentConfig, TrainingSettings

_LOGGER: Final = logging.getLogger(__name__)
_SAMPLE_TEMPERATURE: Final = 0.8
_SAMPLE_TOP_K: Final = 20

__all__ = (
    "InvalidTrainingStateError",
    "TrainingResult",
    "evaluate",
    "run_training",
)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Expose durable artifacts produced by a training run."""

    final_step: int
    metrics_path: Path
    samples_path: Path
    checkpoint_path: Path
    tensorboard_dir: Path


@dataclass(frozen=True, slots=True)
class _OutputPaths:
    metrics: Path
    samples: Path
    checkpoint: Path


def _append_sample(
    path: Path,
    step: int,
    components: TrainingComponents,
    settings: TrainingSettings,
) -> None:
    model = components.model
    tokenizer = components.tokenizer
    was_training = model.training
    _ = model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(settings.sample_prompt)], dtype=torch.long)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=settings.sample_tokens,
        temperature=_SAMPLE_TEMPERATURE,
        top_k=min(_SAMPLE_TOP_K, tokenizer.vocab_size),
        generator=components.sample_generator,
    )
    _ = model.train(was_training)
    token_ids = [int(generated[0, index]) for index in range(generated.shape[1])]
    text = tokenizer.decode(token_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        _ = stream.write(f"step={step}\n{text}\n\n")


def _write_tensorboard(writer: ScalarWriter, metrics: TrainingMetrics) -> None:
    writer.add_scalar("loss/train", metrics.train_loss, metrics.step)
    if metrics.val_loss is not None:
        writer.add_scalar("loss/validation", metrics.val_loss, metrics.step)
    writer.add_scalar("optimization/learning_rate", metrics.learning_rate, metrics.step)
    writer.add_scalar("performance/step_time_ms", metrics.step_time_ms, metrics.step)
    writer.add_scalar("performance/tokens_per_sec", metrics.tokens_per_sec, metrics.step)
    writer.add_scalar("performance/data_time_ms", metrics.data_time_ms, metrics.step)
    writer.add_scalar(
        "performance/forward_backward_time_ms",
        metrics.forward_backward_time_ms,
        metrics.step,
    )
    writer.add_scalar(
        "performance/optimizer_time_ms",
        metrics.optimizer_time_ms,
        metrics.step,
    )
    writer.add_scalar("system/cpu_memory_mb", metrics.cpu_memory_mb, metrics.step)


def _event_due(step: int, interval: int, max_steps: int) -> bool:
    return (step + 1) % interval == 0 or step == max_steps - 1


def _prepare_fresh_outputs(paths: _OutputPaths) -> None:
    for path in (paths.metrics, paths.samples):
        if path.exists():
            path.unlink()


def _record_step_events(
    components: TrainingComponents,
    metrics: TrainingMetrics,
    paths: _OutputPaths,
    writer: ScalarWriter,
) -> None:
    training = components.config.training
    append_jsonl(paths.metrics, metrics)
    _write_tensorboard(writer, metrics)
    if _event_due(metrics.step, training.sample_interval, training.max_steps):
        _append_sample(
            paths.samples,
            metrics.step,
            components,
            training,
        )
    if _event_due(metrics.step, training.checkpoint_interval, training.max_steps):
        save_checkpoint(
            paths.checkpoint,
            resources=components.checkpoint_resources,
            step=metrics.step,
            config=components.config,
        )
        writer.flush()
    if _event_due(metrics.step, training.log_interval, training.max_steps):
        validation_text = "n/a" if metrics.val_loss is None else f"{metrics.val_loss:.4f}"
        _LOGGER.info(
            "step=%d train_loss=%.4f val_loss=%s tokens_per_sec=%.1f",
            metrics.step,
            metrics.train_loss,
            validation_text,
            metrics.tokens_per_sec,
        )


def run_training(
    config: ExperimentConfig,
    *,
    resume_path: Path | None = None,
) -> TrainingResult:
    """Run CPU training, evaluation, logging, sampling, and checkpointing."""
    components = build_training_components(config)
    training = components.config.training
    paths = _OutputPaths(
        metrics=training.output_dir / "metrics.jsonl",
        samples=training.output_dir / "samples.txt",
        checkpoint=training.checkpoint_dir / "latest.pt",
    )
    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(
            resume_path,
            resources=components.checkpoint_resources,
            config=components.config,
        ).next_step
    else:
        _prepare_fresh_outputs(paths)

    training.output_dir.mkdir(parents=True, exist_ok=True)
    training.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = as_scalar_writer(
        SummaryWriter(log_dir=str(training.tensorboard_dir), purge_step=start_step)
    )
    final_step = start_step - 1
    try:
        _ = components.model.train()
        for step in range(start_step, training.max_steps):
            metrics = run_training_step(components, step)
            _record_step_events(components, metrics, paths, writer)
            final_step = step
    finally:
        writer.close()

    return TrainingResult(
        final_step=final_step,
        metrics_path=paths.metrics,
        samples_path=paths.samples,
        checkpoint_path=paths.checkpoint,
        tensorboard_dir=training.tensorboard_dir,
    )
