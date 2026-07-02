from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from minigpt.batching import TokenBatcher
from minigpt.checkpoint import load_checkpoint, save_checkpoint
from minigpt.config import ExperimentConfig
from minigpt.data import CharTokenizer
from minigpt.metrics import TrainingMetrics, append_jsonl, cpu_memory_mb
from minigpt.model import GPT
from minigpt.optimization import (
    create_adamw,
    learning_rate_at_step,
    seed_everything,
    set_learning_rate,
)


@dataclass(frozen=True, slots=True)
class InvalidTrainingStateError(RuntimeError):
    """Report an internal model/trainer contract violation."""

    reason: str

    def __str__(self) -> str:
        return f"invalid training state: {self.reason}"


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Expose durable artifacts produced by a training run."""

    final_step: int
    metrics_path: Path
    samples_path: Path
    checkpoint_path: Path
    tensorboard_dir: Path


@torch.no_grad()
def evaluate(model: GPT, batcher: TokenBatcher, batch_count: int) -> float:
    """Return mean validation loss while preserving the prior model mode."""
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(batch_count):
        inputs, targets = batcher.next_batch()
        _, loss = model(inputs, targets)
        if loss is None:
            raise InvalidTrainingStateError("model returned no validation loss")
        losses.append(float(loss.item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def _append_sample(
    path: Path,
    *,
    step: int,
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    token_count: int,
) -> None:
    was_training = model.training
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=token_count,
        temperature=0.8,
        top_k=min(20, tokenizer.vocab_size),
    )
    model.train(was_training)
    text = tokenizer.decode(generated[0].tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"step={step}\n{text}\n\n")


def _write_tensorboard(
    writer: SummaryWriter,
    metrics: TrainingMetrics,
) -> None:
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


def _prepare_fresh_outputs(metrics_path: Path, samples_path: Path) -> None:
    for path in (metrics_path, samples_path):
        if path.exists():
            path.unlink()


def run_training(
    config: ExperimentConfig,
    *,
    resume_path: Path | None = None,
) -> TrainingResult:
    """Run CPU training, evaluation, logging, sampling, and checkpointing."""
    seed_everything(config.runtime.seed, config.runtime.num_threads)
    tokenizer = CharTokenizer.load(config.data.tokenizer_path)
    resolved_config = config.resolve_vocab_size(tokenizer.vocab_size)
    train_tokens = np.load(resolved_config.data.train_path, mmap_mode="r")
    val_tokens = np.load(resolved_config.data.val_path, mmap_mode="r")
    train_batcher = TokenBatcher(
        train_tokens,
        batch_size=resolved_config.data.batch_size,
        block_size=resolved_config.data.block_size,
        seed=resolved_config.runtime.seed,
    )
    val_batcher = TokenBatcher(
        val_tokens,
        batch_size=resolved_config.data.batch_size,
        block_size=resolved_config.data.block_size,
        seed=resolved_config.runtime.seed + 1,
    )
    model = GPT(resolved_config.model.to_gpt_config(resolved_config.data.block_size))
    optimizer = create_adamw(model, resolved_config.optimizer)

    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            train_batcher=train_batcher,
            val_batcher=val_batcher,
        ).next_step

    training = resolved_config.training
    metrics_path = training.output_dir / "metrics.jsonl"
    samples_path = training.output_dir / "samples.txt"
    checkpoint_path = training.checkpoint_dir / "latest.pt"
    if resume_path is None:
        _prepare_fresh_outputs(metrics_path, samples_path)
    training.output_dir.mkdir(parents=True, exist_ok=True)
    training.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir=str(training.tensorboard_dir),
        purge_step=start_step,
    )

    final_step = start_step - 1
    try:
        model.train()
        for step in range(start_step, training.max_steps):
            learning_rate = learning_rate_at_step(
                step,
                max_learning_rate=resolved_config.optimizer.learning_rate,
                min_learning_rate=resolved_config.optimizer.min_learning_rate,
                warmup_steps=training.warmup_steps,
                max_steps=training.max_steps,
            )
            set_learning_rate(optimizer, learning_rate)
            step_started = perf_counter()
            data_started = perf_counter()
            inputs, targets = train_batcher.next_batch()
            data_time_ms = (perf_counter() - data_started) * 1_000

            forward_backward_started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            if loss is None:
                raise InvalidTrainingStateError("model returned no training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                resolved_config.optimizer.grad_clip,
            )
            forward_backward_time_ms = (perf_counter() - forward_backward_started) * 1_000

            optimizer_started = perf_counter()
            optimizer.step()
            optimizer_time_ms = (perf_counter() - optimizer_started) * 1_000
            step_time_ms = (perf_counter() - step_started) * 1_000
            token_count = resolved_config.data.batch_size * resolved_config.data.block_size
            tokens_per_sec = token_count / (step_time_ms / 1_000)
            val_loss = None
            if _event_due(step, training.eval_interval, training.max_steps):
                val_loss = evaluate(model, val_batcher, training.eval_batches)

            record = TrainingMetrics(
                step=step,
                train_loss=float(loss.item()),
                val_loss=val_loss,
                learning_rate=learning_rate,
                step_time_ms=step_time_ms,
                tokens_per_sec=tokens_per_sec,
                data_time_ms=data_time_ms,
                forward_backward_time_ms=forward_backward_time_ms,
                optimizer_time_ms=optimizer_time_ms,
                cpu_memory_mb=cpu_memory_mb(),
            )
            append_jsonl(metrics_path, record)
            _write_tensorboard(writer, record)

            if _event_due(step, training.sample_interval, training.max_steps):
                _append_sample(
                    samples_path,
                    step=step,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=training.sample_prompt,
                    token_count=training.sample_tokens,
                )
            if _event_due(step, training.checkpoint_interval, training.max_steps):
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=resolved_config,
                    train_batcher=train_batcher,
                    val_batcher=val_batcher,
                )
                writer.flush()
            if _event_due(step, training.log_interval, training.max_steps):
                validation_text = "n/a" if val_loss is None else f"{val_loss:.4f}"
                print(
                    f"step={step} train_loss={loss.item():.4f} "
                    f"val_loss={validation_text} tokens_per_sec={tokens_per_sec:.1f}"
                )
            final_step = step
    finally:
        writer.close()

    return TrainingResult(
        final_step=final_step,
        metrics_path=metrics_path,
        samples_path=samples_path,
        checkpoint_path=checkpoint_path,
        tensorboard_dir=training.tensorboard_dir,
    )
