"""Build typed training components and execute one measured optimization step."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Final, Protocol, cast, final

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from typing_extensions import override

from minigpt.batching import TokenBatcher
from minigpt.checkpoint import (
    CheckpointResources,
    DatasetFingerprints,
    compute_dataset_fingerprints,
)
from minigpt.data import CharTokenizer
from minigpt.metrics import TrainingMetrics, cpu_memory_mb
from minigpt.model import GPT
from minigpt.optimization import (
    create_adamw,
    create_sample_generator,
    learning_rate_at_step,
    seed_everything,
    set_learning_rate,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.tensorboard import SummaryWriter

    from minigpt.settings import ExperimentConfig

_VALIDATION_LOSS_REASON: Final = "model returned no validation loss"
_TRAINING_LOSS_REASON: Final = "model returned no training loss"
_MILLISECONDS_PER_SECOND: Final = 1_000.0


@dataclass(frozen=True, slots=True)
class InvalidTrainingStateError(RuntimeError):
    """Report an internal model/trainer contract violation."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the violated trainer contract."""
        return f"invalid training state: {self.reason}"


@final
class ScalarWriter:
    """Adapt TensorBoard's weak annotations to the used scalar surface."""

    __slots__ = ("_writer",)

    def __init__(self, writer: SummaryWriter) -> None:
        """Wrap one TensorBoard event writer."""
        self._writer = writer

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """Record one scalar at a training step."""
        add_scalar = cast(
            "Callable[[str, float, int], None]",
            self._writer.add_scalar,
        )
        add_scalar(tag, value, step)

    def flush(self) -> None:
        """Flush pending event records."""
        self._writer.flush()

    def close(self) -> None:
        """Close the event writer."""
        self._writer.close()


class _StepOptimizer(Protocol):
    def step(self) -> None: ...


class _BackwardTensor(Protocol):
    def backward(self) -> None: ...


def as_scalar_writer(value: SummaryWriter) -> ScalarWriter:
    """Narrow TensorBoard's weak third-party annotations to the used surface."""
    return ScalarWriter(value)


def _step_optimizer(value: torch.optim.Optimizer) -> None:
    optimizer = cast("_StepOptimizer", value)
    optimizer.step()


def _backward(value: Tensor) -> None:
    tensor = cast("_BackwardTensor", value)
    tensor.backward()


def tensor_scalar(value: Tensor) -> float:
    """Convert a scalar tensor to a concrete Python float."""
    scalar = cast("float | int", value.detach().item())
    return float(scalar)


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    """Hold the resolved objects shared by all training steps."""

    config: ExperimentConfig
    tokenizer: CharTokenizer
    model: GPT
    optimizer: torch.optim.AdamW
    train_batcher: TokenBatcher
    val_batcher: TokenBatcher
    sample_generator: torch.Generator
    dataset_fingerprints: DatasetFingerprints
    checkpoint_resources: CheckpointResources


@torch.no_grad()
def evaluate(model: GPT, batcher: TokenBatcher, batch_count: int) -> float:
    """Return mean validation loss while preserving the prior model mode."""
    was_training = model.training
    _ = model.eval()
    losses: list[float] = []
    for _ in range(batch_count):
        inputs, targets = batcher.next_batch()
        _, loss = cast("tuple[Tensor, Tensor | None]", model(inputs, targets))
        if loss is None:
            raise InvalidTrainingStateError(_VALIDATION_LOSS_REASON)
        losses.append(tensor_scalar(loss))
    _ = model.train(was_training)
    return sum(losses) / len(losses)


def build_training_components(config: ExperimentConfig) -> TrainingComponents:
    """Resolve vocabulary, load token arrays, and construct trainable objects."""
    seed_everything(config.runtime.seed, config.runtime.num_threads)
    tokenizer = CharTokenizer.load(config.data.tokenizer_path)
    resolved = config.resolve_vocab_size(tokenizer.vocab_size)
    train_tokens = cast(
        "npt.NDArray[np.uint16]",
        np.load(resolved.data.train_path, mmap_mode="r"),
    )
    val_tokens = cast(
        "npt.NDArray[np.uint16]",
        np.load(resolved.data.val_path, mmap_mode="r"),
    )
    train_batcher = TokenBatcher(
        train_tokens,
        batch_size=resolved.data.batch_size,
        block_size=resolved.data.block_size,
        seed=resolved.runtime.seed,
    )
    val_batcher = TokenBatcher(
        val_tokens,
        batch_size=resolved.data.batch_size,
        block_size=resolved.data.block_size,
        seed=resolved.runtime.seed + 1,
    )
    model = GPT(resolved.model.to_gpt_config(resolved.data.block_size))
    optimizer = create_adamw(model, resolved.optimizer)
    sample_generator = create_sample_generator(resolved.runtime.seed)
    dataset_fingerprints = compute_dataset_fingerprints(resolved.data)
    resources = CheckpointResources(
        model=model,
        optimizer=optimizer,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        sample_generator=sample_generator,
        dataset_fingerprints=dataset_fingerprints,
    )
    return TrainingComponents(
        config=resolved,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        sample_generator=sample_generator,
        dataset_fingerprints=dataset_fingerprints,
        checkpoint_resources=resources,
    )


def run_training_step(components: TrainingComponents, step: int) -> TrainingMetrics:
    """Run one timed optimization step and any due validation."""
    config = components.config
    training = config.training
    learning_rate = learning_rate_at_step(
        step,
        max_learning_rate=config.optimizer.learning_rate,
        min_learning_rate=config.optimizer.min_learning_rate,
        warmup_steps=training.warmup_steps,
        lr_decay_steps=training.lr_decay_steps,
    )
    set_learning_rate(components.optimizer, learning_rate)
    step_started = perf_counter()
    data_started = perf_counter()
    inputs, targets = components.train_batcher.next_batch()
    data_time_ms = (perf_counter() - data_started) * _MILLISECONDS_PER_SECOND

    forward_backward_started = perf_counter()
    components.optimizer.zero_grad(set_to_none=True)
    _, loss = cast(
        "tuple[Tensor, Tensor | None]",
        components.model(inputs, targets),
    )
    if loss is None:
        raise InvalidTrainingStateError(_TRAINING_LOSS_REASON)
    _backward(loss)
    _ = torch.nn.utils.clip_grad_norm_(
        components.model.parameters(),
        config.optimizer.grad_clip,
    )
    forward_backward_time_ms = (
        perf_counter() - forward_backward_started
    ) * _MILLISECONDS_PER_SECOND

    optimizer_started = perf_counter()
    _step_optimizer(components.optimizer)
    optimizer_time_ms = (perf_counter() - optimizer_started) * _MILLISECONDS_PER_SECOND
    step_time_ms = (perf_counter() - step_started) * _MILLISECONDS_PER_SECOND
    token_count = config.data.batch_size * config.data.block_size
    tokens_per_sec = token_count / (step_time_ms / _MILLISECONDS_PER_SECOND)
    val_loss = None
    if (step + 1) % training.eval_interval == 0 or step == training.max_steps - 1:
        val_loss = evaluate(components.model, components.val_batcher, training.eval_batches)
    return TrainingMetrics(
        step=step,
        train_loss=tensor_scalar(loss),
        val_loss=val_loss,
        learning_rate=learning_rate,
        step_time_ms=step_time_ms,
        tokens_per_sec=tokens_per_sec,
        data_time_ms=data_time_ms,
        forward_backward_time_ms=forward_backward_time_ms,
        optimizer_time_ms=optimizer_time_ms,
        cpu_memory_mb=cpu_memory_mb(),
    )
