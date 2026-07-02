"""Build isolated synthetic training workloads for CPU performance measurement."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, final, override

import numpy as np
import torch
from torch.profiler import record_function

from minigpt.batching import TokenBatcher
from minigpt.model import GPT
from minigpt.optimization import create_adamw, seed_everything
from minigpt.settings import GPTConfig, OptimizerSettings

if TYPE_CHECKING:
    from collections.abc import Callable

    from minigpt.benchmark_types import BenchmarkCase

MISSING_LOSS_REASON = "model returned no loss"


@dataclass(frozen=True, slots=True)
class InvalidBenchmarkStateError(RuntimeError):
    """Report a training-step contract violation during measurement."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the training-step contract violation."""
        return f"invalid benchmark state: {self.reason}"


@final
class TrainingStepWorkload:
    """Own mutable model, optimizer, and sampler state for one benchmark case."""

    def __init__(self, case: BenchmarkCase, *, seed: int, vocab_size: int) -> None:
        """Initialize a deterministic model, optimizer, and synthetic corpus."""
        seed_everything(seed, case.thread_count)
        corpus_size = max(4_096, case.batch_size * (case.block_size + 1) * 8)
        tokens = np.random.default_rng(seed).integers(
            0,
            vocab_size,
            size=corpus_size,
            dtype=np.uint16,
        )
        self.batcher = TokenBatcher(
            tokens,
            batch_size=case.batch_size,
            block_size=case.block_size,
            seed=seed,
        )
        self.model = GPT(
            GPTConfig(
                vocab_size=vocab_size,
                block_size=case.block_size,
                n_layer=case.n_layer,
                n_head=case.n_head,
                n_embd=case.n_embd,
                dropout=0.0,
                bias=False,
            )
        )
        settings = OptimizerSettings(
            learning_rate=3e-4,
            min_learning_rate=3e-5,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            grad_clip=1.0,
        )
        self.optimizer = create_adamw(self.model, settings)
        self.grad_clip = settings.grad_clip
        self.tokens_per_step = case.batch_size * case.block_size

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable and non-trainable model parameters."""
        return self.model.parameter_count()

    def step(self) -> None:
        """Execute one uninstrumented forward/backward/optimizer training step."""
        inputs, targets = self.batcher.next_batch()
        self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.model.forward(inputs, targets)
        if loss is None:
            raise InvalidBenchmarkStateError(MISSING_LOSS_REASON)
        backward = cast("Callable[[], object | None]", loss.backward)
        _ = backward()
        _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        optimizer_step = cast("Callable[[], object | None]", self.optimizer.step)
        _ = optimizer_step()

    def profiled_step(self) -> None:
        """Execute one training step with high-level profiler scopes."""
        with record_function("data_preparation"):
            inputs, targets = self.batcher.next_batch()
        with record_function("forward_backward"):
            self.optimizer.zero_grad(set_to_none=True)
            _, loss = self.model.forward(inputs, targets)
            if loss is None:
                raise InvalidBenchmarkStateError(MISSING_LOSS_REASON)
            backward = cast("Callable[[], object | None]", loss.backward)
            _ = backward()
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        with record_function("optimizer_step"):
            optimizer_step = cast("Callable[[], object | None]", self.optimizer.step)
            _ = optimizer_step()
