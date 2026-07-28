"""Build isolated synthetic training workloads for CPU performance measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, final

import numpy as np
import torch
from torch.profiler import record_function
from typing_extensions import override

import minigpt.benchmark_workload_methodology as methodology
from minigpt.batching import TokenBatcher
from minigpt.benchmark_types import BenchmarkCase
from minigpt.model import GPT
from minigpt.optimization import seed_everything
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from minigpt.benchmark_v2_types import BenchmarkV2Case

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
        corpus_size = max(
            methodology.SYNTHETIC_CORPUS_MIN_TOKENS,
            case.batch_size * (case.block_size + 1) * methodology.SYNTHETIC_CORPUS_BATCH_MULTIPLIER,
        )
        if methodology.BATCHER_METHOD != "TokenBatcher.next_batch":
            msg = f"unsupported synthetic batcher method {methodology.BATCHER_METHOD!r}"
            raise ValueError(msg)
        tokens = np.random.default_rng(seed).integers(
            0,
            vocab_size,
            size=corpus_size,
            dtype=methodology.synthetic_token_dtype(),
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
                dropout=methodology.MODEL_DROPOUT,
                bias=methodology.MODEL_BIAS,
            )
        )
        _ = self.model.to(
            device=methodology.WORKLOAD_DEVICE, dtype=methodology.workload_torch_dtype()
        )
        settings = methodology.benchmark_optimizer_settings()
        self.optimizer = methodology.create_benchmark_optimizer(self.model)
        self.grad_clip = settings.grad_clip
        self.tokens_per_step = case.batch_size * case.block_size

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable and non-trainable model parameters."""
        return self.model.parameter_count()

    def step(self) -> None:
        """Execute one uninstrumented forward/backward/optimizer training step."""
        inputs, targets = self.batcher.next_batch()
        self.optimizer.zero_grad(set_to_none=methodology.ZERO_GRAD_SET_TO_NONE)
        _, loss = cast("tuple[torch.Tensor, torch.Tensor | None]", self.model(inputs, targets))
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
            self.optimizer.zero_grad(set_to_none=methodology.ZERO_GRAD_SET_TO_NONE)
            _, loss = cast("tuple[torch.Tensor, torch.Tensor | None]", self.model(inputs, targets))
            if loss is None:
                raise InvalidBenchmarkStateError(MISSING_LOSS_REASON)
            backward = cast("Callable[[], object | None]", loss.backward)
            _ = backward()
            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        with record_function("optimizer_step"):
            optimizer_step = cast("Callable[[], object | None]", self.optimizer.step)
            _ = optimizer_step()


def create_training_step_workload(
    case: BenchmarkV2Case,
    *,
    seed: int,
    vocab_size: int,
) -> TrainingStepWorkload:
    """Adapt one explicit Benchmark v2 case to the unchanged v1 workload math."""
    return TrainingStepWorkload(
        BenchmarkCase(
            model_size=case.model_name,
            n_layer=case.n_layer,
            n_head=case.n_head,
            n_embd=case.n_embd,
            thread_count=case.torch_num_threads,
            block_size=case.block_size,
            batch_size=case.batch_size,
        ),
        seed=seed,
        vocab_size=vocab_size,
    )
