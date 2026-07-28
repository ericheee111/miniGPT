"""Test that Benchmark v2 methodology defines both identity and executed work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
import torch

import minigpt.benchmark_workload_methodology as methodology_module
from minigpt.batching import TokenArrayLike, TokenBatcher
from minigpt.benchmark_v2_config import case_identity, load_benchmark_v2_config
from minigpt.benchmark_workload import create_training_step_workload

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _IntegerCall:
    """Capture one synthetic-token generator request at the workload boundary."""

    low: int
    high: int
    size: int
    dtype: np.dtype[np.generic]


class _SyntheticGeneratorSpy:
    """Provide deterministic synthetic tokens while recording their requested bounds."""

    def __init__(self) -> None:
        """Initialize an empty generator-call log."""
        self.calls: list[_IntegerCall] = []

    def integers(
        self,
        low: int,
        high: int,
        *,
        size: int,
        dtype: np.dtype[np.generic],
    ) -> npt.NDArray[np.generic]:
        """Record one bounded token generation call and return in-range deterministic values."""
        self.calls.append(_IntegerCall(low=low, high=high, size=size, dtype=dtype))
        return np.arange(size, dtype=dtype) % high


@dataclass(frozen=True, slots=True)
class _BatcherConstruction:
    """Capture the synthetic corpus and sampler controls sent to TokenBatcher."""

    tokens: npt.NDArray[np.generic]
    batch_size: int
    block_size: int
    seed: int | None


def _write_workload_config(tmp_path: Path) -> Path:
    """Write one compact, valid Benchmark v2 configuration for workload construction."""
    config_path = tmp_path / "benchmark.yaml"
    _ = config_path.write_text(
        """
schema_version: 2
experiment_name: methodology_test
benchmark_seed: 1
vocab_size: 17
output_root: reports
worker_timeout_seconds: 1.0
warmup_steps: 0
measurement_steps: 1
replicates: 1
torch_num_interop_threads: 1
cpu_affinity: null
max_cv_percent: 10.0
minimum_replicates: 1
regression_threshold_percent: 5.0
relevant_environment_variables: []
models:
  tiny:
    n_layer: 1
    n_head: 1
    n_embd: 8
cases:
  - name: tiny
    model_name: tiny
    torch_num_threads: 1
    block_size: 4
    batch_size: 1
profile:
  enabled: false
  case_name: tiny
  warmup_steps: 1
  active_steps: 1
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def test_optimizer_methodology_change_affects_identity_and_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A benchmark optimizer change updates the identity and constructed AdamW settings."""
    # Given: a compact real benchmark case and its original workload identity.
    config_path = tmp_path / "benchmark.yaml"
    _ = config_path.write_text(
        """
schema_version: 2
experiment_name: methodology_test
benchmark_seed: 1
vocab_size: 17
output_root: reports
worker_timeout_seconds: 1.0
warmup_steps: 0
measurement_steps: 1
replicates: 1
torch_num_interop_threads: 1
cpu_affinity: null
max_cv_percent: 10.0
minimum_replicates: 1
regression_threshold_percent: 5.0
relevant_environment_variables: []
models:
  tiny:
    n_layer: 1
    n_head: 1
    n_embd: 8
cases:
  - name: tiny
    model_name: tiny
    torch_num_threads: 1
    block_size: 4
    batch_size: 1
profile:
  enabled: false
  case_name: tiny
  warmup_steps: 1
  active_steps: 1
""".lstrip(),
        encoding="utf-8",
    )
    config = load_benchmark_v2_config(config_path)
    before = case_identity(config, config.cases[0])

    # When: the shared methodology changes the actual AdamW learning rate.
    monkeypatch.setattr(methodology_module, "OPTIMIZER_LEARNING_RATE", 0.0125)
    workload = create_training_step_workload(config.cases[0], seed=1, vocab_size=config.vocab_size)

    # Then: both the identity and optimizer behavior bind that one methodology value.
    assert case_identity(config, config.cases[0]) != before
    assert {group["lr"] for group in workload.optimizer.param_groups} == {0.0125}


def test_unsupported_benchmark_optimizer_type_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported methodology optimizer cannot silently construct AdamW."""
    # Given: a requested optimizer with no benchmark implementation.
    monkeypatch.setattr(methodology_module, "OPTIMIZER_TYPE", "sgd")

    # When/Then: the dispatch rejects it before any benchmark measurement occurs.
    with pytest.raises(ValueError, match="unsupported benchmark optimizer type"):
        _ = methodology_module.create_benchmark_optimizer(torch.nn.Linear(1, 1))


def test_workload_uses_shared_synthetic_and_batcher_methodology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic corpus and TokenBatcher construction consume the shared methodology values."""
    # Given: a compact case and typed construction-boundary spies.
    config = load_benchmark_v2_config(_write_workload_config(tmp_path))
    generator = _SyntheticGeneratorSpy()
    generator_seeds: list[int | None] = []
    batcher_constructions: list[_BatcherConstruction] = []

    def create_synthetic_generator(seed: int | None = None) -> _SyntheticGeneratorSpy:
        """Record the synthetic-corpus seed and supply deterministic tokens."""
        generator_seeds.append(seed)
        return generator

    def create_batcher(
        tokens: TokenArrayLike,
        *,
        batch_size: int,
        block_size: int,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> TokenBatcher:
        """Record TokenBatcher inputs while retaining its real implementation."""
        token_array = np.asarray(tokens)
        batcher_constructions.append(
            _BatcherConstruction(
                tokens=token_array,
                batch_size=batch_size,
                block_size=block_size,
                seed=seed,
            )
        )
        return TokenBatcher(
            token_array,
            batch_size=batch_size,
            block_size=block_size,
            seed=seed,
            device=device,
        )

    monkeypatch.setattr(np.random, "default_rng", create_synthetic_generator)
    monkeypatch.setattr(methodology_module, "TokenBatcher", create_batcher)
    monkeypatch.setattr(methodology_module, "SYNTHETIC_CORPUS_MIN_TOKENS", 1)
    monkeypatch.setattr(methodology_module, "SYNTHETIC_CORPUS_BATCH_MULTIPLIER", 9)
    monkeypatch.setattr(methodology_module, "SYNTHETIC_TOKEN_DTYPE", "uint32")

    # When: the benchmark workload builds its deterministic synthetic training inputs.
    _ = create_training_step_workload(config.cases[0], seed=7, vocab_size=config.vocab_size)

    # Then: the documented generator, range, seed, dtype, and batcher dimensions reach construction.
    assert generator_seeds == [7, 7]
    assert generator.calls == [_IntegerCall(low=0, high=17, size=45, dtype=np.dtype("uint32"))]
    assert len(batcher_constructions) == 1
    construction = batcher_constructions[0]
    assert construction.tokens.dtype == np.dtype("uint32")
    assert construction.batch_size == 1
    assert construction.block_size == 4
    assert construction.seed == 7


def test_synthetic_generator_methodology_binds_identity_and_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generator declaration cannot drift from the identity without blocking execution."""
    # Given: the canonical case identity before one synthetic-generator methodology change.
    config = load_benchmark_v2_config(_write_workload_config(tmp_path))
    before = case_identity(config, config.cases[0])

    # When: the documented generator is replaced by an unsupported implementation name.
    monkeypatch.setattr(methodology_module, "SYNTHETIC_CORPUS_GENERATOR", "custom.rng")

    # Then: both the case identity and the executed workload reject the changed methodology.
    assert case_identity(config, config.cases[0]) != before
    with pytest.raises(ValueError, match="unsupported synthetic corpus generator"):
        _ = create_training_step_workload(config.cases[0], seed=1, vocab_size=config.vocab_size)


def test_workload_methodology_controls_model_optimizer_and_training_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructed model and optimizer consume every non-corpus methodology control."""
    # Given: one compact CPU workload and changed supported model and optimizer controls.
    config = load_benchmark_v2_config(_write_workload_config(tmp_path))
    before = case_identity(config, config.cases[0])
    monkeypatch.setattr(methodology_module, "MODEL_DROPOUT", 0.25)
    monkeypatch.setattr(methodology_module, "MODEL_BIAS", True)
    monkeypatch.setattr(methodology_module, "OPTIMIZER_WEIGHT_DECAY", 0.2)
    monkeypatch.setattr(methodology_module, "OPTIMIZER_BETAS", (0.8, 0.9))
    monkeypatch.setattr(methodology_module, "GRAD_CLIP", 0.5)
    monkeypatch.setattr(methodology_module, "ZERO_GRAD_SET_TO_NONE", False)
    zero_grad_calls: list[bool] = []
    clip_norms: list[float] = []

    def zero_grad(*, set_to_none: bool = True) -> None:
        """Capture the zero-grad control at the optimizer boundary."""
        zero_grad_calls.append(set_to_none)

    def clip_grad_norm(parameters: object, max_norm: float) -> torch.Tensor:
        """Capture gradient clipping without replacing the real model forward/backward path."""
        _ = parameters
        clip_norms.append(max_norm)
        return torch.tensor(0.0)

    # When: one construction and one real forward/backward/optimizer step execute.
    workload = create_training_step_workload(config.cases[0], seed=1, vocab_size=config.vocab_size)
    monkeypatch.setattr(workload.optimizer, "zero_grad", zero_grad)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_grad_norm)
    workload.step()

    # Then: identity, model construction, AdamW, CPU float32 placement, and step controls agree.
    assert case_identity(config, config.cases[0]) != before
    assert workload.model.config.dropout == 0.25
    assert workload.model.config.bias is True
    assert {parameter.device.type for parameter in workload.model.parameters()} == {"cpu"}
    assert {parameter.dtype for parameter in workload.model.parameters()} == {torch.float32}
    assert {group["weight_decay"] for group in workload.optimizer.param_groups} == {0.0, 0.2}
    assert workload.optimizer.defaults["betas"] == (0.8, 0.9)
    assert zero_grad_calls == [False]
    assert clip_norms == [0.5]


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        ("WORKLOAD_DEVICE", "accelerator", "unsupported workload device"),
        ("WORKLOAD_DTYPE", "float64", "unsupported workload dtype"),
        ("SYNTHETIC_CORPUS_SEED_SOURCE", "config_seed", "synthetic corpus seed source"),
        ("SYNTHETIC_TOKEN_RANGE", "[1,vocab_size)", "synthetic token range"),
        ("BATCHER_IMPLEMENTATION", "OtherBatcher", "benchmark batcher implementation"),
        ("BATCHER_METHOD", "TokenBatcher.sample", "synthetic batcher method"),
        ("BATCHER_SEED_SOURCE", "config_seed", "benchmark batcher seed source"),
    ],
)
def test_workload_rejects_any_unsupported_shared_methodology_declaration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    reason: str,
) -> None:
    """Each declared CPU workload control changes identity and is consumed before measurement."""
    # Given: one canonical workload identity.
    config = load_benchmark_v2_config(_write_workload_config(tmp_path))
    before = case_identity(config, config.cases[0])

    # When: one shared methodology declaration requests an unsupported implementation.
    monkeypatch.setattr(methodology_module, field, replacement)

    # Then: it cannot silently leave execution at the old hard-coded behavior.
    assert case_identity(config, config.cases[0]) != before
    with pytest.raises(ValueError, match=reason):
        _ = create_training_step_workload(config.cases[0], seed=1, vocab_size=config.vocab_size)
