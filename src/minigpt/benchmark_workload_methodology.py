"""Versioned constants and canonical identity document for Benchmark v2 training work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch

from minigpt.optimization import create_adamw

WorkloadMethodologyValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | list["WorkloadMethodologyValue"]
    | dict[str, "WorkloadMethodologyValue"]
)

WORKLOAD_METHODOLOGY_SCHEMA_VERSION = 1
WORKLOAD_DEVICE: Literal["cpu"] = "cpu"
WORKLOAD_DTYPE = "float32"
MODEL_DROPOUT = 0.0
MODEL_BIAS = False
OPTIMIZER_TYPE: str = "adamw"
OPTIMIZER_LEARNING_RATE = 3e-4
OPTIMIZER_WEIGHT_DECAY = 0.1
OPTIMIZER_BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
ZERO_GRAD_SET_TO_NONE = True
SYNTHETIC_CORPUS_MIN_TOKENS = 4_096
SYNTHETIC_CORPUS_BATCH_MULTIPLIER = 8
SYNTHETIC_TOKEN_DTYPE = "uint16"  # noqa: S105
BATCHER_METHOD = "TokenBatcher.next_batch"


@dataclass(frozen=True, slots=True)
class BenchmarkOptimizerSettings:
    """Hold the optimizer controls actually consumed by benchmark training steps."""

    optimizer_type: str
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float


def benchmark_optimizer_settings() -> BenchmarkOptimizerSettings:
    """Return supported benchmark optimizer settings from the canonical methodology."""
    if OPTIMIZER_TYPE != "adamw":
        msg = f"unsupported benchmark optimizer type {OPTIMIZER_TYPE!r}"
        raise ValueError(msg)
    return BenchmarkOptimizerSettings(
        optimizer_type=OPTIMIZER_TYPE,
        learning_rate=OPTIMIZER_LEARNING_RATE,
        weight_decay=OPTIMIZER_WEIGHT_DECAY,
        beta1=OPTIMIZER_BETAS[0],
        beta2=OPTIMIZER_BETAS[1],
        grad_clip=GRAD_CLIP,
    )


def create_benchmark_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    """Construct the configured benchmark optimizer through its explicit type dispatch."""
    settings = benchmark_optimizer_settings()
    if settings.optimizer_type == "adamw":
        return create_adamw(model, settings)
    msg = f"unsupported benchmark optimizer type {settings.optimizer_type!r}"
    raise ValueError(msg)


def workload_methodology_document() -> dict[str, WorkloadMethodologyValue]:
    """Return every versioned, explicit setting that defines the benchmark training work."""
    return {
        "schema_version": WORKLOAD_METHODOLOGY_SCHEMA_VERSION,
        "device": WORKLOAD_DEVICE,
        "dtype": WORKLOAD_DTYPE,
        "model": {"dropout": MODEL_DROPOUT, "bias": MODEL_BIAS},
        "optimizer": {
            "type": OPTIMIZER_TYPE,
            "learning_rate": OPTIMIZER_LEARNING_RATE,
            "weight_decay": OPTIMIZER_WEIGHT_DECAY,
            "betas": [*OPTIMIZER_BETAS],
        },
        "gradient_clipping": {"max_norm": GRAD_CLIP},
        "zero_grad": {"set_to_none": ZERO_GRAD_SET_TO_NONE},
        "synthetic_corpus": {
            "generator": "numpy.default_rng",
            "minimum_tokens": SYNTHETIC_CORPUS_MIN_TOKENS,
            "batch_size_block_size_multiplier": SYNTHETIC_CORPUS_BATCH_MULTIPLIER,
            "token_dtype": SYNTHETIC_TOKEN_DTYPE,
            "token_range": "[0,vocab_size)",
            "seed": "worker_seed",
        },
        "batcher": {
            "implementation": "TokenBatcher",
            "method": BATCHER_METHOD,
            "seed": "worker_seed",
        },
    }


def synthetic_token_dtype() -> np.dtype[np.uint16]:
    """Resolve the documented synthetic corpus dtype for NumPy generation."""
    try:
        return np.dtype(SYNTHETIC_TOKEN_DTYPE)
    except TypeError as error:
        msg = f"unsupported synthetic token dtype {SYNTHETIC_TOKEN_DTYPE!r}"
        raise ValueError(msg) from error


def workload_torch_dtype() -> torch.dtype:
    """Resolve the documented model dtype without silently falling back."""
    supported_dtypes = {"float32": torch.float32}
    try:
        return supported_dtypes[WORKLOAD_DTYPE]
    except KeyError as error:
        msg = f"unsupported workload dtype {WORKLOAD_DTYPE!r}"
        raise ValueError(msg) from error
