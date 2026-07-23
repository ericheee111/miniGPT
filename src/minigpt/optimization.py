"""Provide reproducible seeding, scheduling, and optimizer construction."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from minigpt.settings import OptimizerSettings

_DECAY_DIMENSION_THRESHOLD: Final = 2


def seed_everything(seed: int, num_threads: int) -> None:
    """Seed Python, NumPy, and PyTorch and configure CPU parallelism."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    _ = torch.default_generator.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(mode=True)


def learning_rate_at_step(
    step: int,
    *,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Return linear warmup followed by cosine decay to the final step."""
    if warmup_steps > 0 and step < warmup_steps:
        return max_learning_rate * (step + 1) / warmup_steps
    decay_step_count = max_steps - warmup_steps
    if decay_step_count <= 1:
        return min_learning_rate
    progress = (step - warmup_steps) / (decay_step_count - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)


def create_adamw(model: nn.Module, settings: OptimizerSettings) -> torch.optim.AdamW:
    """Create AdamW with decay limited to matrix-like parameters."""
    decayed_parameters: list[nn.Parameter] = []
    non_decayed_parameters: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        destination = (
            decayed_parameters
            if parameter.ndim >= _DECAY_DIMENSION_THRESHOLD
            else non_decayed_parameters
        )
        destination.append(parameter)
    parameter_groups = [
        {"params": decayed_parameters, "weight_decay": settings.weight_decay},
        {"params": non_decayed_parameters, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Apply one scheduled learning rate to every optimizer parameter group."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
