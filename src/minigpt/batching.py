import json
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class InvalidBatchConfigurationError(ValueError):
    """Report batch dimensions or token data that cannot form a batch."""

    reason: str

    def __str__(self) -> str:
        return f"invalid token batch configuration: {self.reason}"


class TokenBatcher:
    """Sample mutable RNG-driven next-token batches from a one-dimensional corpus."""

    __slots__ = ("_batch_size", "_block_size", "_device", "_rng", "_tokens")

    def __init__(
        self,
        tokens: npt.ArrayLike,
        *,
        batch_size: int,
        block_size: int,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        token_array = np.asarray(tokens, dtype=np.int64)
        if token_array.ndim != 1:
            raise InvalidBatchConfigurationError("tokens must be one-dimensional")
        if batch_size <= 0:
            raise InvalidBatchConfigurationError("batch_size must be positive")
        if block_size <= 0:
            raise InvalidBatchConfigurationError("block_size must be positive")
        if token_array.size <= block_size:
            raise InvalidBatchConfigurationError(
                f"need more than block_size={block_size} tokens, received {token_array.size}"
            )
        self._tokens = token_array
        self._batch_size = batch_size
        self._block_size = block_size
        self._rng = np.random.default_rng(seed)
        self._device = torch.device(device)

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Return input tokens and their one-position-right training targets."""
        start_limit = self._tokens.size - self._block_size
        starts = self._rng.integers(0, start_limit, size=self._batch_size)
        windows = np.stack(
            [
                self._tokens[int(start) : int(start) + self._block_size + 1]
                for start in starts
            ]
        )
        x = torch.from_numpy(windows[:, :-1].copy()).to(self._device)
        y = torch.from_numpy(windows[:, 1:].copy()).to(self._device)
        return x, y

    def capture_random_state(self) -> str:
        """Serialize the independent sampler RNG for checkpointing."""
        return json.dumps(self._rng.bit_generator.state)

    def restore_random_state(self, state_json: str) -> None:
        """Restore the independent sampler RNG from a checkpoint."""
        self._rng.bit_generator.state = json.loads(state_json)
