"""Build reproducible autoregressive token batches."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, TypeAlias, cast, final

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable

_ONE_DIMENSIONAL_REASON: Final = "tokens must be one-dimensional"
_POSITIVE_BATCH_REASON: Final = "batch_size must be positive"
_POSITIVE_BLOCK_REASON: Final = "block_size must be positive"

TokenArray: TypeAlias = npt.NDArray[np.uint16] | npt.NDArray[np.int64]
TokenArrayLike: TypeAlias = TokenArray | Sequence[int]
_FROM_NUMPY = cast(
    "Callable[[npt.NDArray[np.int64]], Tensor]",
    cast("object", torch.from_numpy),
)
_SLIDING_WINDOW_VIEW = cast(
    "Callable[[TokenArray, int], TokenArray]",
    cast("object", np.lib.stride_tricks.sliding_window_view),
)


@dataclass(frozen=True, slots=True)
class InvalidBatchConfigurationError(ValueError):
    """Report batch dimensions or token data that cannot form a batch."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed batch constraint."""
        return f"invalid token batch configuration: {self.reason}"


@final
class TokenBatcher:
    """Sample mutable RNG-driven next-token batches from a one-dimensional corpus."""

    __slots__ = (
        "_batch_size",
        "_block_size",
        "_device",
        "_rng",
        "_token_windows",
        "_tokens",
    )

    def __init__(
        self,
        tokens: TokenArrayLike,
        *,
        batch_size: int,
        block_size: int,
        seed: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Validate a corpus and initialize its independent sampler."""
        token_array = cast(
            "TokenArray",
            np.asarray(tokens),
        )
        if token_array.ndim != 1:
            raise InvalidBatchConfigurationError(_ONE_DIMENSIONAL_REASON)
        if batch_size <= 0:
            raise InvalidBatchConfigurationError(_POSITIVE_BATCH_REASON)
        if block_size <= 0:
            raise InvalidBatchConfigurationError(_POSITIVE_BLOCK_REASON)
        if token_array.size <= block_size:
            reason = f"need more than block_size={block_size} tokens, received {token_array.size}"
            raise InvalidBatchConfigurationError(reason)
        self._tokens = token_array
        self._batch_size = batch_size
        self._block_size = block_size
        self._rng = np.random.default_rng(seed)
        self._device = torch.device(device)
        self._token_windows = _SLIDING_WINDOW_VIEW(token_array, block_size + 1)

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Return overlapping read-only-contract ``x``/``y`` views.

        The two tensors share one batch owner and overlap by one token position.
        Callers must not modify either view in place.
        """
        start_limit = self._tokens.size - self._block_size
        starts: npt.NDArray[np.int64] = self._rng.integers(
            0,
            start_limit,
            size=self._batch_size,
            dtype=np.int64,
        )
        windows: npt.NDArray[np.int64] = self._token_windows[starts].astype(
            np.int64,
            copy=False,
        )
        batch = _FROM_NUMPY(windows).to(device=self._device)
        return batch[:, :-1], batch[:, 1:]

    def capture_random_state(self) -> str:
        """Serialize the independent sampler RNG for checkpointing."""
        return json.dumps(self._rng.bit_generator.state)

    def restore_random_state(self, state_json: str) -> None:
        """Restore the independent sampler RNG from a checkpoint."""
        self._rng.bit_generator.state = json.loads(state_json)
