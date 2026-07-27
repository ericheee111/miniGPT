"""Define the miniGPT language model and its validation errors."""

from dataclasses import dataclass
from typing import Final, cast, final

import torch
from torch import Tensor, nn
from torch.nn import functional
from typing_extensions import override

from minigpt.layers import MLP, CausalSelfAttention, LayerNorm, TransformerBlock
from minigpt.settings import GPTConfig, InvalidModelConfigError

__all__ = (
    "GPT",
    "MLP",
    "CausalSelfAttention",
    "GPTConfig",
    "InvalidGenerationConfigError",
    "InvalidModelConfigError",
    "InvalidTokenTensorError",
    "LayerNorm",
    "TokenIdOutOfRangeError",
    "TransformerBlock",
    "UnexpectedTransformerBlockError",
)

_INPUT_NAME: Final = "input"
_TARGET_NAME: Final = "target"
_PROMPT_NAME: Final = "prompt"
_SHAPE_REASON: Final = "expected shape [batch, time]"
_DTYPE_REASON: Final = "dtype must be torch.int64"
_NON_EMPTY_REASON: Final = "batch and time dimensions must be non-zero"
_GENERATION_LENGTH_REASON: Final = "max_new_tokens must be non-negative"
_TEMPERATURE_REASON: Final = "temperature must be positive"
_TOP_K_REASON: Final = "top_k must be positive when provided"
_TOKEN_TENSOR_DIMENSIONS: Final = 2


@dataclass(frozen=True, slots=True)
class InvalidTokenTensorError(ValueError):
    """Report an input or target tensor with an invalid shape or dtype."""

    name: str
    reason: str

    @override
    def __str__(self) -> str:
        """Render the tensor name and its failed constraint."""
        return f"invalid {self.name} tensor: {self.reason}"


@dataclass(frozen=True, slots=True)
class TokenIdOutOfRangeError(ValueError):
    """Report token IDs that cannot index the configured vocabulary."""

    name: str
    minimum: int
    maximum: int
    vocab_size: int

    @override
    def __str__(self) -> str:
        """Render the observed token range and configured vocabulary."""
        return (
            f"{self.name} token IDs span [{self.minimum}, {self.maximum}], "
            f"but vocabulary size {self.vocab_size} accepts [0, {self.vocab_size})"
        )


@dataclass(frozen=True, slots=True)
class InvalidGenerationConfigError(ValueError):
    """Report sampling settings that do not define a valid distribution."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed generation constraint."""
        return f"invalid generation configuration: {self.reason}"


@dataclass(frozen=True, slots=True)
class UnexpectedTransformerBlockError(RuntimeError):
    """Report an unexpected module in the Transformer stack."""

    index: int
    actual_type: str

    @override
    def __str__(self) -> str:
        """Render the invalid block position and type."""
        return (
            f"unexpected GPT block {self.index}: expected TransformerBlock, got {self.actual_type}"
        )


@final
class GPT(nn.Module):
    """Run autoregressive language modeling and token generation on GPT blocks."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize embeddings, Transformer blocks, and language-model head."""
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def _validate_token_tensor(self, token_ids: Tensor, *, name: str) -> None:
        if token_ids.ndim != _TOKEN_TENSOR_DIMENSIONS:
            raise InvalidTokenTensorError(name, _SHAPE_REASON)
        if token_ids.dtype != torch.long:
            raise InvalidTokenTensorError(name, _DTYPE_REASON)
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise InvalidTokenTensorError(name, _NON_EMPTY_REASON)
        minimum = int(token_ids.min().item())
        maximum = int(token_ids.max().item())
        if minimum < 0 or maximum >= self.config.vocab_size:
            raise TokenIdOutOfRangeError(
                name=name,
                minimum=minimum,
                maximum=maximum,
                vocab_size=self.config.vocab_size,
            )

    @override
    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return [B,T,V] logits and optional mean next-token cross entropy."""
        self._validate_token_tensor(token_ids, name=_INPUT_NAME)
        batch_size, time_steps = token_ids.shape
        if time_steps > self.config.block_size:
            reason = f"time dimension {time_steps} exceeds block_size {self.config.block_size}"
            raise InvalidTokenTensorError(
                _INPUT_NAME,
                reason,
            )
        if targets is not None:
            self._validate_token_tensor(targets, name=_TARGET_NAME)
            if targets.shape != token_ids.shape:
                reason = (
                    f"shape {tuple(targets.shape)} must equal input shape {tuple(token_ids.shape)}"
                )
                raise InvalidTokenTensorError(
                    _TARGET_NAME,
                    reason,
                )

        positions = torch.arange(time_steps, device=token_ids.device)
        token_embeddings = cast("Tensor", self.token_embedding(token_ids))
        position_embeddings = cast("Tensor", self.position_embedding(positions))
        hidden_states = cast(
            "Tensor",
            self.embedding_dropout(token_embeddings + position_embeddings),
        )
        for index, module in enumerate(self.blocks):
            if not isinstance(module, TransformerBlock):
                raise UnexpectedTransformerBlockError(index, type(module).__name__)
            hidden_states = cast("Tensor", module(hidden_states))
        normalized = cast("Tensor", self.final_norm(hidden_states))
        logits = cast("Tensor", self.lm_head(normalized))

        loss: Tensor | None = None
        if targets is not None:
            loss = functional.cross_entropy(
                logits.reshape(batch_size * time_steps, self.config.vocab_size),
                targets.reshape(batch_size * time_steps),
            )
        return logits, loss

    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Autoregressively append sampled token IDs to a prompt."""
        self._validate_token_tensor(token_ids, name=_PROMPT_NAME)
        if max_new_tokens < 0:
            raise InvalidGenerationConfigError(_GENERATION_LENGTH_REASON)
        if temperature <= 0.0:
            raise InvalidGenerationConfigError(_TEMPERATURE_REASON)
        if top_k is not None and top_k <= 0:
            raise InvalidGenerationConfigError(_TOP_K_REASON)

        generated = token_ids
        for _ in range(max_new_tokens):
            model_input = generated[:, -self.config.block_size :]
            logits, _ = cast("tuple[Tensor, Tensor | None]", self(model_input))
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                retained_count = min(top_k, self.config.vocab_size)
                retained_logits = torch.topk(next_token_logits, retained_count, dim=-1).values
                cutoff = retained_logits[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < cutoff,
                    -torch.inf,
                )
            probabilities = functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
