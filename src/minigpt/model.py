from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from minigpt.config import GPTConfig, InvalidModelConfigError
from minigpt.layers import CausalSelfAttention, LayerNorm, MLP, TransformerBlock

__all__ = (
    "CausalSelfAttention",
    "GPT",
    "GPTConfig",
    "InvalidGenerationConfigError",
    "InvalidModelConfigError",
    "InvalidTokenTensorError",
    "LayerNorm",
    "MLP",
    "TokenIdOutOfRangeError",
    "TransformerBlock",
)


@dataclass(frozen=True, slots=True)
class InvalidTokenTensorError(ValueError):
    """Report an input or target tensor with an invalid shape or dtype."""

    name: str
    reason: str

    def __str__(self) -> str:
        return f"invalid {self.name} tensor: {self.reason}"


@dataclass(frozen=True, slots=True)
class TokenIdOutOfRangeError(ValueError):
    """Report token IDs that cannot index the configured vocabulary."""

    name: str
    minimum: int
    maximum: int
    vocab_size: int

    def __str__(self) -> str:
        return (
            f"{self.name} token IDs span [{self.minimum}, {self.maximum}], "
            f"but vocabulary size {self.vocab_size} accepts [0, {self.vocab_size})"
        )


@dataclass(frozen=True, slots=True)
class InvalidGenerationConfigError(ValueError):
    """Report sampling settings that do not define a valid distribution."""

    reason: str

    def __str__(self) -> str:
        return f"invalid generation configuration: {self.reason}"


class GPT(nn.Module):
    """Run autoregressive language modeling and token generation on GPT blocks."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def _validate_token_tensor(self, token_ids: Tensor, *, name: str) -> None:
        if token_ids.ndim != 2:
            raise InvalidTokenTensorError(name, "expected shape [batch, time]")
        if token_ids.dtype != torch.long:
            raise InvalidTokenTensorError(name, "dtype must be torch.int64")
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise InvalidTokenTensorError(name, "batch and time dimensions must be non-zero")
        minimum = int(token_ids.min().item())
        maximum = int(token_ids.max().item())
        if minimum < 0 or maximum >= self.config.vocab_size:
            raise TokenIdOutOfRangeError(
                name=name,
                minimum=minimum,
                maximum=maximum,
                vocab_size=self.config.vocab_size,
            )

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return [B,T,V] logits and optional mean next-token cross entropy."""
        self._validate_token_tensor(token_ids, name="input")
        batch_size, time_steps = token_ids.shape
        if time_steps > self.config.block_size:
            raise InvalidTokenTensorError(
                "input",
                f"time dimension {time_steps} exceeds block_size {self.config.block_size}",
            )
        if targets is not None:
            self._validate_token_tensor(targets, name="target")
            if targets.shape != token_ids.shape:
                raise InvalidTokenTensorError(
                    "target",
                    f"shape {tuple(targets.shape)} must equal input shape {tuple(token_ids.shape)}",
                )

        positions = torch.arange(time_steps, device=token_ids.device)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
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
        self._validate_token_tensor(token_ids, name="prompt")
        if max_new_tokens < 0:
            raise InvalidGenerationConfigError("max_new_tokens must be non-negative")
        if temperature <= 0.0:
            raise InvalidGenerationConfigError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise InvalidGenerationConfigError("top_k must be positive when provided")

        generated = token_ids
        for _ in range(max_new_tokens):
            model_input = generated[:, -self.config.block_size :]
            logits, _ = self(model_input)
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                retained_count = min(top_k, self.config.vocab_size)
                retained_logits = torch.topk(next_token_logits, retained_count, dim=-1).values
                cutoff = retained_logits[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < cutoff,
                    -torch.inf,
                )
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
