"""Provide the neural-network building blocks used by the GPT model."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final, override

import torch
from torch import Tensor, nn
from torch.nn import functional

if TYPE_CHECKING:
    from minigpt.settings import GPTConfig


@final
class LayerNorm(nn.Module):
    """Normalize each token vector and optionally learn an additive bias."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        bias: bool,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize scale, optional bias, and numerical stability epsilon."""
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim)) if bias else None

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize the final tensor dimension using population variance."""
        centered = hidden_states - hidden_states.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.epsilon)
        shifted = normalized * self.weight
        if self.bias is not None:
            shifted = shifted + self.bias
        return shifted


@final
class CausalSelfAttention(nn.Module):
    """Apply masked multi-head self-attention over a token sequence."""

    causal_mask: Tensor

    def __init__(self, config: GPTConfig) -> None:
        """Initialize projections, dropout, and the reusable causal mask."""
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.qkv_projection = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        self.output_projection = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("causal_mask", mask, persistent=False)
        self.causal_mask = mask

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Transform [B,T,C] states through scaled dot-product attention."""
        batch_size, time_steps, channels = hidden_states.shape
        projected_qkv = self.qkv_projection.forward(hidden_states)
        query = projected_qkv[..., : self.n_embd]
        key = projected_qkv[..., self.n_embd : 2 * self.n_embd]
        value = projected_qkv[..., 2 * self.n_embd :]
        query = query.view(batch_size, time_steps, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(batch_size, time_steps, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(batch_size, time_steps, self.n_head, self.head_size).transpose(1, 2)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        allowed_positions = self.causal_mask[:, :, :time_steps, :time_steps]
        attention_scores = attention_scores.masked_fill(
            ~allowed_positions,
            torch.finfo(attention_scores.dtype).min,
        )
        attention_weights = functional.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout.forward(attention_weights)

        context = attention_weights @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, time_steps, channels)
        projected = self.output_projection.forward(context)
        return self.residual_dropout.forward(projected)


@final
class MLP(nn.Module):
    """Expand, transform, and project each token independently."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the position-wise expansion and projection layers."""
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.input_projection = nn.Linear(
            config.n_embd,
            hidden_dim,
            bias=config.bias,
        )
        self.activation = nn.GELU(approximate="tanh")
        self.output_projection = nn.Linear(
            hidden_dim,
            config.n_embd,
            bias=config.bias,
        )
        self.dropout = nn.Dropout(config.dropout)

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the position-wise feed-forward network."""
        hidden_states = self.input_projection.forward(hidden_states)
        hidden_states = self.activation.forward(hidden_states)
        projected = self.output_projection.forward(hidden_states)
        return self.dropout.forward(projected)


@final
class TransformerBlock(nn.Module):
    """Combine pre-normalized attention and MLP residual branches."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize pre-normalization and both residual branches."""
        super().__init__()
        self.attention_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    @override
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply both residual branches without changing tensor shape."""
        attention_input = self.attention_norm.forward(hidden_states)
        hidden_states = hidden_states + self.attention.forward(attention_input)
        mlp_input = self.mlp_norm.forward(hidden_states)
        return hidden_states + self.mlp.forward(mlp_input)
