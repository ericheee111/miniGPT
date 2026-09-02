"""Story Forge product rules: branch seeds and deterministic story history.

This module owns the product-level Story Forge rules that live above the data
layer and model runtime:

- a documented, deterministic per-branch seed derivation from ``(base_seed,
  branch_index)`` so three concurrent branches stay stable and reproducible; and
- a deterministic maximum story-history policy for later rounds that always
  keeps the control prefix and the most recent story tokens that fit the model
  context, never cutting through a BPE token.

It touches no model, tokenizer, serving, or HTTP state; those remain with the
single engine-owner thread. The public request schema is validated at the HTTP
boundary in :mod:`minigpt.public_demo`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "DEFAULT_BRANCH_COUNT",
    "MAX_BRANCH_TOKENS",
    "MAX_OPENING_CHARS",
    "MAX_ROUNDS",
    "MAX_STORY_SEED",
    "StoryForgeProductError",
    "StoryHistory",
    "branch_seed_for",
    "build_story_history",
)

DEFAULT_BRANCH_COUNT: Final = 3
MAX_BRANCH_TOKENS: Final = 64
MAX_OPENING_CHARS: Final = 10_000
MAX_STORY_SEED: Final = 2**63
MAX_ROUNDS: Final = 4

_BRANCH_HASH_DOMAIN: Final = "minigpt-story-forge-branch-seed-v1"

_BASE_SEED_REASON: Final = "base_seed must be an integer in [0, 2**63)"
_BRANCH_INDEX_REASON: Final = "branch_index must be a non-negative integer"
_TOKENS_REASON: Final = "token IDs must be non-negative integers"
_CONTEXT_REASON: Final = "max_context_tokens must exceed the control prefix length"
_CONTEXT_POSITIVE_REASON: Final = "max_context_tokens must be a positive integer"
_PREFIX_EMPTY_REASON: Final = "control_prefix_ids must be non-empty"


@dataclass(frozen=True, slots=True)
class StoryForgeProductError(ValueError):
    """Report an invalid Story Forge product request."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed product constraint."""
        return f"invalid story forge request: {self.reason}"


def branch_seed_for(base_seed: int, branch_index: int) -> int:
    """Derive a stable, distinct per-branch seed from ``(base_seed, branch_index)``.

    The derivation is a documented SHA-256 of a fixed domain separator and the
    two integers, reduced into ``[0, 2**63)``. Identical inputs always produce
    the same seed; distinct branch indices produce distinct seeds.
    """
    if type(base_seed) is not int or not 0 <= base_seed < MAX_STORY_SEED:
        raise StoryForgeProductError(_BASE_SEED_REASON)
    if type(branch_index) is not int or branch_index < 0:
        raise StoryForgeProductError(_BRANCH_INDEX_REASON)
    payload = f"{_BRANCH_HASH_DOMAIN}\n{base_seed}\n{branch_index}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest, 16) % MAX_STORY_SEED


@dataclass(frozen=True, slots=True)
class StoryHistory:
    """A deterministic story history built from control prefix and story tokens."""

    token_ids: tuple[int, ...]
    control_prefix_length: int
    truncated: bool


def _require_token_sequence(tokens: Sequence[int]) -> tuple[int, ...]:
    """Reject non-integer or negative token IDs while preserving exact int type."""
    result = tuple(tokens)
    # ``bool`` subclasses ``int``, so ``type(token) is not int`` rejects both
    # floats and ``bool`` while still accepting plain integers.
    if any(type(token) is not int or token < 0 for token in result):
        raise StoryForgeProductError(_TOKENS_REASON)
    return result


def build_story_history(
    *,
    control_prefix_ids: Sequence[int],
    story_token_ids: Sequence[int],
    max_context_tokens: int,
) -> StoryHistory:
    """Keep the control prefix plus the most recent story tokens that fit.

    The control prefix is always retained in full. Story tokens are
    right-anchored under a token budget so generation never cuts through a BPE
    token. Returns the ordered token IDs, the control-prefix length, and a
    truncation flag.
    """
    if type(max_context_tokens) is not int or max_context_tokens <= 0:
        raise StoryForgeProductError(_CONTEXT_POSITIVE_REASON)
    prefix = _require_token_sequence(control_prefix_ids)
    story = _require_token_sequence(story_token_ids)
    if not prefix:
        raise StoryForgeProductError(_PREFIX_EMPTY_REASON)
    if max_context_tokens <= len(prefix):
        raise StoryForgeProductError(_CONTEXT_REASON)
    budget = max_context_tokens - len(prefix)
    retained = story[-budget:] if len(story) > budget else story
    truncated = len(retained) < len(story)
    return StoryHistory(
        token_ids=(*prefix, *retained),
        control_prefix_length=len(prefix),
        truncated=truncated,
    )
