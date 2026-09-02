"""Read-only next-token distribution and sequence surprisal inspection math.

Prediction Lab opens two bounded, read-only inspection primitives on top of an
already-loaded model:

- a next-token distribution for a framed prompt (top-k candidates with raw
  logits and stable float32 probabilities), and
- a per-token sequence surprisal score (negative log-likelihood) with the
  aggregate mean NLL and perplexity.

These are maths on model logits only. They are NOT authorship detection or
semantic understanding: a low surprisal means the model assigned the token a
high conditional probability under its own learned distribution, nothing more.

Every inspection runs the model under ``model.eval()`` and ``torch.no_grad()``
and restores the model's prior training mode afterward, so a read-only
observation can never mutate model state or leak gradients. It never samples,
never advances any request RNG, and never touches the serving engine's
lifecycle or KV caches. It is intended to be invoked only from the single
engine-owner thread.
"""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Never, cast

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from minigpt.model import GPT

__all__ = (
    "MAX_INSPECTION_TOKENS",
    "MAX_TOP_K",
    "NextTokenDistribution",
    "NonFiniteLogitsError",
    "PerTokenSurprisal",
    "PredictionInspectionError",
    "PredictionOverflowError",
    "SequenceSurprisal",
    "TokenPrediction",
    "compute_next_token_distribution",
    "compute_sequence_surprisal",
    "per_token_negative_log_likelihood",
    "stable_softmax_float32",
)

MAX_TOP_K: Final = 10
MAX_INSPECTION_TOKENS: Final = 512

_LOGITS_DIMENSIONS: Final = 2
_TARGETS_DIMENSIONS: Final = 1
_MINIMUM_SEQUENCE_TOKENS: Final = 2

# ``math.exp`` overflows to ``OverflowError`` or infinity once the argument
# approaches ``log(float_info.max)``; guard aggregate perplexity against it.
_MAX_EXP_ARGUMENT: Final = math.log(sys.float_info.max)

_TOO_FEW_TOKENS: Final = "token sequence must contain at least two tokens"
_EMPTY_PROMPT: Final = "prompt token sequence must be non-empty"
_NON_FINITE_LOGITS: Final = "model produced non-finite logits; inspection aborted"
_NON_FINITE_NLL: Final = "computed negative log-likelihood is not finite; inspection aborted"
_TOP_K_RANGE: Final = "top_k must be an integer in [1, 10]"
_CONTROL_PREFIX_RANGE: Final = (
    "control_prefix_length must be a non-negative integer within the sequence"
)
_UNSTABLE_PERPLEXITY: Final = "aggregate perplexity overflowed; inspection aborted"


class PredictionInspectionError(ValueError):
    """Report an invalid or unreachable inspection request.

    Not a frozen dataclass so the interpreter may attach ``__traceback__``.
    """

    def __init__(self, reason: str) -> None:
        """Wrap the inspection failure into a stable message."""
        super().__init__(f"prediction inspection failed: {reason}")


class NonFiniteLogitsError(PredictionInspectionError):
    """Report model logits that contain NaN or infinity."""

    def __init__(self) -> None:
        """Construct the fixed non-finite failure."""
        super().__init__(_NON_FINITE_LOGITS)


class PredictionOverflowError(PredictionInspectionError):
    """Report a non-finite or overflowing aggregate inspection output."""


def _invalid(reason: str) -> Never:
    raise PredictionInspectionError(reason)


@dataclass(frozen=True, slots=True)
class TokenPrediction:
    """One top-k candidate: its ID, raw logit, and stable probability."""

    token_id: int
    logit: float
    probability: float


@dataclass(frozen=True, slots=True)
class NextTokenDistribution:
    """A top-k next-token distribution in descending probability order."""

    top_k: int
    candidates: tuple[TokenPrediction, ...]


@dataclass(frozen=True, slots=True)
class PerTokenSurprisal:
    """The negative log-likelihood of one observed token."""

    token_id: int
    surprisal: float
    is_control: bool


@dataclass(frozen=True, slots=True)
class SequenceSurprisal:
    """Per-token and aggregate surprisal for one framed token sequence.

    ``per_token`` carries one entry per predicted position with an ``is_control``
    marker separating control-prefix positions from user-text positions.
    ``mean_nll``/``perplexity`` describe the full framed sequence; when user
    text is present, ``user_mean_nll``/``user_perplexity`` describe only the
    user-text positions. The full framed score is not a user-text score.
    """

    per_token: tuple[PerTokenSurprisal, ...]
    mean_nll: float
    perplexity: float
    control_prefix_length: int
    user_mean_nll: float | None
    user_perplexity: float | None


@contextmanager
def _eval_no_grad(model: GPT) -> Generator[None]:
    """Run the model in eval + no_grad, restoring its prior training mode."""
    was_training = model.training
    try:
        _ = model.eval()
        with torch.no_grad():
            yield
    finally:
        if was_training:
            _ = model.train()


def _require_finite_logits(logits: Tensor) -> None:
    """Reject any NaN or infinite entry before softmax or NLL propagates it."""
    if not bool(torch.isfinite(logits).all().item()):
        raise NonFiniteLogitsError


def _safe_perplexity(mean_nll: float) -> float:
    """Return ``exp(mean_nll)``, rejecting non-finite or overflowing aggregates."""
    if not math.isfinite(mean_nll) or mean_nll >= _MAX_EXP_ARGUMENT:
        raise PredictionOverflowError(_UNSTABLE_PERPLEXITY)
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError as error:
        raise PredictionOverflowError(_UNSTABLE_PERPLEXITY) from error
    if not math.isfinite(perplexity):
        raise PredictionOverflowError(_UNSTABLE_PERPLEXITY)
    return perplexity


def stable_softmax_float32(logits: Tensor) -> Tensor:
    """Compute a numerically stable softmax in float32.

    The maximum logit is subtracted before exponentiation to avoid overflow.
    All arithmetic is forced to ``torch.float32`` so probabilities are stable
    and comparable across the CPU runtime regardless of the model dtype.
    """
    _require_finite_logits(logits)
    shifted = logits.to(torch.float32)
    shifted = shifted - torch.max(shifted)
    exponentials = torch.exp(shifted)
    return exponentials / torch.sum(exponentials)


def per_token_negative_log_likelihood(logits: Tensor, targets: Tensor) -> Tensor:
    """Return per-position negative log-likelihood for a ``[seq, vocab]`` tensor.

    ``logits`` holds one vocabulary distribution per row; ``targets`` holds the
    observed token ID for each row. The result is ``-log_softmax(logits)[t, target]``
    in float32. Non-finite logits are rejected before softmax and non-finite NLL
    after it.
    """
    if logits.ndim != _LOGITS_DIMENSIONS:
        _invalid("logits must have shape [seq, vocab]")
    if targets.ndim != _TARGETS_DIMENSIONS or targets.shape[0] != logits.shape[0]:
        _invalid("targets must have shape [seq] matching logits rows")
    _require_finite_logits(logits)
    log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
    gathered = log_probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
    nll = -gathered
    if not bool(torch.isfinite(nll).all().item()):
        raise PredictionOverflowError(_NON_FINITE_NLL)
    return nll


def _validate_inspection_tokens(token_ids: Sequence[int]) -> list[int]:
    ids = list(token_ids)
    if not ids:
        _invalid(_EMPTY_PROMPT)
    if len(ids) > MAX_INSPECTION_TOKENS:
        reason = f"token sequence exceeds {MAX_INSPECTION_TOKENS} tokens"
        _invalid(reason)
    if any(type(token) is not int or token < 0 for token in ids):
        _invalid("token IDs must be non-negative integers")
    return ids


def _validate_control_prefix_length(token_count: int, control_prefix_length: int) -> None:
    if type(control_prefix_length) is not int or not 0 <= control_prefix_length < token_count:
        _invalid(_CONTROL_PREFIX_RANGE)


def _final_logits(model: GPT, tensor: Tensor) -> Tensor:
    with _eval_no_grad(model):
        logits, _loss = cast("tuple[Tensor, Tensor | None]", model(tensor))
    _require_finite_logits(logits)
    return logits[0, -1]


def compute_next_token_distribution(
    model: GPT,
    prompt_ids: Sequence[int],
    *,
    top_k: int,
) -> NextTokenDistribution:
    """Return the top-k next-token distribution for a framed prompt.

    The prompt is evaluated once with a stateless forward pass. ``probability``
    values are the temperature-1 softmax over raw logits in float32; ``logit``
    values are the raw (pre-softmax) last-position logits so callers can re-apply
    their own temperature.
    """
    if type(top_k) is not int or not 1 <= top_k <= MAX_TOP_K:
        _invalid(_TOP_K_RANGE)
    ids = _validate_inspection_tokens(prompt_ids)
    tensor = torch.tensor([ids], dtype=torch.long)
    logits = _final_logits(model, tensor)
    probabilities = stable_softmax_float32(logits)

    retained = min(top_k, logits.numel())
    _values, indices = torch.topk(logits, retained)
    indices_list = cast("list[int]", indices.tolist())  # pyright: ignore[reportUnknownMemberType]
    candidates = tuple(
        TokenPrediction(
            token_id=token_id,
            logit=float(logits[token_id].item()),
            probability=float(probabilities[token_id].item()),
        )
        for token_id in indices_list
    )
    return NextTokenDistribution(top_k=top_k, candidates=candidates)


def compute_sequence_surprisal(
    model: GPT,
    token_ids: Sequence[int],
    *,
    control_prefix_length: int = 0,
) -> SequenceSurprisal:
    """Return per-token NLL/surprisal and aggregate mean NLL/perplexity.

    For a sequence of length ``T``, position ``t`` predicts token ``t + 1`` from
    the prefix ending at ``t``. Surprisal is the negative log-likelihood of each
    observed next token. Mean NLL is the arithmetic mean over positions and
    perplexity is ``exp(mean_nll)``. This is a model-likelihood measurement, not
    authorship detection or semantic understanding.

    ``control_prefix_length`` marks the leading control-prefix positions so the
    user-text mean NLL/perplexity can be reported separately; the full-framed
    ``mean_nll``/``perplexity`` remain available and are not a user-text score.
    """
    ids = _validate_inspection_tokens(token_ids)
    if len(ids) < _MINIMUM_SEQUENCE_TOKENS:
        _invalid(_TOO_FEW_TOKENS)
    _validate_control_prefix_length(len(ids), control_prefix_length)
    tensor = torch.tensor([ids], dtype=torch.long)
    with _eval_no_grad(model):
        logits, _loss = cast("tuple[Tensor, Tensor | None]", model(tensor))
    _require_finite_logits(logits)
    flat_logits = logits[0]  # [T, V]
    predictor_logits = flat_logits[:-1]  # [T-1, V]
    targets = tensor[0, 1:]  # [T-1]
    nll = per_token_negative_log_likelihood(predictor_logits, targets)
    target_list = cast("list[int]", targets.tolist())  # pyright: ignore[reportUnknownMemberType]
    nll_list = cast("list[float]", nll.tolist())  # pyright: ignore[reportUnknownMemberType]

    # ``targets[j]`` equals ``token_ids[j + 1]``, i.e. the token at 0-based
    # absolute position ``j + 1``. Control-prefix tokens occupy positions
    # ``0 .. control_prefix_length - 1``; the target is a control token when
    # ``j + 1 < control_prefix_length`` and a user-text token otherwise (the
    # ``<bos>`` position 0 is never a predicted target).
    is_control_flags = tuple(
        position < control_prefix_length for position in range(1, len(target_list) + 1)
    )
    per_token = tuple(
        PerTokenSurprisal(
            token_id=token_id,
            surprisal=surprise,
            is_control=is_control,
        )
        for (token_id, surprise), is_control in zip(
            zip(target_list, nll_list, strict=True),
            is_control_flags,
            strict=True,
        )
    )
    mean_nll = float(nll.mean().item())
    if not math.isfinite(mean_nll):
        raise PredictionOverflowError(_NON_FINITE_NLL)
    perplexity = _safe_perplexity(mean_nll)

    user_surprisals = [entry.surprisal for entry in per_token if not entry.is_control]
    user_mean_nll: float | None
    user_perplexity: float | None
    if user_surprisals:
        user_mean_nll = sum(user_surprisals) / len(user_surprisals)
        if not math.isfinite(user_mean_nll):
            raise PredictionOverflowError(_NON_FINITE_NLL)
        user_perplexity = _safe_perplexity(user_mean_nll)
    else:
        user_mean_nll = None
        user_perplexity = None

    return SequenceSurprisal(
        per_token=per_token,
        mean_nll=mean_nll,
        perplexity=perplexity,
        control_prefix_length=control_prefix_length,
        user_mean_nll=user_mean_nll,
        user_perplexity=user_perplexity,
    )
