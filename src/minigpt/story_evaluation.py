"""Deterministic Story Forge model evaluation with strictly bounded metrics.

This module evaluates a trained Story Forge checkpoint through the public
tokenizer, checkpoint, and generation APIs. It reports only bounded,
reproducible quantities:

- validation loss/perplexity over a fixed number of deterministic, freshly
  seeded recorded batches (never the exact training-time val-batcher state);
- EOS stop rate and generated length;
- special-token leakage outside the natural EOS stop;
- invalid-decode detection;
- repeated 3/4-grams and the longest immediate tail loop;
- distinct 1/2/3 ratios;
- documented lexical proxy scores for world/tone/theme;
- same-seed determinism;
- different-seed diversity;
- cached-vs-uncached exact token equality.

The lexical proxy scores count normalized alias-phrase occurrences in decoded
text. They are keyword-presence surrogates only: they do NOT measure semantic
understanding, story quality, or authorship, and no such claim is made.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch

from minigpt.batching import TokenBatcher
from minigpt.checkpoint import (
    load_checkpoint_config,
    load_checkpoint_metadata,
    load_model_state,
)
from minigpt.model import GPT, expected_gpt_parameter_count
from minigpt.story import STORY_EVALUATION_CASES, StoryControls, frame_story_prompt
from minigpt.story_data import THEME_ALIASES, TONE_ALIASES, WORLD_ALIASES
from minigpt.tokenizer import BPE_SPECIAL_TOKEN_IDS, TokenizerProtocol, load_tokenizer
from minigpt.training_runtime import evaluate

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from minigpt.settings import ExperimentConfig

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None

# Conservative public defaults derived from measured samples, not guessed claims.
DEFAULT_EVAL_TEMPERATURE: Final = 0.8
DEFAULT_EVAL_TOP_K: Final = 20
DEFAULT_EVAL_MAX_NEW_TOKENS: Final = 96
DEFAULT_EVAL_VAL_BATCHES: Final = 20
DEFAULT_DIVERSITY_DELTA: Final = 4096

# Special tokens whose emission during decode (other than a terminal EOS) is a leak.
_ALL_SPECIAL_TOKEN_IDS: Final = frozenset(BPE_SPECIAL_TOKEN_IDS.values())


def _fingerprint_file(path: Path) -> dict[str, JsonValue]:
    """Hash one artifact and return its byte size and SHA-256."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


# -- Pure lexical/metric helpers (unit-testable without a model) -----------------


def longest_immediate_tail_loop(tokens: Sequence[int]) -> int:
    """Return the longest k where the tail is ``k`` tokens repeated twice."""
    for k in range(len(tokens) // 2, 0, -1):
        if tokens[-2 * k : -k] == tokens[-k:]:
            return k
    return 0


def _repeated_ngrams(tokens: Sequence[int], n: int) -> int:
    """Count n-grams that occur more than once in the sequence."""
    if len(tokens) < n:
        return 0
    counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return sum(1 for count in counts.values() if count > 1)


def distinct_ngram_ratio(tokens: Sequence[int], n: int) -> float:
    """Return distinct n-grams divided by total n-grams (0 when empty)."""
    if len(tokens) < n:
        return 0.0
    total = len(tokens) - n + 1
    distinct = len({tuple(tokens[i : i + n]) for i in range(total)})
    return distinct / total


def _normalized(text: str) -> str:
    """Lowercase and collapse whitespace for stable alias matching."""
    return " ".join(text.lower().split())


def _alias_hits(text: str, phrases: Sequence[str]) -> int:
    """Count normalized alias-phrase occurrences in decoded text."""
    normalized = _normalized(text)
    return sum(1 for phrase in phrases if _normalized(phrase) in normalized)


def story_lexical_proxy_scores(text: str) -> dict[str, int]:
    """Return world/tone/theme lexical proxy scores for one decoded text.

    These count alias-phrase keyword hits only. They are not semantic
    understanding, story-quality, or authorship measurements.
    """
    return {
        "world": sum(_alias_hits(text, phrases) for phrases in WORLD_ALIASES.values()),
        "tone": sum(_alias_hits(text, phrases) for phrases in TONE_ALIASES.values()),
        "theme": sum(_alias_hits(text, phrases) for phrases in THEME_ALIASES.values()),
    }


def special_token_leak_count(
    tokens: Sequence[int],
    *,
    eos_token_id: int | None,
) -> int:
    """Count emitted special tokens other than the natural EOS stop."""
    excluded = _ALL_SPECIAL_TOKEN_IDS - {eos_token_id}
    return sum(1 for token in tokens if token in excluded)


# -- Generation ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Bounded, reproducible metrics for one sampled continuation."""

    case_id: str
    length: int
    eos_hit: bool
    special_token_leaks: int
    invalid_decode: bool
    repeated_3grams: int
    repeated_4grams: int
    longest_loop: int
    distinct_1: float
    distinct_2: float
    distinct_3: float
    world_score: int
    tone_score: int
    theme_score: int
    text: str


@dataclass(frozen=True, slots=True)
class _Generation:
    tokens: tuple[int, ...]
    eos_hit: bool
    text: str
    invalid_decode: bool


def _generate_tokens(  # noqa: PLR0913
    model: GPT,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
    cached: bool,
    eos_token_id: int | None,
) -> tuple[tuple[int, ...], bool]:
    """Sample one continuation with a fixed seed and return the new tokens."""
    generator = torch.Generator(device="cpu")
    _ = generator.manual_seed(seed)
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long)
    if cached:
        generated = model.generate_cached(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
            eos_token_id=eos_token_id,
        )
    else:
        generated = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
            eos_token_id=eos_token_id,
        )
    flat = [int(generated[0, index]) for index in range(generated.shape[1])]
    new_tokens = tuple(flat[len(prompt_ids) :])
    eos_hit: bool = eos_token_id is not None and bool(new_tokens) and new_tokens[-1] == eos_token_id
    return new_tokens, eos_hit


def _run_generation(  # noqa: PLR0913, PLR0917
    model: GPT,
    tokenizer: TokenizerProtocol,
    world: str,
    tone: str,
    theme: str,
    opening: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
    cached: bool,
) -> _Generation:
    """Frame one case prompt and sample a continuation."""
    framed = frame_story_prompt(
        tokenizer,
        StoryControls(world=world, tone=tone, theme=theme),
        opening,
    )
    tokens, eos_hit = _generate_tokens(
        model,
        framed.token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        cached=cached,
        eos_token_id=tokenizer.eos_token_id,
    )
    invalid_decode = False
    try:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception:  # noqa: BLE001 - decode failure is itself the measured signal.
        text = ""
        invalid_decode = True
    return _Generation(tokens=tokens, eos_hit=eos_hit, text=text, invalid_decode=invalid_decode)


def _metrics_for_generation(
    generation: _Generation,
    *,
    case_id: str,
    eos_token_id: int | None,
) -> GenerationMetrics:
    """Fold one raw generation into bounded metrics."""
    tokens = generation.tokens
    proxies = story_lexical_proxy_scores(generation.text)
    return GenerationMetrics(
        case_id=case_id,
        length=len(tokens),
        eos_hit=generation.eos_hit,
        special_token_leaks=special_token_leak_count(tokens, eos_token_id=eos_token_id),
        invalid_decode=generation.invalid_decode,
        repeated_3grams=_repeated_ngrams(tokens, 3),
        repeated_4grams=_repeated_ngrams(tokens, 4),
        longest_loop=longest_immediate_tail_loop(tokens),
        distinct_1=distinct_ngram_ratio(tokens, 1),
        distinct_2=distinct_ngram_ratio(tokens, 2),
        distinct_3=distinct_ngram_ratio(tokens, 3),
        world_score=proxies["world"],
        tone_score=proxies["tone"],
        theme_score=proxies["theme"],
        text=generation.text,
    )


# -- Checkpoint evaluation --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Identity recorded for a Story Forge checkpoint and its data artifacts."""

    checkpoint_path: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    format_version: int
    completed_step: int
    parameter_count_actual: int
    parameter_count_expected: int
    tokenizer_sha256: str
    train_sha256: str
    val_sha256: str


@dataclass(frozen=True, slots=True)
class StoryEvaluationSummary:
    """Complete bounded evaluation report for one Story Forge checkpoint."""

    identity: CheckpointIdentity
    sampling: dict[str, JsonValue]
    validation: dict[str, JsonValue]
    cases: tuple[GenerationMetrics, ...]
    determinism: dict[str, JsonValue]
    diversity: dict[str, JsonValue]
    cached_uncached: dict[str, JsonValue]
    claim_policy: dict[str, str]

    def to_json(self) -> dict[str, JsonValue]:
        """Render the summary as a JSON-serializable document."""
        return {
            "identity": {
                "checkpoint_path": self.identity.checkpoint_path,
                "checkpoint_bytes": self.identity.checkpoint_bytes,
                "checkpoint_sha256": self.identity.checkpoint_sha256,
                "format_version": self.identity.format_version,
                "completed_step": self.identity.completed_step,
                "parameter_count_actual": self.identity.parameter_count_actual,
                "parameter_count_expected": self.identity.parameter_count_expected,
            },
            "sampling": dict(self.sampling),
            "validation": dict(self.validation),
            "cases": [asdict(case) for case in self.cases],
            "determinism": dict(self.determinism),
            "diversity": dict(self.diversity),
            "cached_uncached": dict(self.cached_uncached),
            "claim_policy": dict(self.claim_policy),
        }


_CLAIM_POLICY: Final = {
    "verdict": "descriptive_only",
    "claim": (
        "bounded lexical and distributional proxy metrics only; no semantic "
        "understanding, story-quality, authorship, production readiness, or "
        "universal speedup is claimed"
    ),
}


def _load_story_model(checkpoint_path: Path) -> tuple[GPT, ExperimentConfig, TokenizerProtocol]:
    """Load a Story Forge model, its resolved config, and tokenizer."""
    config = load_checkpoint_config(checkpoint_path)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    resolved = config.resolve_vocab_size(tokenizer.vocab_size)
    model = GPT(resolved.model.to_gpt_config(resolved.data.block_size))
    load_model_state(checkpoint_path, model)
    _ = model.eval()
    return model, resolved, tokenizer


def _validation_loss(
    model: GPT,
    config: ExperimentConfig,
    batch_count: int,
) -> dict[str, JsonValue]:
    """Return mean validation loss/perplexity on fixed deterministic batches."""
    val_tokens = cast(
        "npt.NDArray[np.uint16]",
        np.load(config.data.val_path, mmap_mode="r"),
    )
    val_batcher = TokenBatcher(
        val_tokens,
        batch_size=config.data.batch_size,
        block_size=config.data.block_size,
        seed=config.runtime.seed + 1,
    )
    mean_loss = evaluate(model, val_batcher, batch_count)
    return {
        "mean_val_loss": mean_loss,
        "val_perplexity": math.exp(mean_loss),
        "batch_count": batch_count,
        "batch_size": config.data.batch_size,
        "block_size": config.data.block_size,
        "val_batcher_seed": config.runtime.seed + 1,
    }


def evaluate_story_checkpoint(  # noqa: PLR0913
    checkpoint_path: Path,
    *,
    max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_EVAL_TEMPERATURE,
    top_k: int | None = DEFAULT_EVAL_TOP_K,
    val_batches: int = DEFAULT_EVAL_VAL_BATCHES,
    case_ids: Sequence[str] | None = None,
) -> StoryEvaluationSummary:
    """Evaluate one Story Forge checkpoint and return a bounded summary."""
    model, config, tokenizer = _load_story_model(checkpoint_path)
    metadata = load_checkpoint_metadata(checkpoint_path)

    gpt_config = config.model.to_gpt_config(config.data.block_size)
    expected_params = expected_gpt_parameter_count(gpt_config)
    actual_params = model.parameter_count()

    identity = CheckpointIdentity(
        checkpoint_path=str(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_sha256=cast(
            "str",
            _fingerprint_file(checkpoint_path)["sha256"],
        ),
        format_version=metadata.format_version,
        completed_step=metadata.completed_step,
        parameter_count_actual=actual_params,
        parameter_count_expected=expected_params,
        tokenizer_sha256=metadata.dataset_fingerprints.tokenizer_sha256,
        train_sha256=metadata.dataset_fingerprints.train_sha256,
        val_sha256=metadata.dataset_fingerprints.val_sha256,
    )

    cases = [case for case in STORY_EVALUATION_CASES if case_ids is None or case.id in case_ids]

    metrics: list[GenerationMetrics] = []
    for case in cases:
        generation = _run_generation(
            model,
            tokenizer,
            case.world,
            case.tone,
            case.theme,
            case.opening,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=case.seed,
            cached=False,
        )
        metrics.append(
            _metrics_for_generation(
                generation,
                case_id=case.id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )

    # Same-seed determinism: regenerate every case with the same seed.
    determinism_tokens: list[bool] = []
    for case in cases:
        first = _run_generation(
            model,
            tokenizer,
            case.world,
            case.tone,
            case.theme,
            case.opening,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=case.seed,
            cached=False,
        )
        second = _run_generation(
            model,
            tokenizer,
            case.world,
            case.tone,
            case.theme,
            case.opening,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=case.seed,
            cached=False,
        )
        determinism_tokens.append(first.tokens == second.tokens)

    # Different-seed diversity: compare the base seed against a shifted seed.
    diversity_tokens: list[bool] = []
    for case in cases:
        base = _run_generation(
            model,
            tokenizer,
            case.world,
            case.tone,
            case.theme,
            case.opening,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=case.seed,
            cached=False,
        )
        shifted = _run_generation(
            model,
            tokenizer,
            case.world,
            case.tone,
            case.theme,
            case.opening,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=case.seed + DEFAULT_DIVERSITY_DELTA,
            cached=False,
        )
        diversity_tokens.append(base.tokens != shifted.tokens)

    # Cached-vs-uncached exact token equality (correctness, not quality).
    cached_equal: list[bool] = []
    for case in cases:
        uncached = _run_generation(
            model,
            tokenizer,
            case.world,
            case.tone,
            case.theme,
            case.opening,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=case.seed,
            cached=False,
        )
        cached = _run_generation(
            model,
            tokenizer,
            case.world,
            case.tone,
            case.theme,
            case.opening,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=case.seed,
            cached=True,
        )
        cached_equal.append(uncached.tokens == cached.tokens)

    return StoryEvaluationSummary(
        identity=identity,
        sampling={
            "temperature": temperature,
            "top_k": top_k if top_k is not None else "none",
            "max_new_tokens": max_new_tokens,
            "case_count": len(cases),
        },
        validation=_validation_loss(model, config, val_batches),
        cases=tuple(metrics),
        determinism={
            "cases_checked": len(determinism_tokens),
            "identical_count": sum(determinism_tokens),
            "all_identical": all(determinism_tokens),
        },
        diversity={
            "cases_checked": len(diversity_tokens),
            "differing_count": sum(diversity_tokens),
            "seed_delta": DEFAULT_DIVERSITY_DELTA,
        },
        cached_uncached={
            "cases_checked": len(cached_equal),
            "exact_equal_count": sum(cached_equal),
            "all_exact_equal": all(cached_equal),
        },
        claim_policy=dict(_CLAIM_POLICY),
    )


def write_evaluation_report(summary: StoryEvaluationSummary, output_path: Path) -> None:
    """Write a deterministic UTF-8/LF evaluation report JSON document."""
    document = summary.to_json()
    _ = output_path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_evaluation_report(path: Path) -> dict[str, JsonValue]:
    """Load a previous evaluation report and return it unchanged."""
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        msg = f"{path} must contain a JSON object"
        raise TypeError(msg)
    return cast("dict[str, JsonValue]", raw)
