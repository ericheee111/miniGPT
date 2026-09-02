"""Validate a Story Forge checkpoint/tokenizer pair before public demo cutover."""  # noqa: INP001

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Never, cast

from minigpt.checkpoint import load_checkpoint_config, load_model_state
from minigpt.model import GPT, expected_gpt_parameter_count
from minigpt.serving_runtime import file_sha256
from minigpt.tokenizer import BPE_MODEL_FAMILY, load_tokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

_STORY_MODEL_PARAMETERS = 4_928_144

_VALIDATION_PREFIX = "story forge model validation failed"


class _ValidationError(ValueError):
    """Report a Story Forge model validation failure."""

    def __init__(self, reason: str) -> None:
        """Wrap the failure into a stable message."""
        super().__init__(f"{_VALIDATION_PREFIX}: {reason}")


def _invalid(reason: str) -> Never:
    raise _ValidationError(reason)


def validate_story_forge_model(  # noqa: C901
    *,
    checkpoint: Path,
    tokenizer: Path,
    checkpoint_sha256: str | None = None,
    tokenizer_sha256: str | None = None,
) -> dict[str, object]:
    """Validate the checkpoint/tokenizer pair and return a portable summary."""
    if not checkpoint.is_file():
        _invalid(f"checkpoint is not a file: {checkpoint}")
    if not tokenizer.is_file():
        _invalid(f"tokenizer is not a file: {tokenizer}")

    tok = load_tokenizer(tokenizer)
    if tok.model_family != BPE_MODEL_FAMILY:
        _invalid(f"tokenizer model_family must be {BPE_MODEL_FAMILY!r}")
    if tok.eos_token_id is None:
        _invalid("Story Forge tokenizer must define a non-null EOS token")
    if tok.bos_token_id is None:
        _invalid("Story Forge tokenizer must define a non-null BOS token")

    experiment = load_checkpoint_config(checkpoint).resolve_vocab_size(tok.vocab_size)
    if experiment.data.block_size <= 0:
        _invalid("checkpoint block_size must be positive")

    model = GPT(experiment.model.to_gpt_config(experiment.data.block_size))
    load_model_state(checkpoint, model)
    _ = model.eval()
    actual = model.parameter_count()
    expected = expected_gpt_parameter_count(model.config)
    if actual != expected:
        _invalid(f"model parameter count {actual} != expected {expected}")
    if expected != _STORY_MODEL_PARAMETERS:
        _invalid(f"expected Story Forge 5M count {_STORY_MODEL_PARAMETERS}, got {expected}")

    if checkpoint_sha256 is not None:
        observed = file_sha256(checkpoint)
        if observed != checkpoint_sha256:
            _invalid(f"checkpoint SHA-256 mismatch: expected {checkpoint_sha256}, got {observed}")
    if tokenizer_sha256 is not None:
        observed = file_sha256(tokenizer)
        if observed != tokenizer_sha256:
            _invalid(f"tokenizer SHA-256 mismatch: expected {tokenizer_sha256}, got {observed}")

    return {
        "tokenizer": {
            "model_family": tok.model_family,
            "tokenizer_type": tok.tokenizer_type,
            "vocab_size": tok.vocab_size,
            "bos_token_id": tok.bos_token_id,
            "eos_token_id": tok.eos_token_id,
        },
        "model": {
            "block_size": experiment.data.block_size,
            "parameter_count": actual,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the Story Forge model validation parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--checkpoint", type=Path, required=True)
    _ = parser.add_argument("--tokenizer", type=Path, required=True)
    _ = parser.add_argument("--checkpoint-sha256")
    _ = parser.add_argument("--tokenizer-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validation and print the portable summary on success."""
    arguments = build_parser().parse_args(argv)
    values = cast("dict[str, object]", vars(arguments))
    summary = validate_story_forge_model(
        checkpoint=cast("Path", values["checkpoint"]),
        tokenizer=cast("Path", values["tokenizer"]),
        checkpoint_sha256=cast("str | None", values["checkpoint_sha256"]),
        tokenizer_sha256=cast("str | None", values["tokenizer_sha256"]),
    )
    print(json.dumps(summary, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
