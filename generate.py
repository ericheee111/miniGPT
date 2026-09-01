"""Generate text from a trained miniGPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING

import torch

from minigpt.checkpoint import load_checkpoint_config, load_model_state
from minigpt.model import GPT
from minigpt.tokenizer import load_tokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the text-generation command-line parser."""
    parser = argparse.ArgumentParser(description="Generate text from a MiniTrainGPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Use KV cache inference instead of the full-context generation baseline.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load a checkpoint and sample text from its persisted configuration."""
    arguments = build_parser().parse_args(argv)
    config = load_checkpoint_config(arguments.checkpoint)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    model = GPT(config.model.to_gpt_config(config.data.block_size))
    load_model_state(arguments.checkpoint, model)
    _ = model.eval()
    seed = config.runtime.seed if arguments.seed is None else arguments.seed
    generator = torch.Generator(device="cpu")
    _ = generator.manual_seed(seed)
    prompt = torch.tensor([tokenizer.encode(arguments.prompt)], dtype=torch.long)
    generate_tokens = model.generate_cached if arguments.cached else model.generate
    generated = generate_tokens(
        prompt,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        generator=generator,
        eos_token_id=tokenizer.eos_token_id,
    )
    token_ids = [int(generated[0, index]) for index in range(generated.shape[1])]
    _ = stdout.write(tokenizer.decode(token_ids) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
