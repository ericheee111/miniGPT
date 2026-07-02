import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from minigpt.checkpoint import load_checkpoint_config, load_model_state
from minigpt.data import CharTokenizer
from minigpt.model import GPT


def build_parser() -> argparse.ArgumentParser:
    """Create the text-generation command-line parser."""
    parser = argparse.ArgumentParser(description="Generate text from a MiniTrainGPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load a checkpoint and sample text from its persisted configuration."""
    arguments = build_parser().parse_args(argv)
    config = load_checkpoint_config(arguments.checkpoint)
    tokenizer = CharTokenizer.load(config.data.tokenizer_path)
    model = GPT(config.model.to_gpt_config(config.data.block_size))
    load_model_state(arguments.checkpoint, model)
    model.eval()
    seed = config.runtime.seed if arguments.seed is None else arguments.seed
    torch.manual_seed(seed)
    prompt = torch.tensor([tokenizer.encode(arguments.prompt)], dtype=torch.long)
    generated = model.generate(
        prompt,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
    )
    print(tokenizer.decode(generated[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
