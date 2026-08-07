"""Run the optional miniGPT HTTP completion service."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import torch
import uvicorn

from minigpt.checkpoint import load_checkpoint_config, load_model_state
from minigpt.data import CharTokenizer
from minigpt.engine_runner import EngineRunner, RunnerConfig
from minigpt.http_server import MODEL_ID, create_app
from minigpt.model import GPT
from minigpt.serving import (
    ContinuousDecodeExecutor,
    ContinuousExecutor,
    EngineConfig,
    ReferenceExecutor,
    SchedulerConfig,
    ServingEngine,
    ServingExecutor,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastapi import FastAPI

ExecutorName = Literal["reference", "continuous_decode", "continuous"]
_EXECUTOR_CHOICES: tuple[ExecutorName, ...] = (
    "reference",
    "continuous_decode",
    "continuous",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the HTTP serving command-line parser."""
    parser = argparse.ArgumentParser(description="Serve miniGPT text completions over HTTP.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--executor", choices=_EXECUTOR_CHOICES, default="continuous")
    parser.add_argument("--max-active-requests", type=int, default=8)
    parser.add_argument("--max-cached-tokens", type=int)
    parser.add_argument("--command-queue-size", type=int, default=256)
    parser.add_argument("--stream-buffer-size", type=int, default=64)
    parser.add_argument(
        "--log-level", choices=("critical", "error", "warning", "info"), default="info"
    )
    return parser


def build_app(arguments: argparse.Namespace) -> FastAPI:
    """Load immutable service resources once and construct the ASGI app."""
    app, _runner = build_runtime(arguments)
    return app


def build_runtime(arguments: argparse.Namespace) -> tuple[FastAPI, EngineRunner]:
    """Load immutable resources and return the app plus its observable runner."""
    checkpoint_path = cast("Path", arguments.checkpoint)
    tokenizer_path = cast("Path", arguments.tokenizer)
    tokenizer = CharTokenizer.load(tokenizer_path)
    experiment = load_checkpoint_config(checkpoint_path).resolve_vocab_size(tokenizer.vocab_size)
    if experiment.runtime.device != "cpu":
        reason = "Stage 12 HTTP serving supports CPU checkpoints only"
        raise ValueError(reason)
    torch.set_num_threads(experiment.runtime.num_threads)
    model = GPT(experiment.model.to_gpt_config(experiment.data.block_size))
    load_model_state(checkpoint_path, model)
    _ = model.eval()

    executor_name = cast("ExecutorName", arguments.executor)
    executor = _executor(executor_name, model)
    max_active_requests = cast("int", arguments.max_active_requests)
    configured_cached_tokens = cast("int | None", arguments.max_cached_tokens)
    max_cached_tokens = (
        max_active_requests * experiment.data.block_size
        if configured_cached_tokens is None
        else configured_cached_tokens
    )
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(
                max_active_requests=max_active_requests,
                max_cached_tokens=max_cached_tokens,
            ),
            block_size=experiment.data.block_size,
        ),
        executor=executor,
    )
    runner = EngineRunner(
        engine=engine,
        config=RunnerConfig(
            command_queue_size=cast("int", arguments.command_queue_size),
            stream_buffer_size=cast("int", arguments.stream_buffer_size),
        ),
    )
    return (
        create_app(
            runner=runner,
            tokenizer=tokenizer,
            model_id=MODEL_ID,
            block_size=experiment.data.block_size,
        ),
        runner,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Load one model and run Uvicorn until graceful process shutdown."""
    arguments = build_parser().parse_args(argv)
    app = build_app(arguments)
    uvicorn.run(
        app,
        host=cast("str", arguments.host),
        port=cast("int", arguments.port),
        log_level=cast("str", arguments.log_level),
    )
    return 0


def _executor(name: ExecutorName, model: GPT) -> ServingExecutor:
    if name == "reference":
        return ReferenceExecutor(model)
    if name == "continuous_decode":
        return ContinuousDecodeExecutor(model)
    return ContinuousExecutor(model)


if __name__ == "__main__":
    raise SystemExit(main())
