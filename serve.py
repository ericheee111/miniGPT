"""Run the optional miniGPT HTTP completion service."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import uvicorn

from minigpt.checkpoint import load_checkpoint_config, load_model_state
from minigpt.data import CharTokenizer
from minigpt.http_server import MODEL_ID, create_app
from minigpt.model import GPT
from minigpt.paged_kv_cache import KVCacheBackend
from minigpt.serving import APCPrefillStrategy
from minigpt.serving_runtime import (
    ServingExecutorName,
    ServingRuntimeConfig,
    build_serving_runtime,
    file_sha256,
    write_runtime_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastapi import FastAPI

    from minigpt.engine_runner import EngineRunner

_EXECUTOR_CHOICES: tuple[ServingExecutorName, ...] = (
    ServingExecutorName.REFERENCE,
    ServingExecutorName.CONTINUOUS_DECODE,
    ServingExecutorName.CONTINUOUS,
    ServingExecutorName.PAGED_ATTENTION,
)
_APC_STRATEGY_CHOICES: tuple[APCPrefillStrategy, ...] = (
    APCPrefillStrategy.SEQUENTIAL,
    APCPrefillStrategy.BATCHED,
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
    parser.add_argument(
        "--kv-cache-backend",
        choices=tuple(backend.value for backend in KVCacheBackend),
        default=KVCacheBackend.DENSE.value,
    )
    parser.add_argument("--kv-block-tokens", type=int, default=16)
    parser.add_argument("--kv-num-blocks", type=int)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--max-scheduled-tokens", type=int)
    parser.add_argument("--prefill-chunk-tokens", type=int)
    parser.add_argument("--kv-preemption", action="store_true")
    parser.add_argument("--lazy-kv-reservation", action="store_true")
    parser.add_argument("--kv-overcommit-ratio", type=float, default=1.0)
    parser.add_argument(
        "--apc-prefill-strategy",
        choices=tuple(strategy.value for strategy in _APC_STRATEGY_CHOICES),
        default=APCPrefillStrategy.SEQUENTIAL.value,
    )
    parser.add_argument("--runtime-manifest", type=Path)
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
    _ = torch.set_num_threads(experiment.runtime.num_threads)
    model = GPT(experiment.model.to_gpt_config(experiment.data.block_size))
    load_model_state(checkpoint_path, model)
    _ = model.eval()

    max_active_requests = cast("int", arguments.max_active_requests)
    configured_cached_tokens = cast("int | None", arguments.max_cached_tokens)
    max_cached_tokens = (
        max_active_requests * experiment.data.block_size
        if configured_cached_tokens is None
        else configured_cached_tokens
    )
    config = ServingRuntimeConfig(
        executor=ServingExecutorName(cast("str", arguments.executor)),
        max_active_requests=max_active_requests,
        max_cached_tokens=max_cached_tokens,
        kv_cache_backend=KVCacheBackend(cast("str", arguments.kv_cache_backend)),
        kv_block_tokens=cast("int", arguments.kv_block_tokens),
        kv_num_blocks=cast("int | None", arguments.kv_num_blocks),
        prefix_cache=cast("bool", arguments.prefix_cache),
        max_scheduled_tokens=cast("int | None", arguments.max_scheduled_tokens),
        prefill_chunk_tokens=cast("int | None", arguments.prefill_chunk_tokens),
        kv_preemption=cast("bool", arguments.kv_preemption),
        lazy_kv_reservation=cast("bool", arguments.lazy_kv_reservation),
        kv_overcommit_ratio=cast("float", arguments.kv_overcommit_ratio),
        apc_prefill_strategy=APCPrefillStrategy(cast("str", arguments.apc_prefill_strategy)),
        command_queue_size=cast("int", arguments.command_queue_size),
        stream_buffer_size=cast("int", arguments.stream_buffer_size),
    )
    runtime = build_serving_runtime(
        model=model,
        block_size=experiment.data.block_size,
        num_threads=experiment.runtime.num_threads,
        checkpoint_sha256=file_sha256(checkpoint_path),
        tokenizer_sha256=file_sha256(tokenizer_path),
        config=config,
    )
    manifest_path = cast("Path | None", getattr(arguments, "runtime_manifest", None))
    if manifest_path is not None:
        _ = write_runtime_manifest(manifest_path, runtime.manifest)
    return (
        create_app(
            runner=runtime.runner,
            tokenizer=tokenizer,
            model_id=MODEL_ID,
            block_size=experiment.data.block_size,
        ),
        runtime.runner,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Load one model and run Uvicorn until graceful process shutdown."""
    arguments = build_parser().parse_args(argv)
    app = build_app(arguments)
    _ = uvicorn.run(
        app,
        host=cast("str", arguments.host),
        port=cast("int", arguments.port),
        log_level=cast("str", arguments.log_level),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
