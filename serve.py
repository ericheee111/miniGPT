"""Run the optional miniGPT HTTP completion service."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import torch
import uvicorn

from minigpt.checkpoint import load_checkpoint_config, load_model_state
from minigpt.data import CharTokenizer
from minigpt.engine_runner import EngineRunner, RunnerConfig
from minigpt.http_server import MODEL_ID, create_app
from minigpt.model import GPT
from minigpt.paged_kv_cache import (
    KVCacheBackend,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PrefixCacheNamespace,
)
from minigpt.serving import (
    ContinuousDecodeExecutor,
    ContinuousExecutor,
    EngineConfig,
    PagedAttentionExecutor,
    ReferenceExecutor,
    SchedulerConfig,
    ServingEngine,
    ServingExecutor,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastapi import FastAPI

ExecutorName = Literal["reference", "continuous_decode", "continuous", "paged_attention"]
_EXECUTOR_CHOICES: tuple[ExecutorName, ...] = (
    "reference",
    "continuous_decode",
    "continuous",
    "paged_attention",
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

    max_active_requests = cast("int", arguments.max_active_requests)
    configured_cached_tokens = cast("int | None", arguments.max_cached_tokens)
    max_cached_tokens = (
        max_active_requests * experiment.data.block_size
        if configured_cached_tokens is None
        else configured_cached_tokens
    )
    kv_cache_backend = KVCacheBackend(cast("str", arguments.kv_cache_backend))
    executor_name = cast("ExecutorName", arguments.executor)
    prefix_cache_enabled = cast("bool", arguments.prefix_cache)
    if prefix_cache_enabled and (
        kv_cache_backend is not KVCacheBackend.PAGED or executor_name != "paged_attention"
    ):
        reason = "--prefix-cache requires --executor paged_attention --kv-cache-backend paged"
        raise ValueError(reason)
    paged_kv_cache: PagedKVCacheConfig | None = None
    paged_cache_pool: PagedKVCachePool | None = None
    if kv_cache_backend is KVCacheBackend.PAGED:
        block_tokens = cast("int", arguments.kv_block_tokens)
        configured_num_blocks = cast("int | None", arguments.kv_num_blocks)
        num_blocks = (
            math.ceil(max_cached_tokens / block_tokens)
            if configured_num_blocks is None and block_tokens > 0
            else configured_num_blocks
        )
        if num_blocks is None:
            reason = "--kv-num-blocks is required when --kv-block-tokens is not positive"
            raise ValueError(reason)
        paged_kv_cache = PagedKVCacheConfig(
            block_tokens=block_tokens,
            num_blocks=num_blocks,
        )
        namespace = (
            PrefixCacheNamespace(
                model_checkpoint_identity=_file_sha256(checkpoint_path),
                model_config_identity=_document_sha256(asdict(model.config)),
                dtype=str(model.token_embedding.weight.dtype),
                device=str(model.token_embedding.weight.device),
                block_tokens=block_tokens,
                cache_schema_version=1,
                position_embedding_semantics="learned_absolute_v1",
            )
            if prefix_cache_enabled
            else None
        )
        paged_cache_pool = PagedKVCachePool.from_model(
            paged_kv_cache,
            model,
            prefix_cache_namespace=namespace,
        )
    executor = _executor(executor_name, model, paged_cache_pool=paged_cache_pool)
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(
                max_active_requests=max_active_requests,
                max_cached_tokens=max_cached_tokens,
            ),
            block_size=experiment.data.block_size,
            kv_cache_backend=kv_cache_backend,
            paged_kv_cache=paged_kv_cache,
        ),
        executor=executor,
        paged_cache_pool=paged_cache_pool,
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


def _executor(
    name: ExecutorName,
    model: GPT,
    *,
    paged_cache_pool: PagedKVCachePool | None,
) -> ServingExecutor:
    if name == "reference":
        return ReferenceExecutor(model)
    if name == "continuous_decode":
        return ContinuousDecodeExecutor(model)
    if name == "continuous":
        return ContinuousExecutor(model)
    if paged_cache_pool is None:
        reason = "--executor paged_attention requires --kv-cache-backend paged"
        raise ValueError(reason)
    return PagedAttentionExecutor(model, paged_cache_pool)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _document_sha256(document: object) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
