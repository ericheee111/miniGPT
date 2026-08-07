"""Run the independent end-to-end HTTP serving benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, cast

import httpx
import uvicorn
from serve import build_runtime

from minigpt.server_benchmark import HTTPBenchmarkConfig, run_http_benchmark

_HTTP_OK = 200

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the HTTP benchmark command-line parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark API, serialization, queues, scheduler, and serving engine.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--executor",
        choices=("reference", "continuous_decode", "continuous"),
        default="continuous",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--requests-per-case", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-active-requests", type=int, default=8)
    parser.add_argument("--max-cached-tokens", type=int)
    parser.add_argument("--command-queue-size", type=int, default=256)
    parser.add_argument("--stream-buffer-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Start one localhost service, execute the matrix, and persist JSON."""
    arguments = build_parser().parse_args(argv)
    host = cast("str", arguments.host)
    configured_port = cast("int", arguments.port)
    port = _free_port(host) if configured_port == 0 else configured_port
    service_arguments = argparse.Namespace(
        checkpoint=arguments.checkpoint,
        tokenizer=arguments.tokenizer,
        executor=arguments.executor,
        max_active_requests=arguments.max_active_requests,
        max_cached_tokens=arguments.max_cached_tokens,
        command_queue_size=arguments.command_queue_size,
        stream_buffer_size=arguments.stream_buffer_size,
    )
    app, runner = build_runtime(service_arguments)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    server_thread = threading.Thread(target=server.run, name="minigpt-http-benchmark")
    server_thread.start()
    try:
        _wait_for_server(server, host, port)
        result = asyncio.run(
            run_http_benchmark(
                HTTPBenchmarkConfig(
                    base_url=f"http://{host}:{port}",
                    requests_per_case=cast("int", arguments.requests_per_case),
                    max_tokens=cast("int", arguments.max_tokens),
                    timeout_seconds=cast("float", arguments.timeout_seconds),
                )
            )
        ).with_engine_metrics(runner.metrics())
        output = cast("Path", arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        _ = output.write_text(
            json.dumps(result.to_document(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _ = stdout.write(f"{output}\n")
    finally:
        server.should_exit = True
        server_thread.join(timeout=30.0)
    if server_thread.is_alive():
        message = "HTTP benchmark Uvicorn server did not stop"
        raise RuntimeError(message)
    return 0


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return cast("int", listener.getsockname()[1])


def _wait_for_server(server: uvicorn.Server, host: str, port: int) -> None:
    deadline = time.perf_counter() + 15.0
    url = f"http://{host}:{port}/healthz"
    while time.perf_counter() < deadline:
        if server.started:
            try:
                if httpx.get(url, timeout=1.0).status_code == _HTTP_OK:
                    return
            except httpx.HTTPError:
                pass
        time.sleep(0.05)
    message = "HTTP benchmark Uvicorn server did not become healthy"
    raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
