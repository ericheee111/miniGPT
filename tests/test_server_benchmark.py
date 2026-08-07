from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import cast

import httpx
import torch
import uvicorn

from minigpt.data import CharTokenizer
from minigpt.engine_runner import EngineRunner, RunnerConfig
from minigpt.http_server import MODEL_ID, create_app
from minigpt.model import GPT
from minigpt.server_benchmark import HTTPBenchmarkConfig, run_http_benchmark
from minigpt.serving import (
    ContinuousExecutor,
    EngineConfig,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig


def test_http_benchmark_covers_fixed_matrix_and_engine_metrics() -> None:
    # Given: a real localhost service with a tiny model covering benchmark prompts.
    tokenizer = CharTokenizer.from_text("ROMEO:JULIET\nFirst Citizen")
    _ = torch.default_generator.manual_seed(712)
    model = GPT(
        GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=32,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
        )
    ).eval()
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=8, max_cached_tokens=256),
            block_size=32,
        ),
        executor=ContinuousExecutor(model),
    )
    runner = EngineRunner(
        engine=engine,
        config=RunnerConfig(command_queue_size=128, stream_buffer_size=16),
    )
    app = create_app(runner=runner, tokenizer=tokenizer, model_id=MODEL_ID, block_size=32)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, name="benchmark-test")
    thread.start()
    _wait_for_server(server, port)

    try:
        # When: the end-to-end HTTP matrix runs with two generated tokens per request.
        result = asyncio.run(
            run_http_benchmark(
                HTTPBenchmarkConfig(
                    base_url=f"http://127.0.0.1:{port}",
                    requests_per_case=1,
                    max_tokens=2,
                    timeout_seconds=10.0,
                )
            )
        ).with_engine_metrics(runner.metrics())
        document = result.to_document()

        # Then: all 16 cells and their concurrency-expanded requests are successful.
        assert len(result.cases) == 16
        assert len(result.measurements) == 60
        assert all(case.http_error_count == 0 for case in result.cases)
        assert all(case.e2e_seconds.p50 is not None for case in result.cases)
        assert all(case.ttft_seconds.p95 is not None for case in result.cases)
        assert result.engine is not None
        assert result.engine.total_requests == 60
        assert result.engine.peak_active_requests >= 1
        assert result.engine.decode_batch_sizes
        assert result.engine.prefill_batch_sizes
        assert document.get("benchmark_scope") == "http_end_to_end"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
    assert not thread.is_alive()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return cast("int", listener.getsockname()[1])


def _wait_for_server(server: uvicorn.Server, port: int) -> None:
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        if server.started:
            response = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
            if response.status_code == 200:
                return
        time.sleep(0.01)
    message = "benchmark test server did not start"
    raise AssertionError(message)
