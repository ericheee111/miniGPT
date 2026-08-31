from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import httpx
import numpy as np
import pytest
import serve
import torch

from minigpt import public_demo
from minigpt.batching import TokenBatcher
from minigpt.checkpoint import (
    CheckpointResources,
    compute_dataset_fingerprints,
    save_checkpoint,
)
from minigpt.data import CharTokenizer, JsonValue
from minigpt.model import GPT
from minigpt.settings import (
    DataSettings,
    ExperimentConfig,
    ModelSettings,
    OptimizerSettings,
    RuntimeSettings,
    TrainingSettings,
)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows is the canonical subprocess runtime")
@pytest.mark.parametrize(
    ("executor", "kv_cache_backend", "prefix_cache_mode"),
    [
        ("continuous", "dense", "disabled"),
        ("continuous", "paged", "disabled"),
        ("paged_attention", "paged", "disabled"),
        ("paged_attention", "paged", "enabled"),
    ],
)
def test_serve_cli_starts_uvicorn_and_exits_on_localhost(
    tmp_path: Path,
    executor: str,
    kv_cache_backend: str,
    prefix_cache_mode: str,
) -> None:
    # Given: a complete tiny checkpoint/tokenizer pair and a loopback-only free port.
    checkpoint_path, tokenizer_path = _write_service_checkpoint(tmp_path)
    port = _free_port()
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(project_root / "serve.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--tokenizer",
        str(tokenizer_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--executor",
        executor,
        "--kv-cache-backend",
        kv_cache_backend,
        "--log-level",
        "warning",
    ]
    if prefix_cache_mode == "enabled":
        command.append("--prefix-cache")

    # When: Uvicorn starts in a separate process and serves real localhost requests.
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(process, port)
        response = httpx.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json={
                "model": "minigpt-char",
                "prompt": "A",
                "max_tokens": 3,
                "temperature": 1.0,
                "stream": False,
                "seed": 42,
            },
            timeout=5.0,
        )
        body = cast("JsonValue", response.json())

        # Then: the subprocess returns a valid completion and can be stopped without hanging.
        assert response.status_code == 200
        assert isinstance(body, dict)
        assert body.get("object") == "text_completion"
    finally:
        process.terminate()
        _ = process.wait(timeout=10.0)
    assert process.returncode is not None


def test_build_runtime_wires_real_lazy_scheduler_and_manifest(tmp_path: Path) -> None:
    # Given: a real checkpoint/tokenizer and Stage 19 lazy serving options.
    checkpoint_path, tokenizer_path = _write_service_checkpoint(tmp_path)
    manifest_path = tmp_path / "runtime-manifest.json"
    arguments = serve.build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--tokenizer",
            str(tokenizer_path),
            "--executor",
            "paged_attention",
            "--kv-cache-backend",
            "paged",
            "--kv-block-tokens",
            "2",
            "--kv-num-blocks",
            "4",
            "--max-active-requests",
            "2",
            "--max-cached-tokens",
            "8",
            "--max-scheduled-tokens",
            "8",
            "--prefill-chunk-tokens",
            "2",
            "--kv-preemption",
            "--lazy-kv-reservation",
            "--kv-overcommit-ratio",
            "2",
            "--runtime-manifest",
            str(manifest_path),
        ]
    )

    # When: the same path used by Uvicorn builds an in-process app.
    app, runner = serve.build_runtime(arguments)

    async def _scenario() -> httpx.Response:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://stage19-lazy",
            ) as client,
        ):
            response = await client.post(
                "/v1/completions",
                json={
                    "model": "minigpt-char",
                    "prompt": "A",
                    "max_tokens": 2,
                    "temperature": 1.0,
                    "stream": False,
                    "seed": 1919,
                },
            )
            metrics = runner.metrics()
            assert metrics.lazy_kv_reservation_enabled is True
            return response

    response = asyncio.run(_scenario())

    # Then: the real HTTP lifecycle completes and the manifest proves wiring.
    assert response.status_code == 200
    document = cast(
        "dict[str, object]",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    assert document["executor"] == "paged_attention"
    assert document["kv_cache_backend"] == "paged"
    scheduler = cast("dict[str, object]", document["scheduler"])
    assert scheduler["lazy_kv_reservation"] is True
    assert scheduler["kv_preemption"] is True
    assert scheduler["kv_overcommit_ratio"] == 2.0
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows is the canonical subprocess runtime")
def test_serve_cli_lazy_kv_runtime_starts_and_exits_on_localhost(tmp_path: Path) -> None:
    # Given: a tiny real checkpoint and a lazy-growth paged service.
    checkpoint_path, tokenizer_path = _write_service_checkpoint(tmp_path)
    manifest_path = tmp_path / "localhost-runtime.json"
    port = _free_port()
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(project_root / "serve.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--tokenizer",
        str(tokenizer_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--executor",
        "paged_attention",
        "--kv-cache-backend",
        "paged",
        "--kv-block-tokens",
        "2",
        "--kv-num-blocks",
        "4",
        "--max-active-requests",
        "2",
        "--max-cached-tokens",
        "8",
        "--max-scheduled-tokens",
        "8",
        "--prefill-chunk-tokens",
        "2",
        "--kv-preemption",
        "--lazy-kv-reservation",
        "--kv-overcommit-ratio",
        "2",
        "--runtime-manifest",
        str(manifest_path),
        "--log-level",
        "warning",
    ]

    # When: Uvicorn starts in a separate process and serves one completion.
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(process, port)
        response = httpx.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json={
                "model": "minigpt-char",
                "prompt": "A",
                "max_tokens": 3,
                "temperature": 1.0,
                "stream": False,
                "seed": 1919,
            },
            timeout=5.0,
        )
    finally:
        process.terminate()
        _ = process.wait(timeout=10.0)

    # Then: the real subprocess completes and publishes the portable manifest.
    assert response.status_code == 200
    assert manifest_path.is_file()
    document = cast(
        "dict[str, object]",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    scheduler = cast("dict[str, object]", document["scheduler"])
    assert scheduler["lazy_kv_reservation"] is True
    assert document["executor"] == "paged_attention"


def test_public_demo_build_runtime_uses_existing_runner_and_safe_info(tmp_path: Path) -> None:
    # Given: a tiny local checkpoint and the conservative public demo config.
    checkpoint_path, tokenizer_path = _write_service_checkpoint(tmp_path)
    config_path = _write_public_demo_config(tmp_path, streaming_enabled=True)
    arguments = public_demo.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--tokenizer",
            str(tokenizer_path),
        ]
    )

    # When: the typed builder creates the real ServingRuntime and serves both modes.
    app, runner, server = public_demo.build_runtime(
        arguments,
        environment={
            "DEMO_ENABLED": "1",
            "PUBLIC_ORIGIN": "https://portfolio.example",
        },
    )

    async def _scenario() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://public-demo",
            ) as client,
        ):
            info = await client.get("/demo/info")
            completion = await client.post(
                "/v1/completions",
                json={
                    "model": "minigpt-char",
                    "prompt": "A",
                    "max_tokens": 2,
                    "temperature": 1.0,
                    "stream": False,
                    "seed": 2026,
                },
            )
            stream = await client.post(
                "/v1/completions",
                json={
                    "model": "minigpt-char",
                    "prompt": "A",
                    "max_tokens": 2,
                    "temperature": 1.0,
                    "stream": True,
                    "seed": 2026,
                },
            )
            assert runner.metrics().active_requests == 0
            return info, completion, stream

    info, completion, stream = asyncio.run(_scenario())

    # Then: it remains loopback-only and exposes no local asset paths.
    assert server.host == "127.0.0.1"
    assert info.status_code == 200
    assert completion.status_code == 200
    assert stream.status_code == 200
    assert "data: [DONE]" in stream.text
    info_text = info.text
    assert str(checkpoint_path) not in info_text
    assert str(tokenizer_path) not in info_text
    assert '"executor":"continuous"' in info_text
    assert '"kv_cache_backend":"dense"' in info_text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows is the canonical subprocess runtime")
def test_public_demo_cli_serves_real_localhost_completion_and_stream(tmp_path: Path) -> None:
    # Given: a real tiny checkpoint, a free loopback port, and the explicit kill switch enabled.
    checkpoint_path, tokenizer_path = _write_service_checkpoint(tmp_path)
    config_path = _write_public_demo_config(tmp_path, streaming_enabled=True)
    port = _free_port()
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "minigpt",
        "demo-serve",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--tokenizer",
        str(tokenizer_path),
        "--port",
        str(port),
    ]
    environment = {
        **os.environ,
        "DEMO_ENABLED": "1",
        "PUBLIC_ORIGIN": "https://portfolio.example",
    }

    # When: the restricted CLI starts and receives health, completion, and SSE traffic.
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=project_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(process, port)
        completion = httpx.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json={
                "model": "minigpt-char",
                "prompt": "A",
                "max_tokens": 3,
                "temperature": 1.0,
                "stream": False,
                "seed": 42,
            },
            timeout=5.0,
        )
        stream = httpx.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json={
                "model": "minigpt-char",
                "prompt": "A",
                "max_tokens": 3,
                "temperature": 1.0,
                "stream": True,
                "seed": 42,
            },
            timeout=5.0,
        )
    finally:
        process.terminate()
        _ = process.wait(timeout=10.0)

    # Then: both public modes succeed and the process exits cleanly without a public bind.
    assert completion.status_code == 200
    assert stream.status_code == 200
    assert "data: [DONE]" in stream.text
    assert process.returncode is not None


def _write_public_demo_config(tmp_path: Path, *, streaming_enabled: bool) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "configs" / "public_demo.yaml").read_text(encoding="utf-8")
    expected = "streaming_enabled: true"
    assert source.count(expected) == 1
    configured = source.replace(
        expected,
        f"streaming_enabled: {str(streaming_enabled).lower()}",
    )
    path = tmp_path / "public_demo.yaml"
    _ = path.write_text(configured, encoding="utf-8")
    return path


def _write_service_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    tokenizer = CharTokenizer.from_text("AB")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    tokens = np.arange(32, dtype=np.uint16) % tokenizer.vocab_size
    _ = (tmp_path / "train.npy").write_bytes(b"tiny-train")
    _ = (tmp_path / "val.npy").write_bytes(b"tiny-validation")
    config = ExperimentConfig(
        runtime=RuntimeSettings(seed=7, num_threads=1, device="cpu"),
        data=DataSettings(directory=tmp_path, block_size=8, batch_size=2),
        model=ModelSettings(
            vocab_size=tokenizer.vocab_size,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        ),
        optimizer=OptimizerSettings(
            optimizer_type="adamw",
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            grad_clip=1.0,
        ),
        training=TrainingSettings(
            max_steps=2,
            warmup_steps=1,
            lr_decay_steps=2,
            eval_interval=1,
            eval_batches=1,
            log_interval=1,
            checkpoint_interval=1,
            sample_interval=1,
            sample_tokens=2,
            sample_prompt="A",
            output_dir=tmp_path / "outputs",
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        ),
    )
    _ = torch.default_generator.manual_seed(1234)
    model = GPT(config.model.to_gpt_config(config.data.block_size))
    sample_generator = torch.Generator(device="cpu")
    _ = sample_generator.manual_seed(config.runtime.seed + 2)
    resources = CheckpointResources(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        train_batcher=TokenBatcher(tokens, batch_size=2, block_size=8, seed=1),
        val_batcher=TokenBatcher(tokens, batch_size=2, block_size=8, seed=2),
        sample_generator=sample_generator,
        dataset_fingerprints=compute_dataset_fingerprints(config.data),
    )
    checkpoint_path = tmp_path / "service.pt"
    save_checkpoint(checkpoint_path, resources=resources, step=1, config=config)
    return checkpoint_path, tokenizer_path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return cast("int", listener.getsockname()[1])


def _wait_for_health(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.perf_counter() + 15.0
    health_url = f"http://127.0.0.1:{port}/healthz"
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            message = f"serve.py exited before health check with code {process.returncode}"
            raise AssertionError(message)
        try:
            if httpx.get(health_url, timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    message = "serve.py did not become healthy"
    raise AssertionError(message)
