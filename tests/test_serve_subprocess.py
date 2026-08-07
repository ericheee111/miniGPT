from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import httpx
import numpy as np
import pytest
import torch

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
def test_serve_cli_starts_uvicorn_and_exits_on_localhost(tmp_path: Path) -> None:
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
        "continuous",
        "--log-level",
        "warning",
    ]

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
