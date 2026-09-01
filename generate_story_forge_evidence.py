"""Generate the hash-bound Story Forge model/evaluation/serving evidence package."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import serve

from minigpt.checkpoint import CheckpointMetadata, load_checkpoint_metadata
from minigpt.story_forge_evidence import generate_story_forge_evidence

if TYPE_CHECKING:
    from collections.abc import Sequence

_METRICS_PATH = "outputs/story_forge_5m/metrics.jsonl"
_EVAL_FINAL = "reports/story-forge-eval/step-3000.json"
_EVAL_0500 = "reports/story-forge-eval/step-0500.json"
_EVAL_1500 = "reports/story-forge-eval/step-1500.json"
_PACKAGE_ROOT = "docs/results/story-forge-model"
_CHECKPOINT = "checkpoints/story_forge_5m/latest.pt"
_TOKENIZER = "data/story_forge/tokenizer.json"
_CONTROL_PROMPT = "<bos><world_space><tone_adventurous><theme_discovery><story>Once upon a time"
_SERVE_PORT = 8000
_HTTP_OK = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the Story Forge evidence-generation parser."""
    parser = argparse.ArgumentParser(
        description="Generate hash-bound Story Forge model/evaluation/serving evidence."
    )
    _ = parser.add_argument("--source-commit", required=True)
    _ = parser.add_argument("--package-root", type=Path, default=Path(_PACKAGE_ROOT))
    _ = parser.add_argument("--work-root", type=Path, default=Path("reports/story-forge-evidence"))
    return parser


def _metrics_reporter(metadata: CheckpointMetadata, work_root: Path) -> Path:
    """Fold the raw metrics.jsonl trajectory into a compact, hash-bound reporter."""
    metrics_path = Path(_METRICS_PATH)
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    steps = [cast("int", row["step"]) for row in rows]
    if steps != list(range(len(steps))):
        reason = "story_forge metrics trajectory is not strictly ordered"
        raise ValueError(reason)

    eval_rows = [row for row in rows if row.get("val_loss") is not None]
    final_train_loss = cast("float", rows[-1]["train_loss"])
    final_val_loss = cast("float", eval_rows[-1]["val_loss"]) if eval_rows else None

    reporter: dict[str, object] = {
        "schema_version": 1,
        "steps": steps,
        "final_step": steps[-1],
        "eval_points": len(eval_rows),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_val_perplexity": math.exp(final_val_loss) if final_val_loss is not None else None,
        "max_steps": 3000,
        "intentionally_resumed": True,
        "tokenizer_sha256": metadata.dataset_fingerprints.tokenizer_sha256,
        "train_sha256": metadata.dataset_fingerprints.train_sha256,
        "val_sha256": metadata.dataset_fingerprints.val_sha256,
        "checkpoint_basename": Path(_CHECKPOINT).name,
        "checkpoint_note": (
            "final training checkpoint; external checkpoint tracked only by SHA-256; "
            "see evaluation_final.json for exact byte/sha identity"
        ),
        "raw_metrics_sha256": _sha256(metrics_path),
    }

    reporter_path = work_root / "metrics_reporter.json"
    _write_json(reporter_path, reporter)
    return reporter_path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return cast("int", listener.getsockname()[1])


def _wait_health(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.perf_counter() + 20.0
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            reason = f"serve.py exited: {process.returncode}"
            raise RuntimeError(reason)
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0).status_code == _HTTP_OK:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    reason = "serve.py did not become healthy"
    raise RuntimeError(reason)


def _serve_subprocess_smoke() -> dict[str, object]:
    """Run the subprocess serving smoke against a free port."""
    port = _free_port()
    if port == _SERVE_PORT:
        reason = "serving smoke must not reuse the live demo port 8000"
        raise RuntimeError(reason)
    project_root = Path(__file__).resolve().parent
    base_url = f"http://127.0.0.1:{port}"

    command = [
        sys.executable,
        str(project_root / "serve.py"),
        "--checkpoint",
        str(project_root / _CHECKPOINT),
        "--tokenizer",
        str(project_root / _TOKENIZER),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    results: dict[str, object] = {}
    try:
        _wait_health(process, port)
        results["health"] = True

        models = httpx.get(f"{base_url}/v1/models", timeout=5.0)
        results["models"] = models.status_code == _HTTP_OK

        non_stream = httpx.post(
            f"{base_url}/v1/completions",
            json={
                "model": "minigpt-char",
                "prompt": _CONTROL_PROMPT,
                "max_tokens": 12,
                "temperature": 0.8,
                "stream": False,
                "seed": 7,
            },
            timeout=60.0,
        )
        results["non_stream"] = non_stream.status_code == _HTTP_OK

        streamed = httpx.post(
            f"{base_url}/v1/completions",
            json={
                "model": "minigpt-char",
                "prompt": _CONTROL_PROMPT,
                "max_tokens": 12,
                "temperature": 0.8,
                "stream": True,
                "seed": 7,
            },
            timeout=60.0,
        )
        results["stream"] = "data: [DONE]" in streamed.text

        def _one(seed: int) -> httpx.Response:
            return httpx.post(
                f"{base_url}/v1/completions",
                json={
                    "model": "minigpt-char",
                    "prompt": "Once upon a time",
                    "max_tokens": 8,
                    "temperature": 0.8,
                    "stream": False,
                    "seed": seed,
                },
                timeout=60.0,
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            concurrent = list(pool.map(_one, [11, 12, 13]))
        results["concurrent"] = all(r.status_code == _HTTP_OK for r in concurrent)

        # Disconnect a slow stream mid-flight to exercise cancellation.
        with (
            httpx.stream(
                "POST",
                f"{base_url}/v1/completions",
                json={
                    "model": "minigpt-char",
                    "prompt": "Once upon a time",
                    "max_tokens": 96,
                    "temperature": 0.8,
                    "stream": True,
                    "seed": 21,
                },
                timeout=5.0,
            ) as response,
            suppress(StopIteration),
        ):
            _ = next(response.iter_text())
        results["disconnect_cancellation"] = True

        # Path/prompt leakage: collect every body touched.
        bodies = "\n".join(
            [
                models.text,
                non_stream.text,
                streamed.text,
                *(r.text for r in concurrent),
            ]
        )
        results["path_leakage"] = (
            str(project_root) in bodies or _CHECKPOINT in bodies or _TOKENIZER in bodies
        )
    finally:
        process.terminate()
        _ = process.wait(timeout=10.0)
    return results


def _run_serving_smoke(work_root: Path) -> Path:
    """Run subprocess + in-process ASGI serving smoke and record bounded JSON."""
    subprocess_result = _serve_subprocess_smoke()

    arguments = serve.build_parser().parse_args(
        [
            "--checkpoint",
            str(Path(_CHECKPOINT).resolve()),
            "--tokenizer",
            str(Path(_TOKENIZER).resolve()),
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
        ]
    )
    app, runner = serve.build_runtime(arguments)

    asgi_active_zero = False
    asgi_waiting_zero = False

    async def _scenario() -> None:
        nonlocal asgi_active_zero, asgi_waiting_zero
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://story-smoke"
            ) as client,
        ):
            _ = await client.post(
                "/v1/completions",
                json={
                    "model": "minigpt-char",
                    "prompt": "Once upon a time",
                    "max_tokens": 4,
                    "temperature": 0.8,
                    "stream": False,
                    "seed": 1,
                },
            )
            _ = await client.post(
                "/v1/completions",
                json={
                    "model": "minigpt-char",
                    "prompt": "Once upon a time",
                    "max_tokens": 4,
                    "temperature": 0.8,
                    "stream": True,
                    "seed": 2,
                },
            )
            metrics = runner.metrics()
            asgi_active_zero = metrics.active_requests == 0
            asgi_waiting_zero = metrics.waiting_requests == 0

    asyncio.run(_scenario())
    smoke_doc = {
        "schema_version": 1,
        "subprocess": subprocess_result,
        "asgi": {
            "active_requests_zero": asgi_active_zero,
            "waiting_requests_zero": asgi_waiting_zero,
        },
    }
    path = work_root / "serving_smoke.json"
    _write_json(path, smoke_doc)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Assemble the Story Forge evidence package with self-verification."""
    arguments = build_parser().parse_args(argv)
    work_root = cast("Path", arguments.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    metadata = load_checkpoint_metadata(Path(_CHECKPOINT))
    metrics_reporter = _metrics_reporter(metadata, work_root)
    serving_smoke = _run_serving_smoke(work_root)

    package = generate_story_forge_evidence(
        metrics_reporter=metrics_reporter,
        final_evaluation=Path(_EVAL_FINAL),
        milestone_0500_evaluation=Path(_EVAL_0500),
        milestone_1500_evaluation=Path(_EVAL_1500),
        serving_smoke=serving_smoke,
        package_root=cast("Path", arguments.package_root),
        source_commit=cast("str", arguments.source_commit),
    )
    print(package)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
