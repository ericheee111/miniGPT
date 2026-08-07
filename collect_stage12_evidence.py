"""Collect real localhost API examples and lifecycle evidence for Stage 12."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import uvicorn
from serve import build_runtime

from minigpt.data import JsonValue

if TYPE_CHECKING:
    from collections.abc import Sequence

EvidenceDocument = dict[str, JsonValue]
_MODEL_ID = "minigpt-char"


def build_parser() -> argparse.ArgumentParser:
    """Build the evidence collector command-line parser."""
    parser = argparse.ArgumentParser(description="Collect Stage 12 localhost evidence.")
    _ = parser.add_argument("--checkpoint", type=Path, required=True)
    _ = parser.add_argument("--tokenizer", type=Path, required=True)
    _ = parser.add_argument("--output-dir", type=Path, default=Path("reports/stage12-evidence"))
    return parser


def _write_json(path: Path, document: EvidenceDocument) -> None:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return cast("int", listener.getsockname()[1])


def _runtime_arguments(arguments: argparse.Namespace, port: int) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=arguments.checkpoint,
        tokenizer=arguments.tokenizer,
        host="127.0.0.1",
        port=port,
        executor="continuous",
        max_active_requests=8,
        max_cached_tokens=None,
        command_queue_size=256,
        stream_buffer_size=64,
        log_level="warning",
    )


def _wait_until_started(server: uvicorn.Server, thread: threading.Thread) -> None:
    deadline = time.monotonic() + 15.0
    while not server.started:
        if not thread.is_alive():
            msg = "Uvicorn stopped before accepting connections"
            raise RuntimeError(msg)
        if time.monotonic() >= deadline:
            msg = "Uvicorn did not start within 15 seconds"
            raise TimeoutError(msg)
        time.sleep(0.01)


def _completion_payload(*, stream: bool, seed: int = 42) -> EvidenceDocument:
    return {
        "model": _MODEL_ID,
        "prompt": "ROMEO:",
        "max_tokens": 8,
        "temperature": 1.0,
        "stream": stream,
        "seed": seed,
    }


def _response_document(response: httpx.Response) -> EvidenceDocument:
    raw = cast("object", response.json())
    if not isinstance(raw, dict):
        msg = "HTTP completion response was not a JSON object"
        raise TypeError(msg)
    return cast("EvidenceDocument", raw)


def _stream_lines(client: httpx.Client, url: str) -> list[str]:
    with client.stream("POST", url, json=_completion_payload(stream=True)) as response:
        response.raise_for_status()
        return [line for line in response.iter_lines() if line]


def _choice_text(document: EvidenceDocument) -> str:
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        msg = "completion response omitted choices"
        raise ValueError(msg)
    text = cast("dict[str, object]", choices[0]).get("text")
    if not isinstance(text, str):
        msg = "completion response choice omitted text"
        raise TypeError(msg)
    return text


def _stream_text(lines: Sequence[str]) -> str:
    pieces: list[str] = []
    for line in lines:
        if line == "data: [DONE]":
            continue
        if not line.startswith("data: "):
            msg = f"unexpected SSE line: {line}"
            raise ValueError(msg)
        document = cast("object", json.loads(line.removeprefix("data: ")))
        if not isinstance(document, dict):
            msg = "SSE payload was not an object"
            raise TypeError(msg)
        pieces.append(_choice_text(cast("EvidenceDocument", document)))
    return "".join(pieces)


def _collect_api(arguments: argparse.Namespace) -> EvidenceDocument:
    port = _free_port()
    app, _runner = build_runtime(_runtime_arguments(arguments, port))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="stage12-evidence-uvicorn")
    thread.start()
    _wait_until_started(server, thread)
    base_url = f"http://127.0.0.1:{port}"
    completion_url = f"{base_url}/v1/completions"
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(completion_url, json=_completion_payload(stream=False))
            response.raise_for_status()
            non_stream = _response_document(response)
            stream_lines = _stream_lines(client, completion_url)
            if _stream_text(stream_lines) != _choice_text(non_stream):
                msg = "streaming token concatenation differs from non-stream completion"
                raise RuntimeError(msg)

        def submit(seed: int) -> tuple[int, str]:
            with httpx.Client(timeout=30.0) as concurrent_client:
                result = concurrent_client.post(
                    completion_url,
                    json=_completion_payload(stream=False, seed=seed),
                )
                return result.status_code, _choice_text(_response_document(result))

        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent_results = list(pool.map(submit, (7, 7, 8, 9)))
        if concurrent_results[0][1] != concurrent_results[1][1]:
            msg = "equal concurrent seeds did not produce equal outputs"
            raise RuntimeError(msg)
        return {
            "server": {"host": "127.0.0.1", "executor": "continuous"},
            "non_stream": {
                "curl": (
                    "curl -X POST http://127.0.0.1:8000/v1/completions "
                    '-H "Content-Type: application/json" '
                    '-d \'{"model":"minigpt-char","prompt":"ROMEO:",'
                    '"max_tokens":8,"temperature":1.0,"stream":false,"seed":42}\''
                ),
                "response": non_stream,
            },
            "streaming": {
                "curl": (
                    "curl -N -X POST http://127.0.0.1:8000/v1/completions "
                    '-H "Content-Type: application/json" '
                    '-d \'{"model":"minigpt-char","prompt":"ROMEO:",'
                    '"max_tokens":8,"temperature":1.0,"stream":true,"seed":42}\''
                ),
                "sse_lines": stream_lines,
                "concatenated_text": _stream_text(stream_lines),
            },
            "concurrent": {
                "seeds": [7, 7, 8, 9],
                "status_codes": [status for status, _text in concurrent_results],
                "texts": [text for _status, text in concurrent_results],
            },
        }
    finally:
        server.should_exit = True
        thread.join(timeout=15.0)
        if thread.is_alive():
            msg = "Uvicorn did not stop within 15 seconds"
            raise TimeoutError(msg)


def _collect_lifecycle() -> EvidenceDocument:
    command = [
        "python",
        "-m",
        "pytest",
        "tests/test_engine_runner.py",
        "tests/test_http_lifecycle.py",
        "tests/test_serve_subprocess.py",
        "-q",
    ]
    execution_command = [sys.executable, *command[1:]]
    completed = subprocess.run(  # noqa: S603
        execution_command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180.0,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "output": completed.stdout,
        "stderr": completed.stderr,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Collect live API examples and execute the lifecycle correctness suite."""
    arguments = build_parser().parse_args(argv)
    output_dir = cast("Path", arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "api_examples.json", _collect_api(arguments))
    lifecycle = _collect_lifecycle()
    _write_json(output_dir / "lifecycle.json", lifecycle)
    print(f"output={output_dir}")  # noqa: T201
    return cast("int", lifecycle["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
