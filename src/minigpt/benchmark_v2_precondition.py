"""Run one explicit, unmeasured Benchmark v2 training-step preconditioning phase."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, cast

import torch

from minigpt.benchmark_v2_environment import apply_cpu_affinity, capture_worker_environment
from minigpt.benchmark_v2_types import BenchmarkV2Case
from minigpt.benchmark_workload import create_training_step_workload

if TYPE_CHECKING:
    from minigpt.benchmark_v2_config import JsonValue

_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "case",
        "case_identity",
        "benchmark_seed",
        "vocab_size",
        "duration_seconds",
        "torch_num_interop_threads",
        "cpu_affinity",
        "relevant_environment_variables",
    }
)
_CASE_KEYS = frozenset(
    {
        "name",
        "model_name",
        "n_layer",
        "n_head",
        "n_embd",
        "torch_num_threads",
        "block_size",
        "batch_size",
    }
)
_SHA256_LENGTH = 64


def _mapping(value: object, expected: frozenset[str], context: str) -> dict[str, object]:
    """Require one exact string-keyed JSON object."""
    if not isinstance(value, dict):
        msg = f"{context} must be an object"
        raise TypeError(msg)
    document = cast("dict[str, object]", value)
    if frozenset(document) != expected:
        msg = f"{context} has an invalid field set"
        raise ValueError(msg)
    return document


def _integer(value: object, context: str, *, positive: bool = False) -> int:
    """Require one strict integer."""
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        msg = f"{context} must be {'positive ' if positive else ''}integer"
        raise ValueError(msg)
    return value


def _case(value: object) -> BenchmarkV2Case:
    """Parse the configured preconditioning case."""
    document = _mapping(value, _CASE_KEYS, "case")
    text_fields = ("name", "model_name")
    if any(not isinstance(document[field], str) or not document[field] for field in text_fields):
        msg = "case text fields must be non-empty"
        raise ValueError(msg)
    return BenchmarkV2Case(
        name=cast("str", document["name"]),
        model_name=cast("str", document["model_name"]),
        n_layer=_integer(document["n_layer"], "n_layer", positive=True),
        n_head=_integer(document["n_head"], "n_head", positive=True),
        n_embd=_integer(document["n_embd"], "n_embd", positive=True),
        torch_num_threads=_integer(
            document["torch_num_threads"],
            "torch_num_threads",
            positive=True,
        ),
        block_size=_integer(document["block_size"], "block_size", positive=True),
        batch_size=_integer(document["batch_size"], "batch_size", positive=True),
    )


def _affinity(value: object) -> tuple[int, ...] | None:
    """Parse optional CPU affinity."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        msg = "cpu_affinity must be a non-empty list or null"
        raise ValueError(msg)
    values = tuple(_integer(item, "cpu_affinity item") for item in cast("list[object]", value))
    if any(item < 0 for item in values) or len(values) != len(set(values)):
        msg = "cpu_affinity must contain unique non-negative IDs"
        raise ValueError(msg)
    return values


def _environment_names(value: object) -> tuple[str, ...]:
    """Parse named environment controls."""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in cast("list[object]", value)
    ):
        msg = "relevant_environment_variables must contain non-empty strings"
        raise ValueError(msg)
    names = tuple(cast("list[str]", value))
    if len(names) != len(set(names)):
        msg = "relevant_environment_variables must not contain duplicates"
        raise ValueError(msg)
    return names


def run_preconditioning_request(request: dict[str, object]) -> dict[str, JsonValue]:
    """Execute training steps until the configured minimum wall duration has elapsed."""
    if request["schema_version"] != 1:
        msg = "schema_version must be 1"
        raise ValueError(msg)
    case = _case(request["case"])
    case_identity = request["case_identity"]
    if not isinstance(case_identity, str) or len(case_identity) != _SHA256_LENGTH:
        msg = "case_identity must be a SHA-256 string"
        raise TypeError(msg)
    duration_value = request["duration_seconds"]
    if isinstance(duration_value, bool) or not isinstance(duration_value, (int, float)):
        msg = "duration_seconds must be positive and finite"
        raise TypeError(msg)
    duration_seconds = float(duration_value)
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        msg = "duration_seconds must be positive and finite"
        raise ValueError(msg)
    affinity = _affinity(request["cpu_affinity"])
    environment_names = _environment_names(request["relevant_environment_variables"])
    started_at_utc = datetime.now(UTC).isoformat()
    effective_affinity = apply_cpu_affinity(affinity)
    torch.set_num_threads(case.torch_num_threads)
    torch.set_num_interop_threads(
        _integer(
            request["torch_num_interop_threads"],
            "torch_num_interop_threads",
            positive=True,
        )
    )
    workload = create_training_step_workload(
        case,
        seed=_integer(request["benchmark_seed"], "benchmark_seed"),
        vocab_size=_integer(request["vocab_size"], "vocab_size", positive=True),
    )
    started = perf_counter()
    completed_steps = 0
    while perf_counter() - started < duration_seconds:
        workload.step()
        completed_steps += 1
    elapsed_seconds = perf_counter() - started
    environment = capture_worker_environment(
        requested_cpu_affinity=affinity,
        effective_cpu_affinity=effective_affinity,
        relevant_environment_variables=environment_names,
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "enabled": True,
        "worker_pid": os.getpid(),
        "started_at_utc": started_at_utc,
        "ended_at_utc": datetime.now(UTC).isoformat(),
        "case_name": case.name,
        "case_identity": case_identity,
        "requested_duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed_seconds,
        "completed_steps": completed_steps,
        "environment": cast("dict[str, JsonValue]", asdict(environment)),
    }


def main() -> int:
    """Read one request from stdin and emit one machine-readable result."""
    try:
        value = cast("object", json.loads(sys.stdin.read()))
        request = _mapping(value, _REQUEST_KEYS, "preconditioning request")
        document = run_preconditioning_request(request)
    except Exception as error:  # noqa: BLE001 - subprocess boundary serializes ordinary failures.
        _ = sys.stderr.write(f"Benchmark v2 preconditioning failed: {error}\n")
        return 1
    _ = sys.stdout.write(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
