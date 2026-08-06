"""Generate and verify the Stage 10 serving-control-plane evidence package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Never, cast

from typing_extensions import override

from minigpt.serving_simulator import JsonValue, load_simulator_config, run_simulation

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvidenceVerificationError(ValueError):
    """Report a missing, unexpected, or hash-mismatched evidence artifact."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the failed evidence invariant."""
        return f"invalid Stage 10 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise EvidenceVerificationError(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, JsonValue]) -> None:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path) -> dict[str, JsonValue]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        _invalid(f"{path} must contain a JSON object")
    return cast("dict[str, JsonValue]", value)


def _readme(scenarios: list[dict[str, JsonValue]]) -> str:
    lines = [
        "# Stage 10 — MiniServe Request Scheduling and Engine Control Plane",
        "",
        "## Outcome",
        "",
        "This package demonstrates deterministic request scheduling, lifecycle transitions, cache",
        "reservation/release, cancellation, backpressure, failure isolation, and metric",
        "accounting.",
        "It does not claim a throughput improvement.",
        "",
        "The executor remains explicitly per-request: each engine iteration may advance multiple",
        "request state machines, but every model call is still a separate `GPT.prefill()` or",
        "`GPT.decode()` invocation. Iteration-level scheduling is not tensor-level continuous",
        "batching; that executor change is reserved for Stage 11.",
        "",
        "## Why online serving needs scheduling",
        "",
        "Independent arrivals compete for active slots and KV-cache capacity. Strict FIFO",
        "admission provides deterministic fairness, while backpressure keeps requests waiting",
        "until both an active slot and their worst-case cache reservation are available.",
        "Completion, cancellation, or failure releases the reservation immediately.",
        "",
        "Prefill and decode are interleaved by engine tick. Newly admitted requests remain",
        "`PREFILLING` until the next tick, so cancellation can be applied before model execution.",
        "Each decoding request emits at most one token per tick. Learned-position overflow",
        "preserves the Stage 9 sliding-window re-prefill fallback.",
        "",
        "## Metric definitions",
        "",
        "Queue time is admission minus arrival. TTFT is first token minus arrival. Prefill latency",
        "measures the initial prompt call. Per-token decode latency records each later token call,",
        "and TPOT is their mean. E2E latency is terminal time minus arrival. Request and token",
        "throughput are descriptive logical-workload accounting, not benchmark performance.",
        "",
        "## Fixed scenarios",
        "",
        "| Scenario | Requests | Completed | Cancelled | Failed | Peak active | Peak cache |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        "".join(  # noqa: FLY002
            (
                "| {scenario_name} | {total_requests} | {completed_requests} | ",
                "{cancelled_requests} | {failed_requests} | {peak_active_requests} | ",
                "{peak_cached_tokens} |",
            )
        ).format_map(scenario)
        for scenario in scenarios
    )
    lines.extend(
        [
            "",
            "Every scenario directory contains the exact workload plus `events.jsonl`,",
            "`requests.csv`, `summary.json`, and `timeline.md`. `artifact_manifest.json` binds all",
            "committed artifacts by byte count and SHA-256.",
            "",
            "## Known limitations",
            "",
            "There is no HTTP layer, tensor batch assembly, paged cache, BPE, GPU path,",
            "distributed execution, or production admission policy. Logical clocks prove metric",
            "formulas and event determinism; they are not wall-clock performance measurements.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_stage10_evidence(
    *,
    config_paths: tuple[Path, ...],
    package_root: Path,
    source_commit: str,
) -> Path:
    """Run fixed scenarios and write a deterministic hash-bound package."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    package_root.mkdir(parents=True, exist_ok=True)
    evidence_root = package_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    scenarios: list[dict[str, JsonValue]] = []
    for config_path in config_paths:
        config = load_simulator_config(config_path)
        scenario_root = evidence_root / config.scenario_name
        scenario_root.mkdir(parents=True, exist_ok=True)
        workload_path = scenario_root / "workload.yaml"
        _ = workload_path.write_bytes(config_path.read_bytes())
        result = run_simulation(config, output_dir=scenario_root)
        scenario_summary = _read_json(result.output_dir / "summary.json")
        scenarios.append(scenario_summary)

    summary = cast(
        "dict[str, JsonValue]",
        {
            "schema_version": 1,
            "stage": "10",
            "source_commit": source_commit,
            "claim": "control-plane correctness only; no throughput improvement claim",
            "executor": "per-request reference executor",
            "continuous_batching": False,
            "scenarios": scenarios,
        },
    )
    _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(scenarios),
        encoding="utf-8",
        newline="\n",
    )

    manifest_path = package_root / "artifact_manifest.json"
    artifact_paths = sorted(
        path for path in package_root.rglob("*") if path.is_file() and path != manifest_path
    )
    artifacts: list[JsonValue] = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in artifact_paths
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "stage": "10",
            "source_commit": source_commit,
            "artifacts": artifacts,
        },
    )
    _ = verify_stage10_evidence(package_root)
    return package_root


def verify_stage10_evidence(package_root: Path) -> dict[str, JsonValue]:
    """Verify exact artifact membership, byte sizes, and SHA-256 digests."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        _invalid("manifest artifacts must be a list")
    expected: dict[str, tuple[int, str]] = {}
    for item in cast("list[object]", raw_artifacts):
        if not isinstance(item, dict):
            _invalid("manifest artifact entries must be objects")
        entry = cast("dict[str, object]", item)
        path = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if not isinstance(path, str) or isinstance(size, bool) or not isinstance(size, int):
            _invalid("manifest artifact path/bytes fields are invalid")
        if not isinstance(digest, str):
            _invalid("manifest artifact sha256 field is invalid")
        expected[path] = (size, digest)
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        unexpected = sorted(actual_paths - set(expected))
        _invalid(f"artifact membership differs; missing={missing}, unexpected={unexpected}")
    for relative, (expected_size, expected_digest) in expected.items():
        path = package_root / relative
        if path.stat().st_size != expected_size or _sha256(path) != expected_digest:
            _invalid(f"artifact hash mismatch for {relative}")
    return manifest
