"""Generate and verify Stage 11A decode continuous batching evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Never, TypeAlias, cast

from typing_extensions import override

from minigpt.serving_simulator import (
    JsonValue,
    load_simulator_config,
    run_executor_equivalence,
)

if TYPE_CHECKING:
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class Stage11AEvidenceVerificationError(ValueError):
    """Report missing, unexpected, or hash-mismatched Stage 11A evidence."""

    reason: str

    @override
    def __str__(self) -> str:
        return f"invalid Stage 11A evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage11AEvidenceVerificationError(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> EvidenceDocument:
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        _invalid(f"{path} must contain a JSON object")
    return cast("EvidenceDocument", raw)


def _write_json(path: Path, document: EvidenceDocument) -> None:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _verify_manifest(root: Path, manifest_path: Path) -> EvidenceDocument:
    manifest = _read_json(manifest_path)
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        _invalid(f"{manifest_path} artifacts must be a list")
    expected: dict[str, tuple[int, str]] = {}
    for raw_entry in cast("list[object]", entries):
        if not isinstance(raw_entry, dict):
            _invalid(f"{manifest_path} artifact entries must be objects")
        entry = cast("dict[str, object]", raw_entry)
        path = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(path, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            _invalid(f"{manifest_path} artifact entry fields are invalid")
        expected[path] = (size, digest)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(expected):
        missing_paths = sorted(set(expected) - actual)
        unexpected_paths = sorted(actual - set(expected))
        reason = f"artifact membership differs: {missing_paths=}, {unexpected_paths=}"
        _invalid(reason)
    for relative, (size, digest) in expected.items():
        path = root / relative
        if path.stat().st_size != size or _sha256(path) != digest:
            _invalid(f"artifact hash mismatch for {relative}")
    return manifest


def _copy_benchmark(benchmark_run_dir: Path, destination: Path) -> EvidenceDocument:
    manifest_path = benchmark_run_dir / "artifact_manifest.json"
    _ = _verify_manifest(benchmark_run_dir, manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    for path in benchmark_run_dir.iterdir():
        if path.is_file():
            _ = shutil.copyfile(path, destination / path.name)
    return _read_json(destination / "summary.json")


def _simulator_evidence(
    config_paths: tuple[Path, ...],
    destination: Path,
) -> list[JsonValue]:
    scenarios: list[JsonValue] = []
    for config_path in config_paths:
        config = load_simulator_config(config_path)
        scenario_root = destination / config.scenario_name
        scenario_root.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(config_path, scenario_root / "workload.yaml")
        comparison = run_executor_equivalence(config, output_dir=scenario_root)
        document: EvidenceDocument = {
            "scenario": config.scenario_name,
            "equivalent": comparison.equivalent,
            "checked_contracts": list(comparison.checked_contracts),
            "generated_tokens": {
                request_id: list(tokens)
                for request_id, tokens in comparison.continuous_decode.generated_tokens.items()
            },
            "terminal_statuses": {
                request_id: status.value
                for request_id, status in comparison.continuous_decode.request_statuses.items()
            },
            "admission_order": list(comparison.continuous_decode.admission_order),
        }
        _write_json(scenario_root / "equivalence.json", document)
        scenarios.append(document)
    return scenarios


def _readme(summary: EvidenceDocument) -> str:
    benchmark = cast("EvidenceDocument", summary["benchmark"])
    scenarios = cast("list[EvidenceDocument]", benchmark["scenarios"])
    lines = [
        "# Stage 11A — Decode Continuous Batching",
        "",
        "## Outcome",
        "",
        "Stage 10 advanced multiple request state machines per engine iteration but invoked the",
        "model separately for every request. Stage 11A retains that control plane and replaces",
        "eligible decode execution with one tensor-level single-token batch per tick.",
        "Prefill remains per request.",
        "",
        "Caller caches stay compact `[1, H, L, D]`. Assembly creates right-padded dense",
        "`[B, H, max(L), D]` layers plus a cache-valid mask and per-row learned-position offset.",
        "Scatter takes each valid historical prefix and the new-token column, so padding never",
        "enters returned caches and old caller tensors are not modified.",
        "",
        "This is dense padded attention, not paged attention. Padding increases memory traffic;",
        "mixed-cache benchmark observed the largest waste. Decode batching can improve aggregate",
        "throughput while increasing a single request's TPOT or E2E latency, so both are reported.",
        "",
        f"Overall strict benchmark verdict: `{benchmark['strict_verdict']}`.",
        "",
        "| Scenario | Verdict | Speedup | Continuous avg batch | Padding waste | Token/s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario in scenarios:
        continuous = cast("EvidenceDocument", scenario["continuous_decode"])
        lines.append(
            "| {scenario} | {verdict} | {speedup} | {batch} | {waste} | {throughput} |".format(
                scenario=scenario["scenario"],
                verdict=scenario["strict_verdict"],
                speedup=scenario["speedup_reference_over_continuous"],
                batch=continuous["median_average_decode_batch_size"],
                waste=continuous["median_padding_waste_ratio"],
                throughput=continuous["median_token_throughput_per_second"],
            )
        )
    lines.extend(
        [
            "",
            "Canonical timings come from alternating fresh processes using `time.perf_counter`.",
            "Raw replicates are unfiltered. Environment, resolved configuration, execution order,",
            "median, MAD, CV, TTFT, TPOT, E2E, throughput, utilization, and hashes are included.",
            "Profiler output is explicitly separate and is not used for throughput claims.",
            "",
            "The simulator evidence runs reference and continuous executors from identical model",
            "weights, workloads, and request seeds. It checks tokens, terminal/cancellation",
            "states, FIFO admission order, complete logical events, and request metrics.",
            "",
            "## Limits and Stage 11B",
            "",
            "There is no HTTP layer, batched prefill, paged cache, BPE, GPU path, distributed",
            "execution, or new model structure. Stage 11B should consider length-bucketed batched",
            "prefill with explicit admission/TTFT guardrails; it should not reuse decode padding",
            "without measuring prompt-side memory amplification.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_stage11a_evidence(
    *,
    benchmark_run_dir: Path,
    simulator_config_paths: tuple[Path, ...],
    package_root: Path,
    source_commit: str,
) -> Path:
    """Generate deterministic simulator evidence plus copied canonical benchmark evidence."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    package_root.mkdir(parents=True, exist_ok=True)
    evidence_root = package_root / "evidence"
    benchmark = _copy_benchmark(benchmark_run_dir, evidence_root / "benchmark")
    simulator = _simulator_evidence(simulator_config_paths, evidence_root / "simulator")
    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": "11A",
        "source_commit": source_commit,
        "prefill_batched": False,
        "paged_attention": False,
        "benchmark": benchmark,
        "simulator": simulator,
    }
    _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = package_root / "artifact_manifest.json"
    artifacts: list[JsonValue] = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "stage": "11A",
            "source_commit": source_commit,
            "artifacts": artifacts,
        },
    )
    _ = verify_stage11a_evidence(package_root)
    return package_root


def verify_stage11a_evidence(package_root: Path) -> EvidenceDocument:
    """Verify exact package membership, byte sizes, and SHA-256 digests."""
    return _verify_manifest(package_root, package_root / "artifact_manifest.json")
