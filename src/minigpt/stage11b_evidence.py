"""Generate and verify Stage 11B length-bucketed prefill evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Never, TypeAlias, cast

from typing_extensions import override

from minigpt.serving_simulator import JsonValue, load_simulator_config, run_executor_equivalence

if TYPE_CHECKING:
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class Stage11BEvidenceVerificationError(ValueError):
    """Report missing, unexpected, or hash-mismatched Stage 11B evidence."""

    reason: str

    @override
    def __str__(self) -> str:
        return f"invalid Stage 11B evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage11BEvidenceVerificationError(reason)


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
        _invalid(f"artifact membership differs: {missing_paths=}, {unexpected_paths=}")
    for relative, (size, digest) in expected.items():
        path = root / relative
        if path.stat().st_size != size or _sha256(path) != digest:
            _invalid(f"artifact hash mismatch for {relative}")
    return manifest


def _copy_benchmark(benchmark_run_dir: Path, destination: Path) -> EvidenceDocument:
    _ = _verify_manifest(benchmark_run_dir, benchmark_run_dir / "artifact_manifest.json")
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
                for request_id, tokens in comparison.continuous.generated_tokens.items()
            },
            "terminal_statuses": {
                request_id: status.value
                for request_id, status in comparison.continuous.request_statuses.items()
            },
            "admission_order": list(comparison.continuous.admission_order),
            "prefill_batch_sizes": list(comparison.continuous.metrics.prefill_batch_sizes),
            "useful_prompt_tokens": comparison.continuous.metrics.useful_prompt_tokens,
            "padded_prompt_tokens": comparison.continuous.metrics.padded_prompt_tokens,
            "prompt_padding_waste_ratio": (
                comparison.continuous.metrics.prompt_padding_waste_ratio
            ),
        }
        _write_json(scenario_root / "equivalence.json", document)
        scenarios.append(document)
    return scenarios


def _readme(summary: EvidenceDocument) -> str:
    benchmark = cast("EvidenceDocument", summary["benchmark"])
    scenarios = cast("list[EvidenceDocument]", benchmark["scenarios"])
    header = "{} {}".format(
        "| Scenario | Verdict | Prefill batch | Prompt waste |",
        "Decode TTFT | Full TTFT | Decode req/s | Full req/s |",
    )
    lines = [
        "# Stage 11B — Length-Bucketed Batched Prefill",
        "",
        "## Outcome",
        "",
        "Stage 11A batches one decode token per eligible request after each prompt has already",
        "been processed separately. Stage 11B keeps that decode path and additionally groups the",
        "currently eligible FIFO prompt prefix into tensor-level prefill batches. It never waits",
        "for future requests and never skips the FIFO head to improve utilization.",
        "",
        "Prompt rows are right-padded to `[B, Tmax]`; valid learned absolute positions remain",
        "`0..L-1`. A per-row causal/key-valid mask prevents valid queries from reading padding.",
        "Final logits are gathered at `L-1`. Dense layer caches are scattered back to compact",
        "`[1, H, L, D]` tensors, so padding does not enter caller-owned request state.",
        "",
        "Prompt padding is more expensive than decode cache padding: prefill computes projection,",
        "attention, MLP, and residual work for every padded query position in every layer, whereas",
        "Stage 11A adds only one new decode query per row. Length bucketing bounds that repeated",
        "work with batch-size, padded-token-budget, and padding-ratio limits.",
        "",
        "This remains dense batching, not paged attention. Throughput can improve while TTFT for",
        "some requests worsens, so the canonical report includes both rather than treating",
        "throughput alone as success.",
        "",
        f"Overall strict benchmark verdict: `{benchmark['strict_verdict']}`.",
        "",
        header,
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in scenarios:
        decode_only = cast("EvidenceDocument", scenario["continuous_decode"])
        continuous = cast("EvidenceDocument", scenario["continuous"])
        row_template = "{} {}".format(
            "| {scenario} | {verdict} | {batch} | {waste} | {decode_ttft} | {full_ttft} |",
            "{decode_throughput} | {full_throughput} |",
        )
        lines.append(
            row_template.format(
                scenario=scenario["scenario"],
                verdict=scenario["strict_verdict"],
                batch=continuous["median_average_prefill_batch_size"],
                waste=continuous["median_prompt_padding_waste_ratio"],
                decode_ttft=decode_only["median_median_ttft_seconds"],
                full_ttft=continuous["median_median_ttft_seconds"],
                decode_throughput=decode_only["median_request_throughput_per_second"],
                full_throughput=continuous["median_request_throughput_per_second"],
            )
        )
    lines.extend(
        [
            "",
            "Every canonical scenario is `not_comparable` because at least one primary executor",
            "exceeded the 10% CV threshold. Therefore the medians above are descriptive only and",
            "do not establish a performance improvement. Equal, mixed, short-heavy, and long-heavy",
            "bursts show higher full-continuous request/s medians, but burst TTFT is generally",
            "higher. Staggered arrivals form smaller batches. High padding pressure is split into",
            "size-one prefills by the FIFO policy and shows no descriptive throughput gain.",
            "",
            "Canonical timings use alternating fresh processes and `time.perf_counter`; raw",
            "replicates are unfiltered. Median, MAD, CV, queue/prefill/TTFT/E2E timing,",
            "request and token throughput, worker peak RSS, utilization, environment, order,",
            "and hashes are included. Profiler output is descriptive only and is not canonical",
            "benchmark data.",
            "",
            "Simulator evidence runs `reference`, `continuous_decode`, and `continuous` from the",
            "same weights, workloads, seeds, scheduler, and cache budget. It verifies tokens,",
            "terminal/cancellation states, FIFO admission, cache accounting, logical request",
            "events, and request metrics. Executor-specific batch events are reported separately.",
            "",
            "There is no HTTP layer, paged cache, BPE, GPU/distributed path, or model upgrade.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_stage11b_evidence(
    *,
    benchmark_run_dir: Path,
    simulator_config_paths: tuple[Path, ...],
    package_root: Path,
    source_commit: str,
) -> Path:
    """Generate simulator and canonical benchmark evidence with one outer manifest."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    package_root.mkdir(parents=True, exist_ok=True)
    evidence_root = package_root / "evidence"
    benchmark = _copy_benchmark(benchmark_run_dir, evidence_root / "benchmark")
    simulator = _simulator_evidence(simulator_config_paths, evidence_root / "simulator")
    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": "11B",
        "source_commit": source_commit,
        "prefill_batched": True,
        "decode_batched": True,
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
            "stage": "11B",
            "source_commit": source_commit,
            "artifacts": artifacts,
        },
    )
    _ = verify_stage11b_evidence(package_root)
    return package_root


def verify_stage11b_evidence(package_root: Path) -> EvidenceDocument:
    """Verify exact package membership, byte sizes, and SHA-256 digests."""
    return _verify_manifest(package_root, package_root / "artifact_manifest.json")
