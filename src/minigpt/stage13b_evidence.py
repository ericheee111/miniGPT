"""Generate and verify Stage 13B block-aware paged-attention evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Never, TypeAlias, cast

from typing_extensions import override

from minigpt.data import JsonValue
from minigpt.serving_simulator import (
    SimulationResult,
    load_simulator_config,
    run_paged_attention_equivalence,
)

if TYPE_CHECKING:
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME = "13B"
CORRECTNESS_CONFIGS = (
    "serving_paged_overflow.yaml",
    "serving_paged_attention.yaml",
)


@dataclass(frozen=True, slots=True)
class Stage13BEvidenceVerificationError(ValueError):
    """Report invalid Stage 13B evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 13B evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage13BEvidenceVerificationError(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_sha256(document: JsonValue) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> EvidenceDocument:
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        _invalid(f"{path} must contain a JSON object")
    return cast("EvidenceDocument", raw)


def _write_json(path: Path, document: EvidenceDocument) -> None:
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _logical_document(result: SimulationResult) -> EvidenceDocument:
    return cast(
        "EvidenceDocument",
        {
            "generated_tokens": result.generated_tokens,
            "request_statuses": result.request_statuses,
            "admission_order": result.admission_order,
            "events": [asdict(event) for event in result.events],
            "request_metrics": {
                request_id: asdict(metrics)
                for request_id, metrics in sorted(result.request_metrics.items())
            },
        },
    )


def generate_stage13b_correctness(
    *,
    config_root: Path,
    work_root: Path,
    output_path: Path,
) -> Path:
    """Run dense, materialized, and direct strategies and bind logical hashes."""
    records: list[JsonValue] = []
    for config_name in CORRECTNESS_CONFIGS:
        config = load_simulator_config(config_root / config_name)
        comparison = run_paged_attention_equivalence(
            config,
            output_dir=work_root / config.scenario_name,
        )
        hashes = {
            "dense": _document_sha256(_logical_document(comparison.dense)),
            "materialized": _document_sha256(_logical_document(comparison.materialized)),
            "direct": _document_sha256(_logical_document(comparison.direct)),
        }
        if len(set(hashes.values())) != 1:
            _invalid(f"logical correctness hashes differ for {config_name}")
        metrics = comparison.direct.metrics
        released = (
            metrics.allocated_blocks == 0
            and metrics.reserved_blocks == 0
            and metrics.free_blocks == metrics.total_blocks
        )
        if not released:
            _invalid(f"direct scenario {config_name} leaked blocks")
        records.append(
            cast(
                "JsonValue",
                {
                    "config": config_name,
                    "scenario_name": config.scenario_name,
                    "equivalent": comparison.equivalent,
                    "checked_contracts": list(comparison.checked_contracts),
                    "correctness_sha256": hashes,
                    "direct_padding_waste_ratio": metrics.padding_waste_ratio,
                    "all_resources_released": released,
                    "overflow_fallback_observed": any(
                        event.used_fallback for event in comparison.direct.events
                    ),
                },
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "scenarios": records,
        },
    )
    return output_path


def generate_stage13b_evidence(
    *,
    correctness_path: Path,
    benchmark_path: Path,
    lifecycle_path: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Build a compact hash-bound correctness and descriptive benchmark package."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    inputs = {
        "correctness.json": correctness_path,
        "benchmark.json": benchmark_path,
        "lifecycle_tests.json": lifecycle_path,
    }
    for path in inputs.values():
        if not path.is_file():
            _invalid(f"evidence input does not exist: {path}")
    package_root.mkdir(parents=True, exist_ok=True)
    evidence_root = package_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        _ = shutil.copyfile(path, evidence_root / name)
    correctness = _read_json(evidence_root / "correctness.json")
    benchmark = _read_json(evidence_root / "benchmark.json")
    lifecycle = _read_json(evidence_root / "lifecycle_tests.json")
    scenarios = correctness.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != len(CORRECTNESS_CONFIGS):
        _invalid("correctness scenario count is invalid")
    if lifecycle.get("exit_code") != 0:
        _invalid("Stage 13B lifecycle tests did not pass")
    if (
        benchmark.get("verdict") != "descriptive_only"
        or benchmark.get("speedup_claim") is not False
    ):
        _invalid("benchmark must remain descriptive without a speedup claim")
    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "correctness_scenarios": len(scenarios),
        "all_correctness_checks_passed": all(
            isinstance(record, dict) and cast("dict[str, object]", record).get("equivalent") is True
            for record in cast("list[object]", scenarios)
        ),
        "all_resources_released": all(
            isinstance(record, dict)
            and cast("dict[str, object]", record).get("all_resources_released") is True
            for record in cast("list[object]", scenarios)
        ),
        "normal_decode_dense_materialization": False,
        "overflow_reprefill_remains_dense": True,
        "benchmark_verdict": benchmark["verdict"],
        "speedup_claim": False,
    }
    _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary, correctness, benchmark),
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
            "stage": STAGE_NAME,
            "source_commit": source_commit,
            "artifacts": artifacts,
        },
    )
    _ = verify_stage13b_evidence(package_root)
    return package_root


def verify_stage13b_evidence(package_root: Path) -> EvidenceDocument:  # noqa: C901
    """Verify exact package membership, hashes, source identity, and bounded claims."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 13B")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        _invalid("manifest source_commit must be non-empty")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        _invalid("manifest artifacts must be a list")
    expected = _manifest_entries(cast("list[object]", entries))
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(expected):
        _invalid("artifact membership differs from manifest")
    for relative, (size, digest) in expected.items():
        path = package_root / relative
        if path.stat().st_size != size or _sha256(path) != digest:
            _invalid(f"artifact hash mismatch for {relative}")
    summary = _read_json(package_root / "summary.json")
    if summary.get("source_commit") != source_commit:
        _invalid("summary source_commit differs from manifest")
    if summary.get("all_correctness_checks_passed") is not True:
        _invalid("correctness checks did not all pass")
    if summary.get("all_resources_released") is not True:
        _invalid("resource release checks did not all pass")
    if summary.get("normal_decode_dense_materialization") is not False:
        _invalid("normal decode materialization claim is invalid")
    if summary.get("speedup_claim") is not False:
        _invalid("evidence must not claim a speedup")
    return manifest


def _manifest_entries(entries: list[object]) -> dict[str, tuple[int, str]]:
    expected: dict[str, tuple[int, str]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            _invalid("manifest artifact entries must be objects")
        entry = cast("dict[str, object]", raw_entry)
        relative = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            _invalid("manifest artifact entry fields are invalid")
        if relative in expected:
            _invalid(f"duplicate manifest path {relative}")
        expected[relative] = (size, digest)
    return expected


def _metric_summary(benchmark: EvidenceDocument, strategy: str, metric: str) -> EvidenceDocument:
    strategies = cast("EvidenceDocument", benchmark["strategies"])
    strategy_document = cast("EvidenceDocument", strategies[strategy])
    return cast("EvidenceDocument", strategy_document[metric])


def _readme(
    summary: EvidenceDocument,
    correctness: EvidenceDocument,
    benchmark: EvidenceDocument,
) -> str:
    scenarios = cast("list[EvidenceDocument]", correctness["scenarios"])
    materialized = _metric_summary(benchmark, "materialized", "e2e_seconds")
    direct = _metric_summary(benchmark, "direct", "e2e_seconds")
    cache_access = cast("EvidenceDocument", benchmark["cache_access"])
    lines = [
        "# Stage 13B — Block-Aware Paged Attention Decode",
        "",
        "## Outcome",
        "",
        "Normal single-token decode now traverses ordered physical K/V block views directly.",
        "Historical K/V is neither concatenated into a compact cache nor padded densely.",
        "One global softmax covers score chunks in logical token order; value context accumulates",
        "block by block. The model returns one K/V delta per layer and the pool appends it",
        "transactionally. Initial prefill and learned-position overflow re-prefill remain dense.",
        "",
        "## Correctness",
        "",
    ]
    for scenario in scenarios:
        hashes = cast("EvidenceDocument", scenario["correctness_sha256"])
        identity = f"- `{scenario['scenario_name']}`: dense/materialized/direct hash "
        release = f"`{hashes['direct']}`, zero leaks `{scenario['all_resources_released']}`, "
        overflow = f"overflow fallback `{scenario['overflow_fallback_observed']}`."
        lines.append(identity + release + overflow)
    lines.extend(
        [
            "",
            "A guard replaces `PagedKVCachePool.materialize` with an exception; direct generation",
            "still completes. Generated tokens, terminal states/cancellation, FIFO",
            "admission, logical events, request metrics, and logical cache accounting match both",
            "reference strategies.",
            "",
            "## Descriptive CPU benchmark",
            "",
            "| Strategy | E2E median (s) | E2E CV |",
            "|---|---:|---:|",
            f"| Stage 13A materialized | {materialized['median']} | {materialized['cv']} |",
            f"| Stage 13B direct | {direct['median']} | {direct['cv']} |",
            "",
            f"Cache-access loop materialize time: `{cache_access['materialize_seconds']}` seconds;",
            f"request-view time: `{cache_access['request_view_seconds']}` seconds.",
            "",
            "These are single-machine descriptive CPU measurements. The verdict is",
            f"`{summary['benchmark_verdict']}` and no speedup is claimed. Stable cross-run and",
            "environment-identity requirements were not used to make a release performance claim.",
        ]
    )
    return "\n".join(lines) + "\n"
