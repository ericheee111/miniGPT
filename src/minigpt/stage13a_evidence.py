"""Generate and verify hash-bound Stage 13A paged-cache evidence."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Never, TypeAlias, cast

import torch
from typing_extensions import override

from minigpt.data import JsonValue
from minigpt.layers import KVCache, LayerKVCache
from minigpt.paged_kv_cache import PagedKVCacheConfig, PagedKVCachePool
from minigpt.serving_simulator import (
    SimulationResult,
    load_simulator_config,
    run_cache_backend_equivalence,
    run_simulation,
)

if TYPE_CHECKING:
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME = "13A"
STRESS_STEPS = 3000
_REBUILD_ACTION = 2
SCENARIO_CONFIGS = (
    "serving_paged_normal_burst.yaml",
    "serving_paged_tiny_pool.yaml",
    "serving_paged_reuse.yaml",
    "serving_paged_cancellation_churn.yaml",
    "serving_paged_failure_rollback.yaml",
    "serving_paged_overflow.yaml",
    "serving_paged_fragmentation.yaml",
)
EQUIVALENCE_CONFIGS = frozenset(
    {
        "serving_paged_normal_burst.yaml",
        "serving_paged_reuse.yaml",
        "serving_paged_cancellation_churn.yaml",
        "serving_paged_overflow.yaml",
        "serving_paged_fragmentation.yaml",
    }
)


@dataclass(frozen=True, slots=True)
class Stage13AEvidenceVerificationError(ValueError):
    """Report invalid Stage 13A evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 13A evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage13AEvidenceVerificationError(reason)


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


def _cache(length: int, *, offset: float) -> KVCache:
    layers: list[LayerKVCache] = []
    elements = 2 * 1 * length * 2
    for layer_index in range(2):
        key = torch.arange(elements, dtype=torch.float32).reshape(1, 2, length, 2)
        key = key + offset + layer_index * 1000.0
        layers.append(LayerKVCache(key=key, value=key + 0.5))
    return tuple(layers)


def run_allocator_stress(*, steps: int = STRESS_STEPS) -> EvidenceDocument:
    """Run deterministic randomized ownership mutations with an invariant check per step."""
    if isinstance(steps, bool) or steps <= 0:
        _invalid("allocator stress steps must be a positive integer")
    pool = PagedKVCachePool(
        PagedKVCacheConfig(block_tokens=4, num_blocks=32),
        n_layer=2,
        n_head=2,
        head_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    random_source = random.Random(20260807)  # noqa: S311
    active: dict[str, int] = {}
    next_request = 0
    maximum_fragmentation_tokens = 0
    maximum_fragmentation_ratio = 0.0
    operation_counts = {"reserve_prefill": 0, "append": 0, "rebuild": 0, "free": 0}
    trace_digest = hashlib.sha256()

    for step in range(steps):
        action = random_source.randrange(4)
        metrics = pool.metrics()
        if (action == 0 or not active) and metrics.reserved_blocks < pool.config.num_blocks:
            available = pool.config.num_blocks - metrics.reserved_blocks
            reserved = random_source.randint(1, min(4, available))
            request_id = f"request-{next_request}"
            next_request += 1
            pool.reserve(request_id, reserved)
            initial_length = random_source.randint(1, reserved * pool.config.block_tokens)
            pool.write_prefill(request_id, _cache(initial_length, offset=float(step)))
            active[request_id] = reserved
            operation_counts["reserve_prefill"] += 1
        elif active:
            request_id = random_source.choice(tuple(active))
            table = pool.request_cache(request_id)
            capacity = active[request_id] * pool.config.block_tokens
            if action == 1 and table.cache_length < capacity:
                pool.append(request_id, _cache(table.cache_length + 1, offset=float(step)))
                operation_counts["append"] += 1
            elif action == _REBUILD_ACTION:
                rebuilt_length = random_source.randint(1, capacity)
                pool.rebuild(request_id, _cache(rebuilt_length, offset=float(step)))
                operation_counts["rebuild"] += 1
            else:
                pool.release(request_id)
                del active[request_id]
                operation_counts["free"] += 1
        pool.verify_invariants()
        metrics = pool.metrics()
        maximum_fragmentation_tokens = max(
            maximum_fragmentation_tokens,
            metrics.internal_fragmentation_tokens,
        )
        maximum_fragmentation_ratio = max(
            maximum_fragmentation_ratio,
            metrics.internal_fragmentation_ratio,
        )
        trace_digest.update(
            (
                f"{step}:{metrics.free_blocks}:{metrics.allocated_blocks}:"
                f"{metrics.reserved_blocks}:{metrics.used_token_slots}\n"
            ).encode()
        )

    for request_id in tuple(active):
        pool.release(request_id)
        operation_counts["free"] += 1
        pool.verify_invariants()
    final_metrics = pool.metrics()
    if (
        final_metrics.free_blocks != final_metrics.total_blocks
        or final_metrics.allocated_blocks != 0
        or final_metrics.reserved_blocks != 0
    ):
        _invalid("allocator stress leaked blocks or reservations")
    return cast(
        "EvidenceDocument",
        {
            "seed": 20260807,
            "steps": steps,
            "invariant_checks": steps + len(active),
            "operation_counts": operation_counts,
            "requests_created": next_request,
            "maximum_fragmentation_tokens": maximum_fragmentation_tokens,
            "maximum_fragmentation_ratio": maximum_fragmentation_ratio,
            "trace_sha256": trace_digest.hexdigest(),
            "final_metrics": asdict(final_metrics),
        },
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


def _scenario_record(
    *,
    config_name: str,
    config_root: Path,
    work_root: Path,
) -> EvidenceDocument:
    config = load_simulator_config(config_root / config_name)
    if config_name in EQUIVALENCE_CONFIGS:
        comparison = run_cache_backend_equivalence(
            config,
            output_dir=work_root / config.scenario_name,
        )
        dense_hash = _document_sha256(_logical_document(comparison.dense))
        paged_hash = _document_sha256(_logical_document(comparison.paged))
        if dense_hash != paged_hash:
            _invalid(f"logical correctness hash differs for {config_name}")
        result = comparison.paged
        equivalent = comparison.equivalent
        checked_contracts: JsonValue = list(comparison.checked_contracts)
    else:
        result = run_simulation(config, output_dir=work_root / config.scenario_name / "paged")
        paged_hash = _document_sha256(_logical_document(result))
        dense_hash = None
        equivalent = None
        checked_contracts = []
    metrics = result.metrics
    released = (
        metrics.allocated_blocks == 0
        and metrics.reserved_blocks == 0
        and metrics.free_blocks == metrics.total_blocks
    )
    if not released:
        _invalid(f"scenario {config_name} leaked paged-cache resources")
    terminal = metrics.completed_requests + metrics.cancelled_requests + metrics.failed_requests
    if terminal != metrics.total_requests:
        _invalid(f"scenario {config_name} has non-terminal requests")
    return cast(
        "EvidenceDocument",
        {
            "config": config_name,
            "scenario_name": config.scenario_name,
            "equivalent": equivalent,
            "checked_contracts": checked_contracts,
            "dense_correctness_sha256": dense_hash,
            "paged_correctness_sha256": paged_hash,
            "all_resources_released": released,
            "metrics": asdict(metrics),
        },
    )


def generate_stage13a_evidence(
    *,
    config_root: Path,
    lifecycle_path: Path,
    package_root: Path,
    work_root: Path,
    source_commit: str,
) -> Path:
    """Run fixed workloads and build the Stage 13A evidence package."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    lifecycle = _read_json(lifecycle_path)
    if lifecycle.get("exit_code") != 0:
        _invalid("lifecycle test command did not pass")
    package_root.mkdir(parents=True, exist_ok=True)
    evidence_root = package_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    scenarios = [
        _scenario_record(config_name=name, config_root=config_root, work_root=work_root)
        for name in SCENARIO_CONFIGS
    ]
    scenario_document = cast(
        "EvidenceDocument",
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "scenarios": scenarios,
        },
    )
    stress = run_allocator_stress()
    _write_json(evidence_root / "scenarios.json", scenario_document)
    _write_json(evidence_root / "allocator_stress.json", stress)
    _ = shutil.copyfile(lifecycle_path, evidence_root / "lifecycle_tests.json")
    equivalent_count = sum(record["equivalent"] is True for record in scenarios)
    summary = cast(
        "EvidenceDocument",
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "source_commit": source_commit,
            "cache_backends": ["dense", "paged"],
            "scenario_count": len(scenarios),
            "dense_paged_equivalence_scenarios": equivalent_count,
            "all_equivalence_checks_passed": equivalent_count == len(EQUIVALENCE_CONFIGS),
            "all_resources_released": all(
                record["all_resources_released"] is True for record in scenarios
            ),
            "allocator_stress_steps": stress["steps"],
            "allocator_trace_sha256": stress["trace_sha256"],
            "dense_materialization_remains": True,
            "paged_attention": False,
            "speedup_claim": False,
        },
    )
    _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary, scenarios, stress),
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
    _ = verify_stage13a_evidence(package_root)
    return package_root


def verify_stage13a_evidence(package_root: Path) -> EvidenceDocument:  # noqa: C901
    """Verify exact hashes, claims, and a fresh deterministic invariant stress run."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 13A")
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
    if summary.get("scenario_count") != len(SCENARIO_CONFIGS):
        _invalid("summary scenario count is invalid")
    if summary.get("all_equivalence_checks_passed") is not True:
        _invalid("dense/paged equivalence is not fully verified")
    if summary.get("all_resources_released") is not True:
        _invalid("scenario resource release is not fully verified")
    recorded_stress = _read_json(package_root / "evidence" / "allocator_stress.json")
    fresh_stress = run_allocator_stress()
    if recorded_stress != _project_recorded_shape(recorded_stress, fresh_stress):
        _invalid("allocator stress result differs from fresh invariant run")
    lifecycle = _read_json(package_root / "evidence" / "lifecycle_tests.json")
    if lifecycle.get("exit_code") != 0:
        _invalid("lifecycle evidence did not pass")
    return manifest


def _project_recorded_shape(recorded: JsonValue, fresh: JsonValue) -> JsonValue:
    """Compare published fields while permitting additive future diagnostics."""
    if isinstance(recorded, dict):
        if not isinstance(fresh, dict):
            _invalid("fresh allocator stress changed a published object shape")
        missing = set(recorded) - set(fresh)
        if missing:
            _invalid(f"fresh allocator stress omitted published field {min(missing)!r}")
        return {key: _project_recorded_shape(value, fresh[key]) for key, value in recorded.items()}
    if isinstance(recorded, list):
        if not isinstance(fresh, list) or len(recorded) != len(fresh):
            _invalid("fresh allocator stress changed a published list shape")
        return [
            _project_recorded_shape(recorded_value, fresh_value)
            for recorded_value, fresh_value in zip(recorded, fresh, strict=True)
        ]
    return fresh


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


def _readme(
    summary: EvidenceDocument,
    scenarios: list[EvidenceDocument],
    stress: EvidenceDocument,
) -> str:
    lines = [
        "# Stage 13A — Paged KV Cache Memory Manager",
        "",
        "## Outcome",
        "",
        "Stage 13A keeps `dense` as the reference and adds a fixed physical K/V block pool,",
        "per-request block tables, block reservations, and transactional prefill/append/rebuild.",
        "Every terminal lifecycle path releases request ownership and reserved capacity.",
        "The physical layout is `[layers, num_blocks, heads, block_tokens, head_size]` for both K",
        "and V; one block ID addresses every layer for the same logical token interval.",
        "",
        "Reservation protects worst-case request capacity while allocation tracks blocks actually",
        "written. Fixed blocks do not have traditional contiguous external fragmentation; reported",
        "fragmentation is unused token slots in the tail block.",
        "",
        "## Correctness and capacity",
        "",
        "| Scenario | Equivalent | Completed | Cancelled | Failed | Peak allocated | Peak reserved | Reuse |",  # noqa: E501
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in scenarios:
        metrics = cast("EvidenceDocument", record["metrics"])
        first = f"| {record['scenario_name']} | {record['equivalent']} | "
        counts = (
            f"{metrics['completed_requests']} | {metrics['cancelled_requests']} | "
            f"{metrics['failed_requests']} | {metrics['peak_allocated_blocks']} | "
        )
        tail = f"{metrics['peak_reserved_blocks']} | {metrics['block_reuse_count']} |"
        lines.append(first + counts + tail)
    lines.extend(
        [
            "",
            "Five capacity-fitting workloads produced identical logical correctness hashes across",
            f"dense and paged storage. All {summary['scenario_count']} scenarios ended with zero",
            "owned blocks and reservations. Tiny-pool and capacity-failure runs are paged-only",
            "because deliberate physical rejection is storage-specific.",
            "",
            "## Allocator stress and rollback",
            "",
            f"The deterministic stress run executed {stress['steps']} operations; invariants were",
            "checked after every mutation. It covered allocation, append, rebuild, free, reuse,",
            f"and tail waste. Its trace hash is `{stress['trace_sha256']}`.",
            "Tests inject partial allocation, prefill write, append write, and overflow rebuild",
            "failures, and cover FIFO pressure, HTTP disconnect, and graceful shutdown cleanup.",
            "",
            "## Performance boundary",
            "",
            "Normal decode still materializes compact dense K/V for the Stage 11 executor, then",
            "writes only the newly appended token back. This is not PagedAttention; no speedup is claimed.",  # noqa: E501
            "A slower paged backend is expected until Stage 13B adds block-aware attention and",
            "measures it independently.",
        ]
    )
    return "\n".join(lines) + "\n"
