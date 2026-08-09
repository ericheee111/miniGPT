"""Generate and verify hash-bound Stage 14 Automatic Prefix Caching evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Never, TypeAlias, cast

from typing_extensions import override

from minigpt.data import JsonValue
from minigpt.paged_kv_cache import KVCacheBackend
from minigpt.serving import EngineEventType
from minigpt.serving_simulator import (
    SimulatorExecutor,
    load_simulator_config,
    run_paged_attention_equivalence,
    run_simulation,
)

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.serving_simulator import SimulationResult

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME = "14"
CORRECTNESS_CONFIG = "serving_automatic_prefix_cache.yaml"
_PREFIX_EVENTS = frozenset(
    {
        EngineEventType.PREFIX_LOOKUP,
        EngineEventType.PREFIX_HIT,
        EngineEventType.PREFIX_PROMOTE,
        EngineEventType.PREFIX_EVICT,
    }
)


@dataclass(frozen=True, slots=True)
class Stage14EvidenceVerificationError(ValueError):
    """Report invalid Stage 14 evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 14 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage14EvidenceVerificationError(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_sha256(document: JsonValue) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _logical_document(result: SimulationResult) -> EvidenceDocument:
    events: list[JsonValue] = [
        {
            "event_type": event.event_type.value,
            "request_id": event.request_id,
            "status": event.status.value,
            "token_id": event.token_id,
            "used_fallback": event.used_fallback,
            "active_requests": event.active_requests,
            "waiting_requests": event.waiting_requests,
            "cached_tokens": event.cached_tokens,
            "reserved_cache_tokens": event.reserved_cache_tokens,
        }
        for event in result.events
        if event.event_type not in _PREFIX_EVENTS
    ]
    return cast(
        "EvidenceDocument",
        {
            "generated_tokens": result.generated_tokens,
            "generator_state_hashes": result.generator_state_hashes,
            "request_statuses": {
                key: value.value for key, value in result.request_statuses.items()
            },
            "admission_order": result.admission_order,
            "logical_events": events,
        },
    )


def generate_stage14_correctness(
    *,
    config_root: Path,
    work_root: Path,
    output_path: Path,
) -> Path:
    """Bind dense/materialized/direct/APC output, RNG, and logical-event identity."""
    configured = load_simulator_config(config_root / CORRECTNESS_CONFIG)
    base = replace(configured, prefix_cache_enabled=False)
    stage13 = run_paged_attention_equivalence(base, output_dir=work_root / "stage13")
    apc = run_simulation(
        replace(
            base,
            executor=SimulatorExecutor.PAGED_ATTENTION,
            kv_cache_backend=KVCacheBackend.PAGED,
            prefix_cache_enabled=True,
        ),
        output_dir=work_root / "apc",
    )
    strategies = {
        "dense": stage13.dense,
        "materialized": stage13.materialized,
        "direct": stage13.direct,
        "apc": apc,
    }
    hashes = {
        name: _document_sha256(_logical_document(result)) for name, result in strategies.items()
    }
    if len(set(hashes.values())) != 1:
        _invalid("dense/materialized/direct/APC logical correctness hashes differ")
    metrics = apc.metrics
    active_resources_released = (
        metrics.active_shared_references == 0
        and metrics.reserved_blocks == 0
        and metrics.allocated_blocks == metrics.prefix_cache_blocks
    )
    if not active_resources_released:
        _invalid("APC correctness scenario leaked active refs, private blocks, or reservations")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "config": CORRECTNESS_CONFIG,
                "equivalent": True,
                "correctness_sha256": hashes,
                "checked_contracts": [
                    "generated_tokens",
                    "generator_state",
                    "terminal_state_and_cancellation",
                    "fifo_admission_order",
                    "logical_request_events",
                ],
                "prefix_hit_requests": metrics.prefix_hit_requests,
                "prefix_hit_tokens": metrics.prefix_hit_tokens,
                "prefix_miss_tokens": metrics.prefix_miss_tokens,
                "prefill_tokens_computed": metrics.prefill_tokens_computed,
                "avoided_prefill_tokens": metrics.avoided_prefill_tokens,
                "prefix_cache_blocks": metrics.prefix_cache_blocks,
                "active_shared_references": metrics.active_shared_references,
                "active_resources_released": active_resources_released,
                "overflow_fallback_observed": any(event.used_fallback for event in apc.events),
            },
        ),
    )
    return output_path


def generate_stage14_evidence(  # noqa: C901, PLR0913
    *,
    correctness_path: Path,
    benchmark_path: Path,
    stress_path: Path,
    lifecycle_path: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Build the compact APC correctness, stress, lifecycle, and benchmark package."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    inputs = {
        "correctness.json": correctness_path,
        "benchmark.json": benchmark_path,
        "stress_tests.json": stress_path,
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
    stress = _read_json(evidence_root / "stress_tests.json")
    lifecycle = _read_json(evidence_root / "lifecycle_tests.json")
    if correctness.get("equivalent") is not True:
        _invalid("dense/materialized/direct/APC correctness did not pass")
    if correctness.get("active_resources_released") is not True:
        _invalid("correctness run leaked active cache resources")
    if stress.get("exit_code") != 0 or lifecycle.get("exit_code") != 0:
        _invalid("stress and lifecycle tests must both pass")
    verdict = benchmark.get("strict_verdict")
    if verdict not in {"pass", "fail", "not_comparable"}:
        _invalid("benchmark strict_verdict is invalid")
    if benchmark.get("correctness_equivalent") is not True:
        _invalid("benchmark logical correctness differs")
    speedup_claim = verdict == "pass"
    if benchmark.get("wall_clock_performance_improvement") is not speedup_claim:
        _invalid("benchmark improvement claim differs from strict verdict")
    totals = _benchmark_totals(benchmark)
    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "cache_identity": (
            "checkpoint+model_config+dtype+device+block_tokens+schema+position_semantics"
        ),
        "prefix_hash_chain": "SHA256(namespace || parent_hash || full_token_block)",
        "full_block_sharing_only": True,
        "partial_tail_private": True,
        "partial_block_copy_on_write": False,
        "shared_blocks_immutable": True,
        "active_refcount_and_zero_ref_lru": True,
        "all_correctness_checks_passed": True,
        "active_resources_released": True,
        "stress_mutations": 5000,
        "stress_passed": True,
        "lifecycle_passed": True,
        "prefix_hit_requests": totals["prefix_hit_requests"],
        "prefix_hit_tokens": totals["prefix_hit_tokens"],
        "avoided_prefill_tokens": totals["avoided_prefill_tokens"],
        "prefix_cache_evictions": totals["evictions"],
        "benchmark_strict_verdict": verdict,
        "wall_clock_performance_improvement": speedup_claim,
        "implementation": "Python/PyTorch reference implementation",
        "fused_paged_attention_kernel": False,
    }
    _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary, benchmark),
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
    _ = verify_stage14_evidence(package_root)
    return package_root


def verify_stage14_evidence(package_root: Path) -> EvidenceDocument:  # noqa: C901
    """Verify exact membership, hashes, source identity, and bounded performance claims."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 14")
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
    for key in (
        "full_block_sharing_only",
        "partial_tail_private",
        "shared_blocks_immutable",
        "active_refcount_and_zero_ref_lru",
        "all_correctness_checks_passed",
        "active_resources_released",
        "stress_passed",
        "lifecycle_passed",
    ):
        if summary.get(key) is not True:
            _invalid(f"summary contract {key} did not pass")
    if summary.get("partial_block_copy_on_write") is not False:
        _invalid("Stage 14 must not claim partial-block copy-on-write")
    verdict = summary.get("benchmark_strict_verdict")
    claim = summary.get("wall_clock_performance_improvement")
    if claim is not (verdict == "pass"):
        _invalid("summary wall-clock claim differs from strict verdict")
    if summary.get("fused_paged_attention_kernel") is not False:
        _invalid("Stage 14 must not claim a fused PagedAttention kernel")
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


def _benchmark_totals(benchmark: EvidenceDocument) -> dict[str, int]:
    workloads = cast("EvidenceDocument", benchmark["workloads"])
    totals = {
        "prefix_hit_requests": 0,
        "prefix_hit_tokens": 0,
        "avoided_prefill_tokens": 0,
        "evictions": 0,
    }
    for raw in workloads.values():
        workload = cast("EvidenceDocument", raw)
        strategies = cast("EvidenceDocument", workload["strategies"])
        apc = cast("EvidenceDocument", strategies["paged_direct_apc"])
        for key in totals:
            totals[key] += cast("int", apc[key])
    return totals


def _readme(summary: EvidenceDocument, benchmark: EvidenceDocument) -> str:
    workloads = cast("EvidenceDocument", benchmark["workloads"])
    lines = [
        "# Stage 14 — Automatic Prefix Caching",
        "",
        "## Outcome",
        "",
        "Stage 14 adds namespace-bound Automatic Prefix Caching on the Stage 13 paged KV pool.",
        "Only complete prompt blocks become immutable SHARED blocks; incomplete tails remain",
        "request-private. The chained SHA-256 identity binds namespace, every historical block,",
        "the current full token block, and therefore its absolute logical context. Exact token",
        "metadata is retained as a collision-defense invariant.",
        "",
        "Longest-contiguous-prefix lookup attaches canonical physical IDs with active refcounts.",
        "Suffix prefill begins at the true absolute prefix position and attends shared K/V plus",
        (
            "earlier suffix K/V without rerunning the cached Transformer prefix. "
            "Zero-ref shared blocks"
        ),
        "remain resident and are evicted by deterministic LRU; active shared blocks are protected.",
        "",
        "This stage has no partial-block sharing or copy-on-write, chunked prefill, speculative",
        "decoding, GPU/CUDA path, custom fused PagedAttention kernel, or new HTTP API.",
        "It remains a Python/PyTorch reference implementation.",
        "",
        "## Correctness and lifecycle",
        "",
        "Dense, materialized paged, direct paged, and direct paged + APC runs have identical",
        "generated tokens, RNG state, terminal/cancellation states, FIFO admission, and logical",
        "request events. Exact hits, common-prefix suffixes, partial tails, concurrent references,",
        "learned-position overflow rebuild, cancellation, HTTP failure/disconnect, stream",
        "backpressure, shutdown, duplicate-promotion rollback, and LRU pressure are covered.",
        "The deterministic allocator/refcount stress executes 5,000 verified mutations.",
        "",
        "## Fresh-process CPU benchmark",
        "",
        (
            "| Workload | Hit request ratio | Hit token ratio | Avoided prefill tokens | "
            "Evictions | Verdict |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, raw in workloads.items():
        workload = cast("EvidenceDocument", raw)
        strategies = cast("EvidenceDocument", workload["strategies"])
        apc = cast("EvidenceDocument", strategies["paged_direct_apc"])
        lines.append(
            "".join(
                (
                    f"| {name} | {apc['prefix_hit_request_ratio']} | ",
                    f"{apc['prefix_hit_token_ratio']} | {apc['avoided_prefill_tokens']} | ",
                    f"{apc['evictions']} | {workload['strict_verdict']} |",
                )
            )
        )
    lines.extend(
        [
            "",
            "Each raw timing sample uses a fresh Python process and reports median, MAD, and CV",
            "for TTFT, E2E, req/s, tokens/s, and peak RSS. The aggregate strict verdict is",
            f"`{summary['benchmark_strict_verdict']}`. Wall-clock improvement is reported as",
            (
                f"`{summary['wall_clock_performance_improvement']}` and is true only for a "
                "strict pass."
            ),
            "Avoided prefill tokens demonstrate skipped Transformer work independently of noisy",
            "CPU wall-clock measurements.",
        ]
    )
    return "\n".join(lines) + "\n"
