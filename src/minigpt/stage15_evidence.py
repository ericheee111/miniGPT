"""Generate and verify hash-bound Stage 15 cache-aware prefill evidence."""

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
from minigpt.model import GPT
from minigpt.paged_kv_cache import (
    KVCacheBackend,
    PagedKVCacheConfig,
    PagedKVCachePool,
    PrefixCacheNamespace,
)
from minigpt.serving import (
    APCPrefillStrategy,
    EngineConfig,
    GenerationRequest,
    PagedAttentionExecutor,
    PrefillBatchConfig,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.serving_simulator import (
    SimulationResult,
    load_simulator_config,
    run_cache_aware_prefill_equivalence,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME = "15"
CORRECTNESS_CONFIG = "serving_cache_aware_prefill.yaml"
_MIN_BATCH_SIZE = 2
_MIN_STRESS_OPERATIONS = 100
_CANCELLATION_PROBABILITY = 0.20


@dataclass(frozen=True, slots=True)
class Stage15EvidenceVerificationError(ValueError):
    """Report invalid Stage 15 evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 15 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage15EvidenceVerificationError(reason)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _logical_document(result: SimulationResult) -> EvidenceDocument:
    return cast(
        "EvidenceDocument",
        {
            "generated_tokens": result.generated_tokens,
            "generator_state_hashes": result.generator_state_hashes,
            "request_statuses": {
                key: value.value for key, value in result.request_statuses.items()
            },
            "admission_order": result.admission_order,
            "events": [
                {
                    "event_type": event.event_type.value,
                    "request_id": event.request_id,
                    "status": event.status.value,
                    "token_id": event.token_id,
                    "detail": event.detail,
                    "used_fallback": event.used_fallback,
                    "active_requests": event.active_requests,
                    "waiting_requests": event.waiting_requests,
                    "cached_tokens": event.cached_tokens,
                    "reserved_cache_tokens": event.reserved_cache_tokens,
                }
                for event in result.events
            ],
        },
    )


def _strategy_metrics(result: SimulationResult) -> EvidenceDocument:
    metrics = result.metrics
    return cast(
        "EvidenceDocument",
        {
            "model_calls": metrics.cache_aware_prefill_model_calls,
            "cache_aware_prefill_batches": metrics.cache_aware_prefill_batches,
            "suffix_prefill_batch_sizes": metrics.suffix_prefill_batch_sizes,
            "average_suffix_prefill_batch_size": metrics.average_suffix_prefill_batch_size,
            "max_suffix_prefill_batch_size": metrics.max_suffix_prefill_batch_size,
            "suffix_useful_tokens": metrics.suffix_useful_tokens,
            "suffix_padded_tokens": metrics.suffix_padded_tokens,
            "suffix_padding_waste_ratio": metrics.suffix_padding_waste_ratio,
            "exact_cache_hit_requests": metrics.exact_cache_hit_requests,
            "batched_suffix_requests": metrics.batched_suffix_requests,
            "prefix_hit_tokens": metrics.prefix_hit_tokens,
            "prefill_tokens_computed": metrics.prefill_tokens_computed,
            "avoided_prefill_tokens": metrics.avoided_prefill_tokens,
        },
    )


def generate_stage15_correctness(
    *,
    config_root: Path,
    work_root: Path,
    output_path: Path,
) -> Path:
    """Bind sequential/batched APC output, RNG, prefix identity, and ownership."""
    config = load_simulator_config(config_root / CORRECTNESS_CONFIG)
    comparison = run_cache_aware_prefill_equivalence(config, output_dir=work_root)
    hashes = {
        "apc_sequential": _document_sha256(_logical_document(comparison.sequential)),
        "apc_batched": _document_sha256(_logical_document(comparison.batched)),
    }
    if len(set(hashes.values())) != 1:
        _invalid("sequential and batched APC logical correctness hashes differ")
    sequential = _strategy_metrics(comparison.sequential)
    batched = _strategy_metrics(comparison.batched)
    sequential_calls = cast("int", sequential["model_calls"])
    batched_calls = cast("int", batched["model_calls"])
    if not batched_calls < sequential_calls:
        _invalid("Stage 15 correctness workload did not reduce prefill model calls")
    metrics = comparison.batched.metrics
    active_resources_released = (
        metrics.active_shared_references == 0
        and metrics.reserved_blocks == 0
        and metrics.allocated_blocks == metrics.prefix_cache_blocks
    )
    if not active_resources_released:
        _invalid("Stage 15 correctness run leaked active refs, private blocks, or reservations")
    _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "config": CORRECTNESS_CONFIG,
                "equivalent": comparison.equivalent,
                "correctness_sha256": hashes,
                "checked_contracts": list(comparison.checked_contracts),
                "strategies": {
                    "apc_sequential": sequential,
                    "apc_batched": batched,
                },
                "model_call_reduction": sequential_calls - batched_calls,
                "cached_prefix_transformer_work_skipped": True,
                "historical_kv_materialized": False,
                "exact_hits_zero_model_calls": metrics.exact_cache_hit_requests > 0,
                "overflow_fallback_observed": any(
                    event.used_fallback for event in comparison.batched.events
                ),
                "active_resources_released": active_resources_released,
            },
        ),
    )
    return output_path


def generate_stage15_batching(*, correctness_path: Path, output_path: Path) -> Path:
    """Project direct structural batching evidence without inferring execution mode."""
    correctness = _read_json(correctness_path)
    strategies = cast("EvidenceDocument", correctness["strategies"])
    sequential = cast("EvidenceDocument", strategies["apc_sequential"])
    batched = cast("EvidenceDocument", strategies["apc_batched"])
    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "fake_batching_guard_passed": (
                cast("int", batched["max_suffix_prefill_batch_size"]) >= _MIN_BATCH_SIZE
                and cast("int", batched["model_calls"]) < cast("int", sequential["model_calls"])
            ),
            "sequential_model_calls": cast("int", sequential["model_calls"]),
            "batched_model_calls": cast("int", batched["model_calls"]),
            "model_call_reduction": correctness["model_call_reduction"],
            "suffix_prefill_batch_sizes": batched["suffix_prefill_batch_sizes"],
            "average_suffix_prefill_batch_size": batched["average_suffix_prefill_batch_size"],
            "max_suffix_prefill_batch_size": batched["max_suffix_prefill_batch_size"],
            "suffix_useful_tokens": batched["suffix_useful_tokens"],
            "suffix_padded_tokens": batched["suffix_padded_tokens"],
            "suffix_padding_waste_ratio": batched["suffix_padding_waste_ratio"],
            "avoided_prefill_tokens": batched["avoided_prefill_tokens"],
            "exact_cache_hit_requests": batched["exact_cache_hit_requests"],
        },
    )
    return output_path


def _stress_model() -> GPT:
    original = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(1515)
        model = GPT(
            GPTConfig(
                vocab_size=32,
                block_size=12,
                n_layer=1,
                n_head=1,
                n_embd=8,
                dropout=0.0,
                bias=False,
            )
        ).eval()
    finally:
        torch.set_rng_state(original)
    return model


def run_stage15_stress(*, operations: int = 1000, seed: int = 1515) -> EvidenceDocument:
    """Run a deterministic mixed APC batch lifecycle with invariants after every mutation."""
    if operations < _MIN_STRESS_OPERATIONS:
        _invalid("Stage 15 stress requires at least 100 operations")
    model = _stress_model()
    paged = PagedKVCacheConfig(block_tokens=2, num_blocks=48)
    namespace = PrefixCacheNamespace(
        model_checkpoint_identity="stage15-stress-model",
        model_config_identity=_document_sha256(cast("JsonValue", asdict(model.config))),
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=paged.block_tokens,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )
    pool = PagedKVCachePool.from_model(paged, model, prefix_cache_namespace=namespace)
    engine = ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(max_active_requests=6, max_cached_tokens=72),
            block_size=model.config.block_size,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged,
        ),
        executor=PagedAttentionExecutor(
            model,
            pool,
            prefill_config=PrefillBatchConfig(
                max_batch_size=6,
                max_batch_tokens=48,
                max_padding_ratio=1.0,
            ),
            prefix_prefill_strategy=APCPrefillStrategy.BATCHED,
        ),
        paged_cache_pool=pool,
        clock=lambda: 0.0,
    )
    rng = random.Random(seed)  # noqa: S311 - deterministic test workload, not cryptography
    prefixes = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10), (11, 12, 13, 14, 15, 16))
    request_index = 0
    all_request_ids: list[str] = []
    mutation_count = 0
    tick_time = 0.0
    while mutation_count < operations:
        batch_size = rng.randint(1, 5)
        submitted: list[str] = []
        for _ in range(batch_size):
            prefix = rng.choice(prefixes)
            suffix_length = rng.randint(0, min(4, model.config.block_size - len(prefix)))
            suffix = tuple(rng.randrange(model.config.vocab_size) for _ in range(suffix_length))
            request_id = f"stress-{request_index}"
            request_index += 1
            all_request_ids.append(request_id)
            engine.submit(
                GenerationRequest(
                    request_id=request_id,
                    prompt_tokens=(*prefix, *suffix),
                    max_new_tokens=rng.randint(1, 4),
                    seed=seed + request_index,
                )
            )
            submitted.append(request_id)
            mutation_count += 1
            pool.verify_invariants()
            if mutation_count >= operations:
                break
        if submitted and rng.random() < _CANCELLATION_PROBABILITY:
            engine.cancel(rng.choice(submitted), at=tick_time)
            mutation_count += 1
            pool.verify_invariants()
        for _ in range(32):
            if engine.is_idle:
                break
            engine.tick(now=tick_time)
            tick_time += 0.001
            mutation_count += 1
            pool.verify_invariants()
        if not engine.is_idle:
            _invalid("Stage 15 stress engine did not become idle")
    terminal = {
        request_id: engine.request_state(request_id).status.value
        for request_id in sorted(all_request_ids)
    }
    logical_hash = _document_sha256(cast("JsonValue", terminal))
    before_cleanup = pool.metrics()
    engine.release_all_cache_resources()
    pool.verify_invariants()
    final = pool.metrics()
    return {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "seed": seed,
        "operations": mutation_count,
        "requests": request_index,
        "terminal_state_sha256": logical_hash,
        "prefix_hits": before_cleanup.prefix_hit_requests,
        "batch_model_calls": engine.metrics().cache_aware_prefill_model_calls,
        "active_refs_before_cleanup": before_cleanup.active_shared_references,
        "active_refs_after_cleanup": final.active_shared_references,
        "private_blocks_after_cleanup": final.private_blocks,
        "reservations_after_cleanup": final.reserved_blocks,
        "allocated_blocks_after_cleanup": final.allocated_blocks,
        "free_blocks_after_cleanup": final.free_blocks,
        "total_blocks": final.total_blocks,
        "all_resources_released": (
            final.active_shared_references == 0
            and final.private_blocks == 0
            and final.reserved_blocks == 0
            and final.allocated_blocks == 0
            and final.free_blocks == final.total_blocks
        ),
    }


def write_stage15_stress(output_path: Path, *, operations: int = 1000) -> Path:
    """Write deterministic Stage 15 stress evidence."""
    _write_json(output_path, run_stage15_stress(operations=operations))
    return output_path


def generate_stage15_evidence(  # noqa: C901, PLR0913
    *,
    correctness_path: Path,
    batching_path: Path,
    benchmark_path: Path,
    stress_path: Path,
    lifecycle_path: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Build the compact correctness, batching, lifecycle, stress, and benchmark package."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    inputs = {
        "correctness.json": correctness_path,
        "batching.json": batching_path,
        "benchmark.json": benchmark_path,
        "stress.json": stress_path,
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
    batching = _read_json(evidence_root / "batching.json")
    benchmark = _read_json(evidence_root / "benchmark.json")
    stress = _read_json(evidence_root / "stress.json")
    lifecycle = _read_json(evidence_root / "lifecycle_tests.json")
    if correctness.get("equivalent") is not True:
        _invalid("sequential/batched APC correctness did not pass")
    if correctness.get("active_resources_released") is not True:
        _invalid("correctness run leaked active cache resources")
    if batching.get("fake_batching_guard_passed") is not True:
        _invalid("fake-batching guard did not pass")
    if stress.get("all_resources_released") is not True or lifecycle.get("exit_code") != 0:
        _invalid("stress and lifecycle evidence must pass without leaks")
    verdict = benchmark.get("strict_verdict")
    if verdict not in {"pass", "fail", "not_comparable"}:
        _invalid("benchmark strict_verdict is invalid")
    if benchmark.get("correctness_equivalent") is not True:
        _invalid("benchmark logical correctness differs")
    if benchmark.get("model_call_reduction") is not True:
        _invalid("benchmark did not prove model-call reduction")
    speedup_claim = verdict == "pass"
    if benchmark.get("wall_clock_performance_improvement") is not speedup_claim:
        _invalid("benchmark improvement claim differs from strict verdict")
    totals = _benchmark_totals(benchmark)
    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "batched_paged_history_primitive": True,
        "batched_prefill_opt_in": True,
        "batched_prefill_default": False,
        "variable_past_and_suffix_lengths": True,
        "absolute_learned_positions": True,
        "historical_kv_materialized": False,
        "suffix_batch_budgeting": True,
        "exact_hits_zero_model_prefill": True,
        "sequential_batched_logical_equivalence": True,
        "per_request_rng_equivalence": True,
        "fake_batching_guard_passed": True,
        "all_correctness_checks_passed": True,
        "active_resources_released": True,
        "stress_operations": stress["operations"],
        "stress_passed": True,
        "lifecycle_passed": True,
        "sequential_model_calls": totals["sequential_model_calls"],
        "batched_model_calls": totals["batched_model_calls"],
        "model_call_reduction": totals["model_call_reduction"],
        "average_suffix_prefill_batch_size": totals["average_batch_size"],
        "max_suffix_prefill_batch_size": totals["max_batch_size"],
        "suffix_useful_tokens": totals["useful_tokens"],
        "suffix_padded_tokens": totals["padded_tokens"],
        "avoided_prefill_tokens": totals["avoided_tokens"],
        "benchmark_strict_verdict": verdict,
        "wall_clock_performance_improvement": speedup_claim,
        "implementation": "Python/PyTorch reference implementation",
        "chunked_prefill": False,
        "partial_block_copy_on_write": False,
        "preemption": False,
        "fused_kernel": False,
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
    _ = verify_stage15_evidence(package_root)
    return package_root


def verify_stage15_evidence(package_root: Path) -> EvidenceDocument:  # noqa: C901, PLR0912
    """Verify exact membership, hashes, source identity, and bounded performance claims."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 15")
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
        "batched_paged_history_primitive",
        "batched_prefill_opt_in",
        "variable_past_and_suffix_lengths",
        "absolute_learned_positions",
        "suffix_batch_budgeting",
        "exact_hits_zero_model_prefill",
        "sequential_batched_logical_equivalence",
        "per_request_rng_equivalence",
        "fake_batching_guard_passed",
        "all_correctness_checks_passed",
        "active_resources_released",
        "stress_passed",
        "lifecycle_passed",
    ):
        if summary.get(key) is not True:
            _invalid(f"summary contract {key} did not pass")
    for key in (
        "batched_prefill_default",
        "historical_kv_materialized",
        "chunked_prefill",
        "partial_block_copy_on_write",
        "preemption",
        "fused_kernel",
    ):
        if summary.get(key) is not False:
            _invalid(f"Stage 15 scope boundary {key} must be false")
    sequential_calls = summary.get("sequential_model_calls")
    batched_calls = summary.get("batched_model_calls")
    if not isinstance(sequential_calls, int) or not isinstance(batched_calls, int):
        _invalid("summary model-call counts must be integers")
    if not batched_calls < sequential_calls:
        _invalid("summary must prove fewer batched model calls")
    verdict = summary.get("benchmark_strict_verdict")
    claim = summary.get("wall_clock_performance_improvement")
    if claim is not (verdict == "pass"):
        _invalid("summary wall-clock claim differs from strict verdict")
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


def _benchmark_totals(benchmark: EvidenceDocument) -> dict[str, int | float]:
    workloads = cast("EvidenceDocument", benchmark["workloads"])
    totals: dict[str, int | float] = {
        "sequential_model_calls": 0,
        "batched_model_calls": 0,
        "model_call_reduction": 0,
        "batched_requests": 0,
        "batches": 0,
        "max_batch_size": 0,
        "useful_tokens": 0,
        "padded_tokens": 0,
        "avoided_tokens": 0,
        "average_batch_size": 0.0,
    }
    for raw in workloads.values():
        workload = cast("EvidenceDocument", raw)
        strategies = cast("EvidenceDocument", workload["strategies"])
        sequential = cast("EvidenceDocument", strategies["apc_sequential"])
        batched = cast("EvidenceDocument", strategies["apc_batched"])
        sequential_calls = cast("int", sequential["cache_aware_prefill_model_calls"])
        batched_calls = cast("int", batched["cache_aware_prefill_model_calls"])
        totals["sequential_model_calls"] += sequential_calls
        totals["batched_model_calls"] += batched_calls
        totals["model_call_reduction"] += sequential_calls - batched_calls
        totals["batched_requests"] += cast("int", batched["batched_suffix_requests"])
        totals["batches"] += cast("int", batched["cache_aware_prefill_batches"])
        totals["max_batch_size"] = max(
            totals["max_batch_size"], cast("int", batched["max_suffix_prefill_batch_size"])
        )
        totals["useful_tokens"] += cast("int", batched["suffix_useful_tokens"])
        totals["padded_tokens"] += cast("int", batched["suffix_padded_tokens"])
        totals["avoided_tokens"] += cast("int", batched["avoided_prefill_tokens"])
    batches = cast("int", totals["batches"])
    totals["average_batch_size"] = (
        cast("int", totals["batched_requests"]) / batches if batches else 0.0
    )
    return totals


def _readme(summary: EvidenceDocument, benchmark: EvidenceDocument) -> str:
    workloads = cast("EvidenceDocument", benchmark["workloads"])
    lines = [
        "# Stage 15 — Cache-Aware Batched Paged Prefill",
        "",
        "Stage 15 removes Stage 14's sequential APC suffix-prefill regression. Cached-prefix",
        "Transformer work is still skipped, while multiple variable-length suffix segments execute",
        "in one batched paged-history model call. Historical K/V remains in physical paged blocks",
        "and is never dense-materialized on the normal path.",
        "",
        "Batch admission uses computed suffix tokens for max batch size, token budget, and padding",
        (
            "ratio. Exact full-boundary hits use cached boundary logits and perform zero model "
            "prefill."
        ),
        "All valid suffix logits and suffix-only K/V deltas are scattered through the existing",
        "owner-thread write, promotion, duplicate canonicalization, and refcount transactions.",
        "Because the aggregate strict benchmark verdict is not `pass`, sequential APC suffix",
        "prefill remains the production default. The Stage 15 config opts into batched prefill",
        "explicitly; model-call reduction alone does not justify a silent default change.",
        "",
        "",
        "This remains a Python/PyTorch reference implementation. There is no chunked prefill,",
        "partial-block sharing or copy-on-write, KV-pressure preemption, speculative decoding,",
        "GPU/CUDA path, custom fused kernel, or new HTTP API.",
        "",
        "## Fresh-process benchmark",
        "",
        "| Workload | Sequential calls | Batched calls | Avg batch | Max batch | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, raw in workloads.items():
        workload = cast("EvidenceDocument", raw)
        strategies = cast("EvidenceDocument", workload["strategies"])
        sequential = cast("EvidenceDocument", strategies["apc_sequential"])
        batched = cast("EvidenceDocument", strategies["apc_batched"])
        lines.append(
            "".join(
                (
                    f"| {name} | {sequential['cache_aware_prefill_model_calls']} | ",
                    f"{batched['cache_aware_prefill_model_calls']} | ",
                    f"{batched['average_suffix_prefill_batch_size']} | ",
                    f"{batched['max_suffix_prefill_batch_size']} | ",
                    f"{workload['strict_verdict']} |",
                )
            )
        )
    lines.extend(
        [
            "",
            (
                "Every raw timing sample runs in a fresh Python process and retains median, MAD, "
                "and CV"
            ),
            "for TTFT, E2E, req/s, tokens/s, and peak RSS. Avoided prefix work and model-call",
            "reduction are reported independently from wall-clock timing.",
            "",
            f"Aggregate strict verdict: `{summary['benchmark_strict_verdict']}`.",
            (
                "Wall-clock performance improvement claim: "
                f"`{summary['wall_clock_performance_improvement']}`."
            ),
            (
                "When the aggregate verdict is not `pass`, this package makes no wall-clock "
                "performance improvement claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)
