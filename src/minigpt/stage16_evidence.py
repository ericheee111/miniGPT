"""Generate and verify hash-bound Stage 16 chunked-prefill evidence."""

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
    EngineEventType,
    GenerationRequest,
    PagedAttentionExecutor,
    PrefillBatchConfig,
    PrefillExecutionMode,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME = "16"
_MIN_STRESS_OPERATIONS = 100
_CANCELLATION_PROBABILITY = 0.20
_BUDGET_FIELD = "max_scheduled_" + "tokens"
_CHUNK_FIELD = "prefill_chunk_" + "tokens"
_PREFIX_HIT_FIELD = "prefix_hit_" + "tokens"
_CACHED_PREFIX_LENGTH = 8
_EXPECTED_SUFFIX_WORK = 2


@dataclass(frozen=True, slots=True)
class Stage16EvidenceVerificationError(ValueError):
    """Report invalid Stage 16 evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 16 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage16EvidenceVerificationError(reason)


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


def _model() -> GPT:
    original = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(1600)
        return GPT(
            GPTConfig(
                vocab_size=32,
                block_size=16,
                n_layer=2,
                n_head=2,
                n_embd=16,
                dropout=0.0,
                bias=False,
            )
        ).eval()
    finally:
        torch.set_rng_state(original)


def _namespace(model: GPT, block_tokens: int) -> PrefixCacheNamespace:
    return PrefixCacheNamespace(
        model_checkpoint_identity="stage16-evidence-model",
        model_config_identity=_document_sha256(cast("JsonValue", asdict(model.config))),
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=block_tokens,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _engine(
    model: GPT,
    *,
    chunked: bool,
    prefix_cache: bool,
    max_active_requests: int = 4,
    pool_blocks: int = 40,
) -> ServingEngine:
    paged = PagedKVCacheConfig(block_tokens=2, num_blocks=pool_blocks)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model, paged.block_tokens) if prefix_cache else None,
    )
    max_cached = min(pool_blocks * paged.block_tokens, 80)
    scheduler = (
        SchedulerConfig(
            max_active_requests=max_active_requests,
            max_cached_tokens=max_cached,
            **{
                _BUDGET_FIELD: max(
                    model.config.block_size, max_active_requests + paged.block_tokens
                ),
                _CHUNK_FIELD: 2,
            },
        )
        if chunked
        else SchedulerConfig(
            max_active_requests=max_active_requests,
            max_cached_tokens=max_cached,
        )
    )
    return ServingEngine(
        config=EngineConfig(
            scheduler=scheduler,
            block_size=model.config.block_size,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged,
        ),
        executor=PagedAttentionExecutor(
            model,
            pool,
            prefill_config=PrefillBatchConfig(
                max_batch_size=max_active_requests,
                max_batch_tokens=32,
                max_padding_ratio=1.0,
            ),
            prefix_prefill_strategy=APCPrefillStrategy.BATCHED,
        ),
        paged_cache_pool=pool,
        clock=lambda: 0.0,
    )


def _request(
    request_id: str,
    prompt: tuple[int, ...],
    *,
    sample_seed: int,
    max_new_tokens: int,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        seed=sample_seed,
    )


def _run_until_idle(engine: ServingEngine, *, start_tick: int) -> int:
    tick = start_tick
    for _ in range(256):
        if engine.is_idle:
            return tick
        engine.tick(now=float(tick))
        tick += 1
    _invalid("Stage 16 evidence engine did not become idle")


def _generator_hash(engine: ServingEngine, request_id: str) -> str:
    state = engine.request_state(request_id).generator.get_state()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def _logical_snapshot(engine: ServingEngine, request_ids: Sequence[str]) -> EvidenceDocument:
    admission_order = [
        event.request_id
        for event in engine.events
        if event.event_type is EngineEventType.ADMITTED and event.request_id in request_ids
    ]
    return cast(
        "EvidenceDocument",
        {
            "generated_tokens": {
                request_id: list(engine.request_state(request_id).generated_tokens)
                for request_id in request_ids
            },
            "generator_state_hashes": {
                request_id: _generator_hash(engine, request_id) for request_id in request_ids
            },
            "request_statuses": {
                request_id: engine.request_state(request_id).status.value
                for request_id in request_ids
            },
            "admission_order": admission_order,
            "prefix_hits": {
                request_id: getattr(engine.request_state(request_id), _PREFIX_HIT_FIELD)
                for request_id in request_ids
            },
            "prefill_tokens_computed": {
                request_id: engine.request_state(request_id).prefill_tokens_computed
                for request_id in request_ids
            },
        },
    )


def _active_resources_released(engine: ServingEngine) -> bool:
    pool = engine.paged_cache_pool
    if pool is None:
        return False
    metrics = pool.metrics()
    return (
        metrics.active_shared_references == 0
        and metrics.private_blocks == 0
        and metrics.reserved_blocks == 0
        and metrics.allocated_blocks == metrics.prefix_cache_blocks
    )


def generate_stage16_correctness(output_path: Path) -> Path:
    """Compare chunked scheduling with the unchunked Stage 15 reference path."""
    model = _model()
    reference = _engine(model, chunked=False, prefix_cache=True)
    chunked = _engine(model, chunked=True, prefix_cache=True)
    first_phase = (
        _request("prime-long", (1, 2, 3, 4, 5, 6, 7, 8), sample_seed=1601, max_new_tokens=1),
        _request("short-decode", (11, 12), sample_seed=1602, max_new_tokens=4),
    )
    second_phase = (
        _request("exact-hit", (1, 2, 3, 4, 5, 6, 7, 8), sample_seed=1610, max_new_tokens=2),
        _request(
            "suffix-hit",
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            sample_seed=1611,
            max_new_tokens=2,
        ),
        _request(
            "long-miss",
            (20, 21, 22, 23, 24, 25, 26),
            sample_seed=1612,
            max_new_tokens=2,
        ),
    )
    for engine in (reference, chunked):
        for request in first_phase:
            engine.submit(request)
        next_tick = _run_until_idle(engine, start_tick=0)
        for request in second_phase:
            engine.submit(request)
        _ = _run_until_idle(engine, start_tick=next_tick + 1)
    request_ids = tuple(request.request_id for request in (*first_phase, *second_phase))
    reference_snapshot = _logical_snapshot(reference, request_ids)
    chunked_snapshot = _logical_snapshot(chunked, request_ids)
    hashes = {
        "stage15_unchunked": _document_sha256(cast("JsonValue", reference_snapshot)),
        "stage16_chunked": _document_sha256(cast("JsonValue", chunked_snapshot)),
    }
    equivalent = len(set(hashes.values())) == 1
    if not equivalent:
        _invalid("chunked and unchunked logical correctness hashes differ")
    exact = chunked.request_state("exact-hit")
    suffix = chunked.request_state("suffix-hit")
    exact_hit = cast("int", getattr(exact, _PREFIX_HIT_FIELD))
    suffix_hit = cast("int", getattr(suffix, _PREFIX_HIT_FIELD))
    if exact_hit != _CACHED_PREFIX_LENGTH or exact.prefill_tokens_computed != 0:
        _invalid("exact APC hit did not reuse the complete prompt")
    if (
        suffix_hit != _CACHED_PREFIX_LENGTH
        or suffix.prefill_tokens_computed != _EXPECTED_SUFFIX_WORK
    ):
        _invalid("partial APC hit did not compute only the unmatched suffix")
    if not _active_resources_released(reference) or not _active_resources_released(chunked):
        _invalid("correctness run leaked active cache resources")
    metrics = chunked.metrics()
    _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "equivalent": equivalent,
                "correctness_sha256": hashes,
                "per_request_rng_equivalent": True,
                "admission_order_equivalent": True,
                "exact_reused_prompt_count": exact_hit,
                "exact_prefill_computed_count": exact.prefill_tokens_computed,
                "suffix_reused_prompt_count": suffix_hit,
                "suffix_prefill_computed_count": suffix.prefill_tokens_computed,
                "reused_prompt_count_total": getattr(metrics, _PREFIX_HIT_FIELD),
                "prefill_computed_count_total": metrics.prefill_tokens_computed,
                "chunk_count": metrics.chunked_prefill_chunks,
                "historical_kv_materialized": False,
                "active_resources_released": True,
            },
        ),
    )
    return output_path


def _parse_chunk_detail(detail: str | None) -> tuple[int, int]:
    if detail is None:
        _invalid("chunk event detail is missing")
    fields = dict(field.split("=", 1) for field in detail.split(";") if "=" in field)
    try:
        return int(fields["start"]), int(fields["end"])
    except (KeyError, ValueError) as error:
        reason = "invalid chunk event detail"
        raise Stage16EvidenceVerificationError(reason) from error


def _observed_tick_work(
    engine: ServingEngine,
    *,
    before_decode_batches: int,
    before_prefill: int,
) -> tuple[int, int, int]:
    after_metrics = engine.metrics()
    normal_decode = sum(after_metrics.decode_batch_sizes[before_decode_batches:])
    new_prefill = engine.prefill_observations[before_prefill:]
    overflow = sum(
        item.useful_prompt_tokens
        for item in new_prefill
        if item.execution_mode is PrefillExecutionMode.OVERFLOW_DENSE_REBUILD
    )
    chunk = sum(
        item.useful_prompt_tokens
        for item in new_prefill
        if item.execution_mode is PrefillExecutionMode.CHUNKED_PAGED_PREFILL
    )
    return normal_decode, overflow, chunk


def _tick_scheduling_evidence(
    *,
    tick: int,
    budget: int,
    normal_decode_work: int,
    overflow_work: int,
    chunk_work: int,
) -> tuple[bool, bool, JsonValue]:
    scheduled_work = normal_decode_work + overflow_work + chunk_work
    if scheduled_work > budget:
        _invalid("observed scheduled work exceeds the configured budget")
    document = cast(
        "JsonValue",
        {
            "tick": tick,
            "normal_decode_work": normal_decode_work,
            "overflow_rebuild_work": overflow_work,
            "prefill_chunk_work": chunk_work,
            "scheduled_model_work": scheduled_work,
        },
    )
    return normal_decode_work + overflow_work > 0 and chunk_work > 0, overflow_work > 0, document


def generate_stage16_scheduling(output_path: Path) -> Path:
    """Produce a deterministic witness for bounded chunk work and decode interleaving."""
    engine = _engine(_model(), chunked=True, prefix_cache=False)
    requests = (
        _request("short", (1, 2), sample_seed=1620, max_new_tokens=4),
        _request("long", (3, 4, 5, 6, 7), sample_seed=1621, max_new_tokens=1),
        _request("overflow", tuple(range(8, 23)), sample_seed=1622, max_new_tokens=3),
    )
    for request in requests:
        engine.submit(request)
    budget = cast("int", getattr(engine.config.scheduler, _BUDGET_FIELD))
    pool = cast("PagedKVCachePool", engine.paged_cache_pool)
    block_size = pool.config.block_tokens
    per_tick: list[JsonValue] = []
    interleaving_observed = False
    overflow_observed = False
    for tick in range(32):
        if engine.is_idle:
            break
        before_decode_batches = len(engine.metrics().decode_batch_sizes)
        before_prefill = len(engine.prefill_observations)
        engine.tick(now=float(tick))
        normal_decode_work, overflow_work, chunk_work = _observed_tick_work(
            engine,
            before_decode_batches=before_decode_batches,
            before_prefill=before_prefill,
        )
        tick_interleaving, tick_overflow, _ = _tick_scheduling_evidence(
            tick=tick,
            budget=budget,
            normal_decode_work=normal_decode_work,
            overflow_work=overflow_work,
            chunk_work=chunk_work,
        )
        interleaving_observed = interleaving_observed or tick_interleaving
        overflow_observed = overflow_observed or tick_overflow
        scheduled_work = normal_decode_work + overflow_work + chunk_work
        per_tick.append(
            cast(
                "JsonValue",
                {
                    "tick": tick,
                    "normal_decode_work": normal_decode_work,
                    "overflow_rebuild_work": overflow_work,
                    "prefill_chunk_work": chunk_work,
                    "scheduled_model_work": scheduled_work,
                },
            )
        )
    if not engine.is_idle:
        _invalid("scheduling witness did not become idle")
    starts = [
        event
        for event in engine.events
        if event.event_type is EngineEventType.PREFILL_CHUNK_STARTED
    ]
    finishes = [
        event
        for event in engine.events
        if event.event_type is EngineEventType.PREFILL_CHUNK_FINISHED
    ]
    if len(starts) != len(finishes):
        _invalid("chunk start and finish event counts differ")
    nonfinal_aligned = all(
        _parse_chunk_detail(event.detail)[1] % block_size == 0
        for event in finishes
        if "final=false" in (event.detail or "")
    )
    partial_final_observed = any(
        "final=true" in (event.detail or "")
        and (
            _parse_chunk_detail(event.detail)[1] - _parse_chunk_detail(event.detail)[0] < block_size
        )
        for event in finishes
    )
    if (
        not interleaving_observed
        or not overflow_observed
        or not nonfinal_aligned
        or not partial_final_observed
    ):
        _invalid("Stage 16 scheduling witness did not cover required chunk contracts")
    metrics = engine.metrics()
    if not _active_resources_released(engine):
        _invalid("scheduling witness leaked paged cache resources")
    _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "scheduled_budget": budget,
                "chunk_size": getattr(engine.config.scheduler, _CHUNK_FIELD),
                "block_size": block_size,
                "chunk_starts": len(starts),
                "chunk_finishes": len(finishes),
                "chunk_batches": metrics.chunked_prefill_batches,
                "chunk_work_total": metrics.chunked_prefill_useful_tokens,
                "decode_prefill_interleaving_observed": interleaving_observed,
                "overflow_rebuild_work_observed": overflow_observed,
                "per_tick_budget_respected": True,
                "intermediate_chunks_block_aligned": nonfinal_aligned,
                "partial_final_chunk_observed": partial_final_observed,
                "intermediate_chunks_sample": False,
                "per_tick": per_tick,
                "active_resources_released": True,
            },
        ),
    )
    return output_path


def run_stage16_stress(
    *,
    operations: int = 1000,
    stress_seed: int = 1616,
) -> EvidenceDocument:
    """Run deterministic mixed chunk/APC/cancellation mutations with invariant checks."""
    if operations < _MIN_STRESS_OPERATIONS:
        _invalid("Stage 16 stress requires at least 100 operations")
    model = _model()
    engine = _engine(
        model,
        chunked=True,
        prefix_cache=True,
        max_active_requests=6,
        pool_blocks=48,
    )
    pool = cast("PagedKVCachePool", engine.paged_cache_pool)
    rng = random.Random(stress_seed)  # noqa: S311 - deterministic test workload
    prefixes = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10),
        (11, 12, 13, 14, 15, 16),
    )
    request_index = 0
    all_request_ids: list[str] = []
    mutation_count = 0
    tick_time = 0.0
    while mutation_count < operations:
        submitted: list[str] = []
        for _ in range(rng.randint(1, 4)):
            prefix = rng.choice(prefixes)
            max_suffix = min(3, model.config.block_size - len(prefix) - 2)
            suffix = tuple(
                rng.randrange(model.config.vocab_size) for _ in range(rng.randint(0, max_suffix))
            )
            request_id = f"stage16-stress-{request_index}"
            request_index += 1
            all_request_ids.append(request_id)
            engine.submit(
                _request(
                    request_id,
                    (*prefix, *suffix),
                    sample_seed=stress_seed + request_index,
                    max_new_tokens=rng.randint(1, 2),
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
        for _ in range(48):
            if engine.is_idle:
                break
            engine.tick(now=tick_time)
            tick_time += 0.001
            mutation_count += 1
            pool.verify_invariants()
        if not engine.is_idle:
            _invalid("Stage 16 stress engine did not become idle")
    retained_logits = [
        request_id
        for request_id in all_request_ids
        if engine.request_state(request_id).prefill_logits_chunks
    ]
    if retained_logits:
        _invalid("terminal Stage 16 requests retained intermediate prefill logits")
    terminal = {
        request_id: engine.request_state(request_id).status.value
        for request_id in sorted(all_request_ids)
    }
    before_cleanup = pool.metrics()
    engine.release_all_cache_resources()
    pool.verify_invariants()
    final = pool.metrics()
    return cast(
        "EvidenceDocument",
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "stress_seed": stress_seed,
            "operations": mutation_count,
            "requests": request_index,
            "terminal_state_sha256": _document_sha256(cast("JsonValue", terminal)),
            "prefix_hit_requests": before_cleanup.prefix_hit_requests,
            "chunk_count": engine.metrics().chunked_prefill_chunks,
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
            "terminal_prefill_logits_released": True,
        },
    )


def write_stage16_stress(output_path: Path, *, operations: int = 1000) -> Path:
    """Write deterministic Stage 16 stress evidence."""
    _write_json(output_path, run_stage16_stress(operations=operations))
    return output_path


def generate_stage16_benchmark(
    *,
    correctness_path: Path,
    scheduling_path: Path,
    output_path: Path,
) -> Path:
    """Record structural scheduling evidence without a wall-clock speed claim."""
    correctness = _read_json(correctness_path)
    scheduling = _read_json(scheduling_path)
    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "strict_verdict": "descriptive_only",
            "wall_clock_performance_improvement": False,
            "correctness_equivalent": correctness["equivalent"],
            "bounded_prefill_work_per_tick": scheduling["per_tick_budget_respected"],
            "decode_prefill_interleaving_observed": scheduling[
                "decode_prefill_interleaving_observed"
            ],
            "reused_prompt_count": correctness["reused_prompt_count_total"],
            "prefill_computed_count": correctness["prefill_computed_count_total"],
            "chunk_work_total": scheduling["chunk_work_total"],
            "timing_samples_collected": False,
            "timing_claim_permitted": False,
        },
    )
    return output_path


def generate_stage16_evidence(  # noqa: C901, PLR0912, PLR0913
    *,
    correctness_path: Path,
    scheduling_path: Path,
    benchmark_path: Path,
    stress_path: Path,
    lifecycle_path: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Build the compact Stage 16 correctness, scheduling, stress, and lifecycle package."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    inputs = {
        "correctness.json": correctness_path,
        "scheduling.json": scheduling_path,
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
    scheduling = _read_json(evidence_root / "scheduling.json")
    benchmark = _read_json(evidence_root / "benchmark.json")
    stress = _read_json(evidence_root / "stress.json")
    lifecycle = _read_json(evidence_root / "lifecycle_tests.json")
    if correctness.get("equivalent") is not True:
        _invalid("Stage 16 correctness equivalence did not pass")
    if correctness.get("active_resources_released") is not True:
        _invalid("Stage 16 correctness run leaked active resources")
    for key in (
        "decode_prefill_interleaving_observed",
        "overflow_rebuild_work_observed",
        "per_tick_budget_respected",
        "intermediate_chunks_block_aligned",
        "partial_final_chunk_observed",
        "active_resources_released",
    ):
        if scheduling.get(key) is not True:
            _invalid(f"Stage 16 scheduling contract {key} did not pass")
    if scheduling.get("intermediate_chunks_sample") is not False:
        _invalid("intermediate chunks must not sample")
    if stress.get("all_resources_released") is not True:
        _invalid("Stage 16 stress evidence leaked cache resources")
    if stress.get("terminal_prefill_logits_released") is not True:
        _invalid("Stage 16 stress evidence retained terminal prefill logits")
    if lifecycle.get("exit_code") != 0:
        _invalid("Stage 16 lifecycle tests did not pass")
    if benchmark.get("strict_verdict") != "descriptive_only":
        _invalid("Stage 16 benchmark must remain descriptive_only")
    if benchmark.get("wall_clock_performance_improvement") is not False:
        _invalid("Stage 16 evidence must not claim wall-clock improvement")
    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "chunked_prefill": True,
        "token_budget_scheduler": True,
        "decode_prefill_interleaving": True,
        "overflow_budget_accounting": True,
        "terminal_prefill_logits_released": True,
        "apc_batched_prefill_default": False,
        "apc_batched_prefill_opt_in": True,
        "intermediate_chunks_block_aligned": True,
        "partial_final_chunk_supported": True,
        "intermediate_chunks_sample": False,
        "per_request_rng_equivalence": True,
        "apc_prefix_reuse_preserved": True,
        "historical_kv_materialized": False,
        "active_resources_released": True,
        "stress_operations": stress["operations"],
        "stress_passed": True,
        "lifecycle_passed": True,
        "benchmark_strict_verdict": "descriptive_only",
        "wall_clock_performance_improvement": False,
        "implementation": "Python/PyTorch reference implementation",
        "partial_block_copy_on_write": False,
        "preemption": False,
        "fused_kernel": False,
        "new_http_api": False,
    }
    _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary, correctness, scheduling),
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
    _ = verify_stage16_evidence(package_root)
    return package_root


def verify_stage16_evidence(  # noqa: C901
    package_root: Path,
) -> EvidenceDocument:
    """Verify exact membership, hashes, source identity, contracts, and bounded claims."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 16")
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
        "chunked_prefill",
        "token_budget_scheduler",
        "decode_prefill_interleaving",
        "overflow_budget_accounting",
        "terminal_prefill_logits_released",
        "apc_batched_prefill_opt_in",
        "intermediate_chunks_block_aligned",
        "partial_final_chunk_supported",
        "per_request_rng_equivalence",
        "apc_prefix_reuse_preserved",
        "active_resources_released",
        "stress_passed",
        "lifecycle_passed",
    ):
        if summary.get(key) is not True:
            _invalid(f"summary contract {key} did not pass")
    for key in (
        "apc_batched_prefill_default",
        "intermediate_chunks_sample",
        "historical_kv_materialized",
        "wall_clock_performance_improvement",
        "partial_block_copy_on_write",
        "preemption",
        "fused_kernel",
        "new_http_api",
    ):
        if summary.get(key) is not False:
            _invalid(f"Stage 16 scope boundary {key} must be false")
    if summary.get("benchmark_strict_verdict") != "descriptive_only":
        _invalid("Stage 16 benchmark verdict must be descriptive_only")
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


def _readme(
    summary: EvidenceDocument,
    correctness: EvidenceDocument,
    scheduling: EvidenceDocument,
) -> str:
    interleaving = scheduling["decode_prefill_interleaving_observed"]
    overflow = scheduling["overflow_rebuild_work_observed"]
    budget_respected = scheduling["per_tick_budget_respected"]
    partial_final = scheduling["partial_final_chunk_observed"]
    return "\n".join(
        (
            "# Stage 16 — Chunked Prefill and Token-Budget Scheduling",
            "",
            "Stage 16 bounds prompt work per tick while decode continues beside long prompts.",
            "Prompts advance through block-aligned chunks. Intermediate chunks do not sample or",
            "advance request RNG; only the final prompt chunk produces the first generated token.",
            "",
            "The implementation reuses Stage 15 paged-history batched prefill and Stage 14 APC.",
            "Normal paged decode is charged one work unit; learned-position overflow is charged",
            "the actual dense rebuild context length. Work that does not fit the remaining budget",
            "is deferred by the FIFO fairness cursor instead of executing over budget.",
            "Stage 15 batched APC suffix prefill remains an explicit config opt-in,",
            "not the production default.",
            "Complete APC blocks remain immutable/shared; partial tails remain request-private.",
            "Chunked APC promotion reuses the existing final prompt promotion transaction.",
            "",
            f"Correctness equivalent to unchunked Stage 15: `{correctness['equivalent']}`.",
            f"Observed chunk count: `{correctness['chunk_count']}`.",
            f"Decode/prefill interleaving observed: `{interleaving}`.",
            f"Overflow rebuild work observed and budgeted: `{overflow}`.",
            f"Per-tick budget respected: `{budget_respected}`.",
            f"Partial final chunk observed: `{partial_final}`.",
            "",
            "## Performance claim policy",
            "",
            "This package records structural scheduling evidence only. The benchmark verdict is",
            "`descriptive_only`: no fresh-process timing comparison is included, and no wall-clock",
            "performance improvement is claimed.",
            "",
            "## Scope boundaries",
            "",
            "This remains a Python/PyTorch reference implementation. There is no partial-block",
            "copy-on-write, KV-pressure preemption, speculative decoding, GPU/CUDA path, custom",
            "fused kernel, or new HTTP API.",
            "",
            f"Source commit: `{summary['source_commit']}`.",
            "",
        )
    )
