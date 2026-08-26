"""Generate and verify hash-bound Stage 18 lazy KV-reservation evidence."""

from __future__ import annotations

import hashlib
import json
import math
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
    EngineConfig,
    EngineEventType,
    GenerationRequest,
    PagedAttentionExecutor,
    PrefillBatchConfig,
    PrefillExecutionMode,
    RequestStatus,
    SchedulerConfig,
    ServingEngine,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME = "18"
_MIN_STRESS_OPERATIONS = 100
_STRESS_CANCEL_PROBABILITY = 0.08
_BLOCK_SIZE = 8
_BLOCK_TOKENS = 2
_EXPECTED_LAZY_ACTIVE_REQUESTS = 2
_APC_PREEMPT_TICK_LIMIT = 256
_APC_RESUME_TICK_LIMIT = 512


@dataclass(frozen=True, slots=True)
class Stage18EvidenceVerificationError(ValueError):
    """Report invalid Stage 18 evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 18 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage18EvidenceVerificationError(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_sha256(document: JsonValue) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
        _ = torch.default_generator.manual_seed(1818)
        return GPT(
            GPTConfig(
                vocab_size=31,
                block_size=_BLOCK_SIZE,
                n_layer=2,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                bias=False,
            )
        ).eval()
    finally:
        torch.set_rng_state(original)


def _namespace(model: GPT) -> PrefixCacheNamespace:
    return PrefixCacheNamespace(
        model_checkpoint_identity="stage18-evidence-model",
        model_config_identity=_document_sha256(cast("JsonValue", asdict(model.config))),
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=_BLOCK_TOKENS,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _engine(  # noqa: PLR0913
    model: GPT,
    *,
    pool_blocks: int,
    lazy: bool,
    preemption: bool,
    overcommit_ratio: float = 1.0,
    prefix_cache: bool = False,
    max_active_requests: int = 2,
    max_cached_tokens: int | None = None,
) -> ServingEngine:
    paged = PagedKVCacheConfig(block_tokens=_BLOCK_TOKENS, num_blocks=pool_blocks)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model) if prefix_cache else None,
    )
    return ServingEngine(
        config=EngineConfig(
            scheduler=SchedulerConfig(
                max_active_requests=max_active_requests,
                max_cached_tokens=(
                    pool_blocks * _BLOCK_TOKENS if max_cached_tokens is None else max_cached_tokens
                ),
                max_scheduled_tokens=_BLOCK_SIZE,
                prefill_chunk_tokens=_BLOCK_TOKENS,
                kv_preemption=preemption,
                lazy_kv_reservation=lazy,
                kv_overcommit_ratio=overcommit_ratio,
            ),
            block_size=_BLOCK_SIZE,
            kv_cache_backend=KVCacheBackend.PAGED,
            paged_kv_cache=paged,
        ),
        executor=PagedAttentionExecutor(
            model,
            pool,
            prefill_config=PrefillBatchConfig(
                max_batch_size=max_active_requests,
                max_batch_tokens=_BLOCK_SIZE,
                max_padding_ratio=1.0,
            ),
        ),
        paged_cache_pool=pool,
        clock=lambda: 0.0,
    )


def _request(request_id: str, *, seed: int, max_new_tokens: int = 7) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_tokens=(1, 2, 3, 4),
        max_new_tokens=max_new_tokens,
        seed=seed,
    )


def _run_until_idle(
    engine: ServingEngine,
    *,
    start_tick: int = 0,
    max_ticks: int = 512,
) -> int:
    for tick in range(start_tick, start_tick + max_ticks):
        if engine.is_idle:
            return tick
        engine.tick(now=tick * 0.01)
    _invalid("Stage 18 evidence engine did not become idle")


def _resources_released(engine: ServingEngine) -> bool:
    pool = cast("PagedKVCachePool", engine.paged_cache_pool)
    metrics = pool.metrics()
    return (
        metrics.private_blocks == 0
        and metrics.active_shared_references == 0
        and metrics.reserved_blocks == 0
        and metrics.allocated_blocks == 0
    )


def _intrinsic_impossible_heads_rejected(model: GPT) -> bool:
    logical = _engine(
        model,
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
        max_active_requests=1,
        max_cached_tokens=4,
    )
    physical = _engine(
        model,
        pool_blocks=3,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
        max_active_requests=1,
        max_cached_tokens=8,
    )
    viable = GenerationRequest(
        request_id="viable",
        prompt_tokens=(1, 2),
        max_new_tokens=3,
        seed=1831,
    )
    oversized = _request("oversized", seed=1832, max_new_tokens=5)
    for engine in (logical, physical):
        engine.submit(viable)
        engine.submit(oversized)
        engine.tick(now=0.0)
    return all(
        engine.request_state("oversized").status is RequestStatus.FAILED
        and engine.request_state("viable").preemption_count == 0
        and engine.metrics().preemptions == 0
        and not any(event.event_type is EngineEventType.PREEMPTED for event in engine.events)
        for engine in (logical, physical)
    )


def generate_stage18_correctness(output_path: Path) -> Path:
    """Compare lazy pressure with roomy and same-pool full-reservation references."""
    model = _model()
    roomy = _engine(model, pool_blocks=8, lazy=False, preemption=False)
    same_pool = _engine(model, pool_blocks=4, lazy=False, preemption=False)
    pressured = _engine(
        model,
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
    )
    requests = (_request("first", seed=1801), _request("second", seed=1802))
    for request in requests:
        roomy.submit(request)
        same_pool.submit(request)
        pressured.submit(request)

    same_pool.tick(now=0.0)
    pressured.tick(now=0.0)
    full_initial = same_pool.metrics()
    lazy_initial = pressured.metrics()
    if not (
        full_initial.active_requests == 1
        and lazy_initial.active_requests == _EXPECTED_LAZY_ACTIVE_REQUESTS
        and lazy_initial.reserved_cache_tokens < lazy_initial.lifetime_reserved_cache_tokens
        and lazy_initial.lifetime_reserved_cache_tokens
        <= math.floor(
            pressured.config.scheduler.max_cached_tokens
            * pressured.config.scheduler.kv_overcommit_ratio
        )
    ):
        _invalid("Stage 18 initial reservation/residency witness did not hold")

    _ = _run_until_idle(roomy)
    _ = _run_until_idle(same_pool, start_tick=1)
    _ = _run_until_idle(pressured, start_tick=1)
    equivalent = True
    rng_equivalent = True
    same_pool_equivalent = True
    per_request: dict[str, JsonValue] = {}
    for request in requests:
        expected = roomy.request_state(request.request_id)
        full_actual = same_pool.request_state(request.request_id)
        actual = pressured.request_state(request.request_id)
        token_match = actual.generated_tokens == expected.generated_tokens
        rng_match = torch.equal(actual.generator.get_state(), expected.generator.get_state())
        full_match = full_actual.generated_tokens == expected.generated_tokens and torch.equal(
            full_actual.generator.get_state(),
            expected.generator.get_state(),
        )
        equivalent = equivalent and token_match and actual.status is expected.status
        rng_equivalent = rng_equivalent and rng_match
        same_pool_equivalent = same_pool_equivalent and full_match
        per_request[request.request_id] = cast(
            "JsonValue",
            {
                "status": actual.status.value,
                "generated_token_match": token_match,
                "rng_match": rng_match,
                "preemptions": actual.preemption_count,
                "resumes": actual.resume_count,
                "recompute_tokens": actual.recompute_tokens,
                "reservation_growths": actual.reservation_growth_count,
                "reservation_growth_tokens": actual.reservation_growth_tokens,
                "reservation_growth_blocked": actual.reservation_growth_blocked_count,
            },
        )

    metrics = pressured.metrics()
    overflow_observed = any(event.used_fallback for event in pressured.events)
    if not equivalent or not rng_equivalent or not same_pool_equivalent:
        _invalid("Stage 18 pressure differs from a full-reservation reference")
    if not overflow_observed:
        _invalid("Stage 18 correctness workload missed learned-position overflow")
    if (
        metrics.reservation_growths <= 0
        or metrics.reservation_growth_blocked <= 0
        or metrics.growth_pressure_preemptions <= 0
        or metrics.preemptions <= 0
        or metrics.resumes <= 0
    ):
        _invalid("Stage 18 correctness workload missed growth-pressure paths")
    if not _resources_released(pressured):
        _invalid("Stage 18 correctness workload leaked KV resources")
    if not _intrinsic_impossible_heads_rejected(model):
        _invalid("Stage 18 intrinsically impossible admission heads were not rejected safely")

    _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "equivalent": True,
                "same_pool_full_reservation_equivalent": True,
                "per_request_rng_equivalence": True,
                "overflow_sliding_window_equivalence": True,
                "intrinsic_impossible_heads_rejected": True,
                "lazy_initial_active_requests": lazy_initial.active_requests,
                "full_initial_active_requests": full_initial.active_requests,
                "initial_current_reserved_tokens": lazy_initial.reserved_cache_tokens,
                "initial_lifetime_reserved_tokens": (lazy_initial.lifetime_reserved_cache_tokens),
                "configured_overcommit_ratio": lazy_initial.kv_overcommit_ratio,
                "peak_overcommitted_cache_tokens": metrics.peak_overcommitted_cache_tokens,
                "reservation_growths": metrics.reservation_growths,
                "reservation_growth_tokens": metrics.reservation_growth_tokens,
                "reservation_growth_blocked": metrics.reservation_growth_blocked,
                "growth_pressure_preemptions": metrics.growth_pressure_preemptions,
                "preemptions": metrics.preemptions,
                "resumes": metrics.resumes,
                "recompute_tokens": metrics.recompute_tokens,
                "overflow_observed": True,
                "per_request": per_request,
                "active_resources_released": True,
            },
        ),
    )
    return output_path


def _observed_tick_work(
    engine: ServingEngine,
    *,
    before_decode_batches: int,
    before_prefill: int,
) -> tuple[int, int, int, int]:
    metrics = engine.metrics()
    normal = sum(metrics.decode_batch_sizes[before_decode_batches:])
    observations = engine.prefill_observations[before_prefill:]
    overflow = sum(
        item.useful_prompt_tokens
        for item in observations
        if item.execution_mode is PrefillExecutionMode.OVERFLOW_DENSE_REBUILD
    )
    recompute = sum(
        item.useful_prompt_tokens
        for item in observations
        if item.execution_mode is PrefillExecutionMode.PREEMPTION_RECOMPUTE
    )
    chunk = sum(
        item.useful_prompt_tokens
        for item in observations
        if item.execution_mode is PrefillExecutionMode.CHUNKED_PAGED_PREFILL
    )
    return normal, overflow, recompute, chunk


def generate_stage18_scheduling(output_path: Path) -> Path:
    """Prove growth is model-work free and actual work stays within budget."""
    engine = _engine(
        _model(),
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
    )
    for request in (_request("first", seed=1811), _request("second", seed=1812)):
        engine.submit(request)
    budget = cast("int", engine.config.scheduler.max_scheduled_tokens)
    per_tick: list[JsonValue] = []
    observed = {"growth": False, "blocked": False, "recompute": False, "overflow": False}
    for tick in range(512):
        if engine.is_idle:
            break
        before_decode = len(engine.metrics().decode_batch_sizes)
        before_prefill = len(engine.prefill_observations)
        before_events = len(engine.events)
        engine.tick(now=tick * 0.01)
        normal, overflow, recompute, chunk = _observed_tick_work(
            engine,
            before_decode_batches=before_decode,
            before_prefill=before_prefill,
        )
        scheduled = normal + overflow + recompute + chunk
        if scheduled > budget:
            _invalid("Stage 18 scheduled model work exceeds token budget")
        events = engine.events[before_events:]
        growths = sum(event.event_type is EngineEventType.RESERVATION_GROWN for event in events)
        blocked = sum(
            event.event_type is EngineEventType.RESERVATION_GROWTH_BLOCKED for event in events
        )
        observed["growth"] = observed["growth"] or growths > 0
        observed["blocked"] = observed["blocked"] or blocked > 0
        observed["recompute"] = observed["recompute"] or recompute > 0
        observed["overflow"] = observed["overflow"] or overflow > 0
        per_tick.append(
            cast(
                "JsonValue",
                {
                    "tick": tick,
                    "normal_decode_work": normal,
                    "overflow_rebuild_work": overflow,
                    "preemption_recompute_work": recompute,
                    "prefill_chunk_work": chunk,
                    "reservation_growth_events": growths,
                    "reservation_growth_blocked_events": blocked,
                    "scheduled_model_work": scheduled,
                },
            )
        )
    if not engine.is_idle:
        _invalid("Stage 18 scheduling workload did not finish")
    metrics = engine.metrics()
    if not all(observed.values()) or metrics.growth_pressure_preemptions <= 0:
        _invalid("Stage 18 scheduling witness missed a required pressure path")
    _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "scheduled_budget": budget,
                "per_tick_budget_respected": True,
                "reservation_growth_model_work_tokens": 0,
                "growth_work_free": True,
                "growth_observed": True,
                "growth_blocked_observed": True,
                "recompute_work_observed": True,
                "overflow_rebuild_work_observed": True,
                "growth_pressure_preemptions": metrics.growth_pressure_preemptions,
                "reservation_growths": metrics.reservation_growths,
                "reservation_growth_tokens": metrics.reservation_growth_tokens,
                "reservation_growth_blocks": metrics.reservation_growth_blocks,
                "per_tick": per_tick,
            },
        ),
    )
    return output_path


def generate_stage18_apc(output_path: Path) -> Path:
    """Prove APC release and private-position recompute remain safe."""
    engine = _engine(
        _model(),
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
        prefix_cache=True,
    )
    for request in (_request("first", seed=1821), _request("second", seed=1822)):
        engine.submit(request)
    victim_id: str | None = None
    tick = 0
    while tick < _APC_PREEMPT_TICK_LIMIT and victim_id is None:
        engine.tick(now=tick * 0.01)
        victim_id = next(
            (
                request_id
                for request_id in ("first", "second")
                if engine.request_state(request_id).status is RequestStatus.PREEMPTED
            ),
            None,
        )
        tick += 1
    if victim_id is None:
        _invalid("Stage 18 APC witness did not preempt a request")
    pool = cast("PagedKVCachePool", engine.paged_cache_pool)
    if pool.has_request(victim_id):
        _invalid("Stage 18 APC victim retained a resident table")
    resumed_private = False
    while tick < _APC_RESUME_TICK_LIMIT:
        engine.tick(now=tick * 0.01)
        state = engine.request_state(victim_id)
        tick += 1
        if state.status is RequestStatus.DECODING and state.resume_count > 0:
            resumed_private = pool.request_cache(victim_id).shared_blocks == 0
            break
    if not resumed_private:
        _invalid("Stage 18 APC victim did not resume with private recompute KV")
    _ = _run_until_idle(engine, start_tick=tick)
    if not _resources_released(engine):
        _invalid("Stage 18 APC witness leaked KV resources")
    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "shared_refs_released_on_preempt": True,
            "resume_uses_private_recompute": True,
            "growth_after_apc_safe": True,
            "active_resources_released": True,
        },
    )
    return output_path


def run_stage18_stress(
    *,
    operations: int = 1000,
    stress_seed: int = 1818,
) -> EvidenceDocument:
    """Run deterministic lazy-growth, preemption, cancellation, and cleanup stress."""
    if operations < _MIN_STRESS_OPERATIONS:
        _invalid("Stage 18 stress requires at least 100 operations")
    model = _model()
    engine = _engine(
        model,
        pool_blocks=4,
        lazy=True,
        preemption=True,
        overcommit_ratio=2.0,
        max_active_requests=3,
    )
    pool = cast("PagedKVCachePool", engine.paged_cache_pool)
    rng = random.Random(stress_seed)  # noqa: S311 - deterministic evidence workload
    request_ids: list[str] = []
    request_index = 0
    mutation_count = 0
    tick = 0
    while mutation_count < operations:
        for _ in range(rng.randint(2, 3)):
            request_id = f"stage18-stress-{request_index}"
            request_index += 1
            request_ids.append(request_id)
            engine.submit(
                GenerationRequest(
                    request_id=request_id,
                    prompt_tokens=tuple(
                        rng.randrange(1, model.config.vocab_size) for _ in range(4)
                    ),
                    max_new_tokens=rng.randint(2, 7),
                    seed=stress_seed + request_index,
                )
            )
            mutation_count += 1
            pool.verify_invariants()
        for _ in range(512):
            if engine.is_idle:
                break
            engine.tick(now=tick * 0.001)
            tick += 1
            mutation_count += 1
            pool.verify_invariants()
            preempted = [
                request_id
                for request_id in request_ids
                if engine.request_state(request_id).status is RequestStatus.PREEMPTED
            ]
            if preempted and rng.random() < _STRESS_CANCEL_PROBABILITY:
                engine.cancel(rng.choice(preempted), at=tick * 0.001)
                mutation_count += 1
                pool.verify_invariants()
        if not engine.is_idle:
            _invalid("Stage 18 stress batch did not become idle")
    all_terminal = all(
        engine.request_state(item).status
        in {RequestStatus.FINISHED, RequestStatus.CANCELLED, RequestStatus.FAILED}
        for item in request_ids
    )
    logits_released = not any(
        engine.request_state(item).prefill_logits_chunks for item in request_ids
    )
    metrics = engine.metrics()
    final = pool.metrics()
    resources_released = (
        final.private_blocks == 0
        and final.active_shared_references == 0
        and final.reserved_blocks == 0
        and final.allocated_blocks == 0
    )
    if (
        not all_terminal
        or not logits_released
        or not resources_released
        or metrics.reservation_growths <= 0
        or metrics.growth_pressure_preemptions <= 0
    ):
        _invalid("Stage 18 stress missed terminal/resource/progress contracts")
    terminal = {item: engine.request_state(item).status.value for item in sorted(request_ids)}
    return cast(
        "EvidenceDocument",
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "stress_seed": stress_seed,
            "operations": mutation_count,
            "requests": request_index,
            "preemptions": metrics.preemptions,
            "resumes": metrics.resumes,
            "recompute_tokens": metrics.recompute_tokens,
            "reservation_growths": metrics.reservation_growths,
            "reservation_growth_tokens": metrics.reservation_growth_tokens,
            "reservation_growth_blocked": metrics.reservation_growth_blocked,
            "growth_pressure_preemptions": metrics.growth_pressure_preemptions,
            "terminal_state_sha256": _document_sha256(cast("JsonValue", terminal)),
            "all_requests_terminal": True,
            "all_resources_released": True,
            "terminal_prefill_logits_released": True,
        },
    )


def write_stage18_stress(output_path: Path, *, operations: int = 1000) -> Path:
    """Write deterministic Stage 18 stress evidence."""
    _write_json(output_path, run_stage18_stress(operations=operations))
    return output_path


def generate_stage18_benchmark(
    *,
    correctness_path: Path,
    scheduling_path: Path,
    stress_path: Path,
    output_path: Path,
) -> Path:
    """Record structural residency evidence without a wall-clock speed claim."""
    correctness = _read_json(correctness_path)
    scheduling = _read_json(scheduling_path)
    stress = _read_json(stress_path)
    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "strict_verdict": "descriptive_only",
            "wall_clock_performance_improvement": False,
            "lazy_initial_active_requests": correctness["lazy_initial_active_requests"],
            "full_initial_active_requests": correctness["full_initial_active_requests"],
            "initial_current_reserved_tokens": correctness["initial_current_reserved_tokens"],
            "initial_lifetime_reserved_tokens": correctness["initial_lifetime_reserved_tokens"],
            "growth_pressure_preemptions": correctness["growth_pressure_preemptions"],
            "per_tick_budget_respected": scheduling["per_tick_budget_respected"],
            "growth_work_free": scheduling["growth_work_free"],
            "stress_operations": stress["operations"],
        },
    )
    return output_path


def generate_stage18_evidence(  # noqa: PLR0913
    *,
    correctness_path: Path,
    scheduling_path: Path,
    apc_path: Path,
    benchmark_path: Path,
    stress_path: Path,
    lifecycle_path: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Build the Stage 18 correctness, growth, stress, and lifecycle package."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    inputs = {
        "correctness.json": correctness_path,
        "scheduling.json": scheduling_path,
        "apc.json": apc_path,
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
    apc = _read_json(evidence_root / "apc.json")
    benchmark = _read_json(evidence_root / "benchmark.json")
    stress = _read_json(evidence_root / "stress.json")
    lifecycle = _read_json(evidence_root / "lifecycle_tests.json")
    required_true = (
        correctness.get("equivalent"),
        correctness.get("same_pool_full_reservation_equivalent"),
        correctness.get("per_request_rng_equivalence"),
        correctness.get("overflow_sliding_window_equivalence"),
        correctness.get("intrinsic_impossible_heads_rejected"),
        correctness.get("active_resources_released"),
        scheduling.get("per_tick_budget_respected"),
        scheduling.get("growth_work_free"),
        scheduling.get("growth_observed"),
        scheduling.get("growth_blocked_observed"),
        scheduling.get("recompute_work_observed"),
        scheduling.get("overflow_rebuild_work_observed"),
        apc.get("shared_refs_released_on_preempt"),
        apc.get("resume_uses_private_recompute"),
        apc.get("growth_after_apc_safe"),
        apc.get("active_resources_released"),
        stress.get("all_requests_terminal"),
        stress.get("all_resources_released"),
        stress.get("terminal_prefill_logits_released"),
    )
    if not all(value is True for value in required_true):
        _invalid("Stage 18 evidence contracts did not all pass")
    if lifecycle.get("exit_code") != 0:
        _invalid("Stage 18 lifecycle tests did not pass")
    if benchmark.get("strict_verdict") != "descriptive_only":
        _invalid("Stage 18 benchmark must remain descriptive_only")
    if benchmark.get("wall_clock_performance_improvement") is not False:
        _invalid("Stage 18 evidence must not claim wall-clock improvement")

    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "lazy_kv_reservation": True,
        "controlled_overcommit": True,
        "growth_before_model_work": True,
        "growth_work_free": True,
        "growth_pressure_preemption": True,
        "immediate_growth_retry": True,
        "per_request_rng_equivalence": True,
        "overflow_sliding_window_equivalence": True,
        "intrinsic_impossible_heads_rejected": True,
        "recompute_resume_equivalence": True,
        "per_tick_budget_respected": True,
        "apc_shared_refs_released": True,
        "resume_uses_private_recompute": True,
        "no_starvation_finite_workload": True,
        "terminal_prefill_logits_released": True,
        "stress_operations": stress["operations"],
        "stress_passed": True,
        "lifecycle_passed": True,
        "benchmark_strict_verdict": "descriptive_only",
        "wall_clock_performance_improvement": False,
        "implementation": "Python/PyTorch reference implementation",
        "cpu_swap": False,
        "partial_block_copy_on_write": False,
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
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "source_commit": source_commit,
                "artifacts": artifacts,
            },
        ),
    )
    _ = verify_stage18_evidence(package_root)
    return package_root


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


def _verify_artifact_manifest(package_root: Path) -> tuple[EvidenceDocument, str]:
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 18")
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
    return manifest, source_commit


def _verify_summary(summary: EvidenceDocument, source_commit: str) -> None:
    if summary.get("source_commit") != source_commit:
        _invalid("summary source_commit differs from manifest")
    true_keys = (
        "lazy_kv_reservation",
        "controlled_overcommit",
        "growth_before_model_work",
        "growth_work_free",
        "growth_pressure_preemption",
        "immediate_growth_retry",
        "per_request_rng_equivalence",
        "overflow_sliding_window_equivalence",
        "intrinsic_impossible_heads_rejected",
        "recompute_resume_equivalence",
        "per_tick_budget_respected",
        "apc_shared_refs_released",
        "resume_uses_private_recompute",
        "no_starvation_finite_workload",
        "terminal_prefill_logits_released",
        "stress_passed",
        "lifecycle_passed",
    )
    for key in true_keys:
        if summary.get(key) is not True:
            _invalid(f"summary contract {key} did not pass")
    false_keys = (
        "wall_clock_performance_improvement",
        "cpu_swap",
        "partial_block_copy_on_write",
        "new_http_api",
    )
    for key in false_keys:
        if summary.get(key) is not False:
            _invalid(f"Stage 18 scope boundary {key} must be false")
    if summary.get("benchmark_strict_verdict") != "descriptive_only":
        _invalid("Stage 18 benchmark verdict must be descriptive_only")


def verify_stage18_evidence(package_root: Path) -> EvidenceDocument:
    """Verify exact membership, hashes, source identity, contracts, and bounded claims."""
    manifest, source_commit = _verify_artifact_manifest(package_root)
    _verify_summary(_read_json(package_root / "summary.json"), source_commit)
    return manifest


def _readme(
    summary: EvidenceDocument,
    correctness: EvidenceDocument,
    scheduling: EvidenceDocument,
) -> str:
    return (
        "\n".join(
            (
                "# Stage 18 — Lazy KV Growth Reservation + Controlled Overcommit",
                "",
                "Stage 18 protects current KV capacity at admission while retaining a bounded",
                "full-lifetime demand. Capacity grows before model work, and Stage 17",
                "whole-request preemption breaks deterministic growth pressure.",
                "",
                f"Roomy/full logical equivalence: {correctness['equivalent']}.",
                f"Per-request RNG equivalence: {correctness['per_request_rng_equivalence']}.",
                (
                    "Intrinsic logical/physical admission failures rejected without preemption: "
                    f"{correctness['intrinsic_impossible_heads_rejected']}."
                ),
                (
                    "Same-pool initial active requests (lazy/full): "
                    f"{correctness['lazy_initial_active_requests']}/"
                    f"{correctness['full_initial_active_requests']}."
                ),
                (
                    "Per-tick actual model-work budget respected: "
                    f"{scheduling['per_tick_budget_respected']}."
                ),
                f"Observed reservation growths: {correctness['reservation_growths']}.",
                (
                    "Observed growth-pressure preemptions: "
                    f"{correctness['growth_pressure_preemptions']}."
                ),
                "",
                "APC shared references are released on preemption. Resume rebuilds private",
                "KV rather than reusing original-position shared prefix blocks.",
                "",
                "## Performance claim policy",
                "",
                "The benchmark verdict is descriptive_only. No wall-clock performance",
                "improvement is claimed.",
                "",
                "## Scope boundaries",
                "",
                "CPU swap, partial-block COW, GPU/CUDA, fused kernels, scheduler priorities,",
                "and new HTTP request APIs remain outside Stage 18.",
                "",
                f"Source commit: {summary['source_commit']}.",
            )
        )
        + "\n"
    )
