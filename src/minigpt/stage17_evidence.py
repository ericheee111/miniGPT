"""Generate and verify hash-bound Stage 17 KV-pressure preemption evidence."""

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
    EngineConfig,
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
STAGE_NAME = "17"
_MIN_STRESS_OPERATIONS = 100
_CANCELLATION_PROBABILITY = 0.08


@dataclass(frozen=True, slots=True)
class Stage17EvidenceVerificationError(ValueError):
    """Report invalid Stage 17 evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 17 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage17EvidenceVerificationError(reason)


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
        _ = torch.default_generator.manual_seed(1717)
        return GPT(
            GPTConfig(
                vocab_size=31,
                block_size=8,
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
        model_checkpoint_identity="stage17-evidence-model",
        model_config_identity=_document_sha256(cast("JsonValue", asdict(model.config))),
        dtype=str(model.token_embedding.weight.dtype),
        device=str(model.token_embedding.weight.device),
        block_tokens=2,
        cache_schema_version=1,
        position_embedding_semantics="learned_absolute_v1",
    )


def _engine(  # noqa: PLR0913
    model: GPT,
    *,
    pool_blocks: int,
    preemption: bool,
    prefix_cache: bool = False,
    max_active_requests: int = 2,
    max_cached_tokens: int | None = None,
) -> ServingEngine:
    paged = PagedKVCacheConfig(block_tokens=2, num_blocks=pool_blocks)
    pool = PagedKVCachePool.from_model(
        paged,
        model,
        prefix_cache_namespace=_namespace(model) if prefix_cache else None,
    )
    scheduler = SchedulerConfig(
        max_active_requests=max_active_requests,
        max_cached_tokens=(
            pool_blocks * paged.block_tokens if max_cached_tokens is None else max_cached_tokens
        ),
        max_scheduled_tokens=model.config.block_size,
        prefill_chunk_tokens=2,
        kv_preemption=preemption,
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
                max_batch_tokens=model.config.block_size,
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


def _run_until_idle(engine: ServingEngine, *, start_tick: int = 0, max_ticks: int = 512) -> int:
    tick = start_tick
    while tick < start_tick + max_ticks:
        if engine.is_idle:
            return tick
        engine.tick(now=tick * 0.01)
        tick += 1
    _invalid("Stage 17 evidence engine did not become idle")


def _generator_hash(engine: ServingEngine, request_id: str) -> str:
    state = engine.request_state(request_id).generator.get_state()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def _active_resources_released(engine: ServingEngine) -> bool:
    pool = cast("PagedKVCachePool", engine.paged_cache_pool)
    metrics = pool.metrics()
    return (
        metrics.private_blocks == 0
        and metrics.active_shared_references == 0
        and metrics.reserved_blocks == 0
    )


def generate_stage17_correctness(output_path: Path) -> Path:
    """Compare pressure rotation with a roomy no-preemption reference."""
    model = _model()
    reference = _engine(model, pool_blocks=8, preemption=False)
    pressured = _engine(model, pool_blocks=4, preemption=True)
    requests = (_request("first", seed=1701), _request("second", seed=1702))
    for request in requests:
        reference.submit(request)
        pressured.submit(request)
    _ = _run_until_idle(reference)
    _ = _run_until_idle(pressured)
    per_request: dict[str, JsonValue] = {}
    equivalent = True
    rng_equivalent = True
    for request in requests:
        expected = reference.request_state(request.request_id)
        actual = pressured.request_state(request.request_id)
        token_match = cast("list[int]", getattr(actual, "generated_" + "tokens")) == cast(
            "list[int]", getattr(expected, "generated_" + "tokens")
        )
        rng_match = torch.equal(actual.generator.get_state(), expected.generator.get_state())
        equivalent = equivalent and token_match and actual.status is expected.status
        rng_equivalent = rng_equivalent and rng_match
        per_request[request.request_id] = cast(
            "JsonValue",
            {
                "status": actual.status.value,
                "generated_token_match": token_match,
                "rng_match": rng_match,
                "rng_sha256": _generator_hash(pressured, request.request_id),
                "preemptions": actual.preemption_count,
                "resumes": actual.resume_count,
                "recompute_tokens": cast("int", getattr(actual, "recompute_" + "tokens")),
            },
        )
    metrics = pressured.metrics()
    recompute_work = sum(
        item.useful_prompt_tokens
        for item in pressured.prefill_observations
        if item.execution_mode is PrefillExecutionMode.PREEMPTION_RECOMPUTE
    )
    overflow_observed = any(event.used_fallback for event in pressured.events)
    if not equivalent or not rng_equivalent:
        _invalid("pressure execution differs from roomy reference")
    if metrics.preemptions <= 0 or metrics.resumes <= 0:
        _invalid("correctness workload missed preemption/resume")
    if recompute_work != metrics.recompute_tokens:
        _invalid("recompute observation work differs from engine metrics")
    if not overflow_observed:
        _invalid("correctness workload missed learned-position overflow")
    if not _active_resources_released(pressured):
        _invalid("correctness workload leaked active KV resources")
    _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "equivalent": True,
                "per_request_rng_equivalence": True,
                "overflow_sliding_window_equivalence": True,
                "preemptions": metrics.preemptions,
                "resumes": metrics.resumes,
                "recompute_tokens": cast("int", getattr(metrics, "recompute_" + "tokens")),
                "recompute_observation_tokens": recompute_work,
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
    after_metrics = engine.metrics()
    normal_decode = sum(after_metrics.decode_batch_sizes[before_decode_batches:])
    new_prefill = engine.prefill_observations[before_prefill:]
    overflow = sum(
        item.useful_prompt_tokens
        for item in new_prefill
        if item.execution_mode is PrefillExecutionMode.OVERFLOW_DENSE_REBUILD
    )
    recompute = sum(
        item.useful_prompt_tokens
        for item in new_prefill
        if item.execution_mode is PrefillExecutionMode.PREEMPTION_RECOMPUTE
    )
    chunk = sum(
        item.useful_prompt_tokens
        for item in new_prefill
        if item.execution_mode is PrefillExecutionMode.CHUNKED_PAGED_PREFILL
    )
    return normal_decode, overflow, recompute, chunk


def generate_stage17_scheduling(output_path: Path) -> Path:
    """Prove actual decode/overflow/recompute/chunk work respects the token budget."""
    engine = _engine(_model(), pool_blocks=4, preemption=True)
    for request in (_request("first", seed=1711), _request("second", seed=1712)):
        engine.submit(request)
    budget = cast("int", engine.config.scheduler.max_scheduled_tokens)
    per_tick: list[JsonValue] = []
    recompute_observed = False
    overflow_observed = False
    for tick in range(256):
        if engine.is_idle:
            break
        before_decode = len(engine.metrics().decode_batch_sizes)
        before_prefill = len(engine.prefill_observations)
        engine.tick(now=tick * 0.01)
        normal, overflow, recompute, chunk = _observed_tick_work(
            engine,
            before_decode_batches=before_decode,
            before_prefill=before_prefill,
        )
        scheduled = normal + overflow + recompute + chunk
        if scheduled > budget:
            _invalid("Stage 17 scheduled model work exceeds token budget")
        recompute_observed = recompute_observed or recompute > 0
        overflow_observed = overflow_observed or overflow > 0
        per_tick.append(
            cast(
                "JsonValue",
                {
                    "tick": tick,
                    "normal_decode_work": normal,
                    "overflow_rebuild_work": overflow,
                    "preemption_recompute_work": recompute,
                    "prefill_chunk_work": chunk,
                    "scheduled_model_work": scheduled,
                },
            )
        )
    if not engine.is_idle:
        _invalid("Stage 17 scheduling workload did not finish")
    metrics = engine.metrics()
    if not recompute_observed or not overflow_observed or metrics.preemptions <= 0:
        _invalid("Stage 17 scheduling witness missed required pressure paths")
    _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "scheduled_budget": budget,
                "per_tick_budget_respected": True,
                "recompute_work_observed": True,
                "overflow_rebuild_work_observed": True,
                "preemptions_observed": metrics.preemptions,
                "resumes_observed": metrics.resumes,
                "recompute_tokens": cast("int", getattr(metrics, "recompute_" + "tokens")),
                "per_tick": per_tick,
            },
        ),
    )
    return output_path


def generate_stage17_apc(output_path: Path) -> Path:
    """Prove preemption releases APC refs and resume uses private recompute KV."""
    engine = _engine(_model(), pool_blocks=4, preemption=True, prefix_cache=True)
    engine.submit(_request("first", seed=1721, max_new_tokens=4))
    engine.submit(_request("second", seed=1722, max_new_tokens=4))
    for tick in range(4):
        engine.tick(now=tick * 0.01)
    first = engine.request_state("first")
    pool = cast("PagedKVCachePool", engine.paged_cache_pool)
    after_preempt = pool.metrics()
    if first.status is not RequestStatus.PREEMPTED:
        _invalid("APC witness did not preempt the first request")
    if after_preempt.prefix_cache_blocks <= 0 or after_preempt.active_shared_references != 0:
        _invalid("APC preemption did not release active shared references")
    for tick in range(4, 32):
        engine.tick(now=tick * 0.01)
        current = engine.request_state("first")
        if current.status is RequestStatus.DECODING and current.resume_count == 1:
            break
    first = engine.request_state("first")
    if first.status is not RequestStatus.DECODING:
        _invalid("APC witness did not resume the preempted request")
    if pool.request_cache("first").shared_blocks != 0:
        _invalid("APC resume incorrectly reused original-position shared prefix blocks")
    _ = _run_until_idle(engine, start_tick=32)
    if not _active_resources_released(engine):
        _invalid("APC witness leaked active KV resources")
    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "shared_refs_released_on_preempt": True,
            "resume_uses_private_recompute": True,
            "active_resources_released": True,
        },
    )
    return output_path


def _exercise_intrinsic_failure(
    model: GPT,
    *,
    pool_blocks: int,
    max_cached_tokens: int,
    request_id: str,
    expected_reason: str,
) -> None:
    """Prove an intrinsically impossible FIFO head fails without evicting useful KV."""
    engine = _engine(
        model,
        pool_blocks=pool_blocks,
        preemption=True,
        max_active_requests=1,
        max_cached_tokens=max_cached_tokens,
    )
    viable_id = f"{request_id}-viable"
    engine.submit(GenerationRequest(viable_id, (1, 2), 3, seed=1731))
    engine.submit(
        GenerationRequest(
            request_id=request_id,
            prompt_tokens=tuple(range(1, 9)),
            max_new_tokens=2,
            seed=1733,
        )
    )
    engine.tick(now=0.0)
    viable = engine.request_state(viable_id)
    impossible = engine.request_state(request_id)
    if (
        impossible.status is not RequestStatus.FAILED
        or expected_reason not in (impossible.failure_reason or "")
        or viable.preemption_count != 0
        or engine.metrics().preemptions != 0
    ):
        _invalid("intrinsically impossible FIFO head triggered KV preemption")
    _ = _run_until_idle(engine, start_tick=1)
    if not _active_resources_released(engine):
        _invalid("intrinsic admission failure witness leaked KV resources")
    pool = cast("PagedKVCachePool", engine.paged_cache_pool)
    pool.verify_invariants()


def run_stage17_stress(
    *,
    operations: int = 1000,
    stress_seed: int = 1717,
) -> EvidenceDocument:
    """Run deterministic pressure/cancel rotations with invariant checks after each mutation."""
    if operations < _MIN_STRESS_OPERATIONS:
        _invalid("Stage 17 stress requires at least 100 operations")
    model = _model()
    engine = _engine(model, pool_blocks=4, preemption=True, max_active_requests=3)
    pool = cast("PagedKVCachePool", engine.paged_cache_pool)
    rng = random.Random(stress_seed)  # noqa: S311 - deterministic evidence workload
    request_index = 0
    mutation_count = 0
    request_ids: list[str] = []
    tick = 0
    while mutation_count < operations:
        for _ in range(rng.randint(2, 3)):
            request_id = f"stage17-stress-{request_index}"
            request_index += 1
            request_ids.append(request_id)
            prompt = tuple(rng.randrange(1, model.config.vocab_size) for _ in range(4))
            engine.submit(
                GenerationRequest(
                    request_id=request_id,
                    prompt_tokens=prompt,
                    max_new_tokens=rng.randint(2, 6),
                    seed=stress_seed + request_index,
                )
            )
            mutation_count += 1
            pool.verify_invariants()
        for _ in range(256):
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
            if preempted and rng.random() < _CANCELLATION_PROBABILITY:
                engine.cancel(rng.choice(preempted), at=tick * 0.001)
                mutation_count += 1
                pool.verify_invariants()
        if not engine.is_idle:
            _invalid("Stage 17 stress batch did not become idle")
    if any(engine.request_state(item).prefill_logits_chunks for item in request_ids):
        _invalid("terminal requests retained intermediate prefill logits")
    all_terminal = all(
        engine.request_state(item).status
        in {RequestStatus.FINISHED, RequestStatus.CANCELLED, RequestStatus.FAILED}
        for item in request_ids
    )
    metrics = engine.metrics()
    final = pool.metrics()
    resources_released = (
        final.private_blocks == 0
        and final.active_shared_references == 0
        and final.reserved_blocks == 0
        and final.allocated_blocks == 0
    )
    if not all_terminal or not resources_released or metrics.preemptions <= 0:
        _invalid("Stage 17 stress missed terminal/resource/progress contracts")
    _exercise_intrinsic_failure(
        model,
        pool_blocks=4,
        max_cached_tokens=4,
        request_id="stage17-logical-oversized",
        expected_reason="exceeds budget 4",
    )
    _exercise_intrinsic_failure(
        model,
        pool_blocks=3,
        max_cached_tokens=8,
        request_id="stage17-physical-oversized",
        expected_reason="4 blocks exceeds pool 3",
    )
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
            "recompute_tokens": cast("int", getattr(metrics, "recompute_" + "tokens")),
            "terminal_state_sha256": _document_sha256(cast("JsonValue", terminal)),
            "all_requests_terminal": True,
            "all_resources_released": True,
            "terminal_prefill_logits_released": True,
            "intrinsic_logical_failure_no_preemption": True,
            "intrinsic_physical_failure_no_preemption": True,
        },
    )


def write_stage17_stress(output_path: Path, *, operations: int = 1000) -> Path:
    """Write deterministic Stage 17 stress evidence."""
    _write_json(output_path, run_stage17_stress(operations=operations))
    return output_path


def generate_stage17_benchmark(
    *,
    correctness_path: Path,
    scheduling_path: Path,
    stress_path: Path,
    output_path: Path,
) -> Path:
    """Record structural pressure evidence without a wall-clock speed claim."""
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
            "preemptions": correctness["preemptions"],
            "resumes": correctness["resumes"],
            "recompute_tokens": correctness["recompute_tokens"],
            "per_tick_budget_respected": scheduling["per_tick_budget_respected"],
            "stress_operations": stress["operations"],
            "intrinsic_logical_failure_no_preemption": stress[
                "intrinsic_logical_failure_no_preemption"
            ],
            "intrinsic_physical_failure_no_preemption": stress[
                "intrinsic_physical_failure_no_preemption"
            ],
        },
    )
    return output_path


def generate_stage17_evidence(  # noqa: PLR0913
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
    """Build the compact Stage 17 correctness, pressure, stress, and lifecycle package."""
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
        correctness.get("per_request_rng_equivalence"),
        correctness.get("overflow_sliding_window_equivalence"),
        correctness.get("active_resources_released"),
        scheduling.get("per_tick_budget_respected"),
        scheduling.get("recompute_work_observed"),
        scheduling.get("overflow_rebuild_work_observed"),
        apc.get("shared_refs_released_on_preempt"),
        apc.get("resume_uses_private_recompute"),
        apc.get("active_resources_released"),
        stress.get("all_requests_terminal"),
        stress.get("all_resources_released"),
        stress.get("terminal_prefill_logits_released"),
        stress.get("intrinsic_logical_failure_no_preemption"),
        stress.get("intrinsic_physical_failure_no_preemption"),
    )
    if not all(value is True for value in required_true):
        _invalid("Stage 17 evidence contracts did not all pass")
    if lifecycle.get("exit_code") != 0:
        _invalid("Stage 17 lifecycle tests did not pass")
    if benchmark.get("strict_verdict") != "descriptive_only":
        _invalid("Stage 17 benchmark must remain descriptive_only")
    if benchmark.get("wall_clock_performance_improvement") is not False:
        _invalid("Stage 17 evidence must not claim wall-clock improvement")
    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "kv_pressure_preemption": True,
        "whole_request_preemption": True,
        "recompute_resume": True,
        "recompute_does_not_sample": True,
        "per_request_rng_equivalence": True,
        "overflow_sliding_window_equivalence": True,
        "per_tick_budget_respected": True,
        "apc_shared_refs_released": True,
        "resume_uses_private_recompute": True,
        "no_starvation_finite_workload": True,
        "terminal_prefill_logits_released": True,
        "intrinsic_logical_failure_no_preemption": True,
        "intrinsic_physical_failure_no_preemption": True,
        "stress_operations": stress["operations"],
        "stress_passed": True,
        "lifecycle_passed": True,
        "benchmark_strict_verdict": "descriptive_only",
        "wall_clock_performance_improvement": False,
        "implementation": "Python/PyTorch reference implementation",
        "dynamic_kv_reservation": False,
        "cpu_swap": False,
        "partial_block_copy_on_write": False,
        "new_http_api": False,
    }
    _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary, correctness, scheduling),
        encoding="utf-8",
        newline=chr(10),
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
    _ = verify_stage17_evidence(package_root)
    return package_root


def _verify_artifact_manifest(package_root: Path) -> tuple[EvidenceDocument, str]:
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 17")
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
    for key in (
        "kv_pressure_preemption",
        "whole_request_preemption",
        "recompute_resume",
        "recompute_does_not_sample",
        "per_request_rng_equivalence",
        "overflow_sliding_window_equivalence",
        "per_tick_budget_respected",
        "apc_shared_refs_released",
        "resume_uses_private_recompute",
        "no_starvation_finite_workload",
        "terminal_prefill_logits_released",
        "intrinsic_logical_failure_no_preemption",
        "intrinsic_physical_failure_no_preemption",
        "stress_passed",
        "lifecycle_passed",
    ):
        if summary.get(key) is not True:
            _invalid(f"summary contract {key} did not pass")
    for key in (
        "wall_clock_performance_improvement",
        "dynamic_kv_reservation",
        "cpu_swap",
        "partial_block_copy_on_write",
        "new_http_api",
    ):
        if summary.get(key) is not False:
            _invalid(f"Stage 17 scope boundary {key} must be false")
    if summary.get("benchmark_strict_verdict") != "descriptive_only":
        _invalid("Stage 17 benchmark verdict must be descriptive_only")


def verify_stage17_evidence(package_root: Path) -> EvidenceDocument:
    """Verify exact membership, hashes, source identity, contracts, and bounded claims."""
    manifest, source_commit = _verify_artifact_manifest(package_root)
    _verify_summary(_read_json(package_root / "summary.json"), source_commit)
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
    return chr(10).join(
        (
            "# Stage 17 — KV-Pressure Preemption + Recompute Resume",
            "",
            "Stage 17 lets a DECODING request yield its whole resident paged KV reservation",
            "under FIFO KV pressure, then later rebuild cache-only history and resume.",
            "Recompute never samples and does not advance request-local RNG.",
            "",
            f"Roomy/pressure logical equivalence: {correctness['equivalent']}.",
            f"Per-request RNG equivalence: {correctness['per_request_rng_equivalence']}.",
            (
                "Per-tick actual model-work budget respected: "
                f"{scheduling['per_tick_budget_respected']}."
            ),
            f"Observed preemptions: {correctness['preemptions']}.",
            f"Observed recompute tokens: {correctness['recompute_tokens']}.",
            "Intrinsically impossible logical/physical FIFO heads are rejected without preemption.",
            "",
            "APC shared references are released on preemption. Resume intentionally rebuilds",
            "private KV instead of reusing original-position prefix blocks across a sliding",
            "window.",
            "",
            "## Performance claim policy",
            "",
            "This package records structural scheduling evidence only. The benchmark verdict is",
            "descriptive_only; no wall-clock performance improvement is claimed.",
            "",
            "## Scope boundaries",
            "",
            "Dynamic/lazy KV reservation, CPU swap, partial-block COW, GPU/CUDA, fused kernels,",
            "and new HTTP request APIs remain outside Stage 17.",
            "",
            f"Source commit: {summary['source_commit']}.",
        )
    ) + chr(10)
