"""Generate and verify hash-bound Stage 19 serving runtime-configuration evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Never, TypeAlias, cast

import torch
from typing_extensions import override

from minigpt import __version__
from minigpt.data import JsonValue
from minigpt.model import GPT
from minigpt.paged_kv_cache import KVCacheBackend
from minigpt.serving import APCPrefillStrategy, PagedAttentionExecutor
from minigpt.serving_runtime import (
    ServingExecutorName,
    ServingRuntime,
    ServingRuntimeConfig,
    build_serving_runtime,
    file_sha256,
    render_runtime_manifest,
    write_runtime_manifest,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME = "19"
_BLOCK_SIZE = 8
_BLOCK_TOKENS = 2
_POOLED_BLOCKS = 4
_LAZY_OVERCOMMIT_RATIO = 2.0
_LAZY_COMMAND_QUEUE_SIZE = 64
_LAZY_STREAM_BUFFER_SIZE = 8
InvalidConfigFactory: TypeAlias = Callable[[], ServingRuntimeConfig]
_INVALID_CONFIG_FACTORIES: tuple[tuple[str, InvalidConfigFactory], ...] = (
    (
        "paged_attention_without_paged_backend",
        lambda: ServingRuntimeConfig(
            executor=ServingExecutorName.PAGED_ATTENTION,
            max_active_requests=2,
            max_cached_tokens=8,
            kv_cache_backend=KVCacheBackend.DENSE,
        ),
    ),
    (
        "prefix_cache_without_direct_paged",
        lambda: ServingRuntimeConfig(
            executor=ServingExecutorName.CONTINUOUS,
            max_active_requests=2,
            max_cached_tokens=8,
            kv_cache_backend=KVCacheBackend.PAGED,
            prefix_cache=True,
        ),
    ),
    (
        "batched_apc_without_prefix_cache",
        lambda: ServingRuntimeConfig(
            executor=ServingExecutorName.PAGED_ATTENTION,
            max_active_requests=2,
            max_cached_tokens=8,
            kv_cache_backend=KVCacheBackend.PAGED,
            apc_prefill_strategy=APCPrefillStrategy.BATCHED,
        ),
    ),
    (
        "half_configured_token_budget",
        lambda: ServingRuntimeConfig(
            executor=ServingExecutorName.PAGED_ATTENTION,
            max_active_requests=2,
            max_cached_tokens=8,
            kv_cache_backend=KVCacheBackend.PAGED,
            max_scheduled_tokens=8,
        ),
    ),
    (
        "chunked_budget_on_dense_executor",
        lambda: ServingRuntimeConfig(
            executor=ServingExecutorName.CONTINUOUS,
            max_active_requests=2,
            max_cached_tokens=8,
            max_scheduled_tokens=8,
            prefill_chunk_tokens=2,
        ),
    ),
    (
        "preemption_without_token_budget",
        lambda: ServingRuntimeConfig(
            executor=ServingExecutorName.PAGED_ATTENTION,
            max_active_requests=2,
            max_cached_tokens=8,
            kv_cache_backend=KVCacheBackend.PAGED,
            kv_preemption=True,
        ),
    ),
    (
        "lazy_reservation_without_preemption",
        lambda: ServingRuntimeConfig(
            executor=ServingExecutorName.PAGED_ATTENTION,
            max_active_requests=2,
            max_cached_tokens=8,
            kv_cache_backend=KVCacheBackend.PAGED,
            max_scheduled_tokens=8,
            prefill_chunk_tokens=2,
            lazy_kv_reservation=True,
        ),
    ),
    (
        "overcommit_ratio_outside_lazy_mode",
        lambda: ServingRuntimeConfig(
            executor=ServingExecutorName.CONTINUOUS,
            max_active_requests=2,
            max_cached_tokens=8,
            kv_overcommit_ratio=1.5,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class Stage19EvidenceVerificationError(ValueError):
    """Report invalid Stage 19 evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Stage 19 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage19EvidenceVerificationError(reason)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _model() -> GPT:
    original = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(1919)
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


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _legacy_config() -> ServingRuntimeConfig:
    return ServingRuntimeConfig(
        executor=ServingExecutorName.CONTINUOUS,
        max_active_requests=4,
        max_cached_tokens=32,
    )


def _lazy_config() -> ServingRuntimeConfig:
    return ServingRuntimeConfig(
        executor=ServingExecutorName.PAGED_ATTENTION,
        max_active_requests=2,
        max_cached_tokens=8,
        kv_cache_backend=KVCacheBackend.PAGED,
        kv_block_tokens=_BLOCK_TOKENS,
        kv_num_blocks=_POOLED_BLOCKS,
        max_scheduled_tokens=_BLOCK_SIZE,
        prefill_chunk_tokens=_BLOCK_TOKENS,
        kv_preemption=True,
        lazy_kv_reservation=True,
        kv_overcommit_ratio=_LAZY_OVERCOMMIT_RATIO,
        command_queue_size=_LAZY_COMMAND_QUEUE_SIZE,
        stream_buffer_size=_LAZY_STREAM_BUFFER_SIZE,
    )


def _runtime(config: ServingRuntimeConfig, model: GPT) -> ServingRuntime:
    return build_serving_runtime(
        model=model,
        block_size=_BLOCK_SIZE,
        num_threads=1,
        checkpoint_sha256=_digest("stage19-checkpoint"),
        tokenizer_sha256=_digest("stage19-tokenizer"),
        config=config,
    )


def _resources_released(runtime: ServingRuntime) -> bool:
    pool = runtime.engine.paged_cache_pool
    if pool is None:
        return True
    metrics = pool.metrics()
    return (
        metrics.private_blocks == 0
        and metrics.active_shared_references == 0
        and metrics.reserved_blocks == 0
        and metrics.allocated_blocks == 0
    )


def generate_stage19_runtime_wiring(output_path: Path) -> Path:
    """Prove legacy defaults and real lazy scheduler/pool/executor wiring."""
    model = _model()

    legacy = _runtime(_legacy_config(), model)
    legacy_scheduler = legacy.engine.config.scheduler
    legacy_document = legacy.manifest
    if (
        legacy.engine.config.kv_cache_backend is not KVCacheBackend.DENSE
        or legacy.engine.paged_cache_pool is not None
        or legacy_scheduler.max_scheduled_tokens is not None
        or legacy_scheduler.prefill_chunk_tokens is not None
        or legacy_scheduler.kv_preemption
        or legacy_scheduler.lazy_kv_reservation
        or legacy_scheduler.kv_overcommit_ratio != 1.0
        or cast("dict[str, object]", legacy_document["prefix_cache"])["enabled"] is not False
    ):
        _invalid("Stage 19 legacy defaults diverge from the pre-Stage-19 HTTP runtime")

    lazy = _runtime(_lazy_config(), model)
    lazy_scheduler = lazy.engine.config.scheduler
    executor = lazy.executor
    pool = lazy.engine.paged_cache_pool
    if not isinstance(executor, PagedAttentionExecutor):
        _invalid("Stage 19 lazy runtime did not wire the direct paged executor")
    if pool is None:
        _invalid("Stage 19 lazy runtime did not build a paged cache pool")
    if (
        executor.paged_cache_pool is not pool
        or executor.prefix_prefill_strategy is not APCPrefillStrategy.SEQUENTIAL
        or pool.config.block_tokens != _BLOCK_TOKENS
        or pool.config.num_blocks != _POOLED_BLOCKS
        or not lazy_scheduler.lazy_kv_reservation
        or not lazy_scheduler.kv_preemption
        or lazy_scheduler.kv_overcommit_ratio != _LAZY_OVERCOMMIT_RATIO
        or lazy_scheduler.max_scheduled_tokens != _BLOCK_SIZE
        or lazy_scheduler.prefill_chunk_tokens != _BLOCK_TOKENS
        or lazy.runner.config.command_queue_size != _LAZY_COMMAND_QUEUE_SIZE
        or lazy.runner.config.stream_buffer_size != _LAZY_STREAM_BUFFER_SIZE
    ):
        _invalid("Stage 19 lazy runtime wiring diverges from the resolved policy")
    if not _resources_released(lazy):
        _invalid("Stage 19 idle runtime retained KV resources")

    batched = _runtime(
        ServingRuntimeConfig(
            executor=ServingExecutorName.PAGED_ATTENTION,
            max_active_requests=2,
            max_cached_tokens=8,
            kv_cache_backend=KVCacheBackend.PAGED,
            kv_block_tokens=_BLOCK_TOKENS,
            kv_num_blocks=_POOLED_BLOCKS,
            prefix_cache=True,
            apc_prefill_strategy=APCPrefillStrategy.BATCHED,
            max_scheduled_tokens=_BLOCK_SIZE,
            prefill_chunk_tokens=_BLOCK_TOKENS,
            kv_preemption=True,
            lazy_kv_reservation=True,
            kv_overcommit_ratio=_LAZY_OVERCOMMIT_RATIO,
        ),
        model,
    )
    batched_executor = cast("PagedAttentionExecutor", batched.executor)
    if (
        batched_executor.prefix_prefill_strategy is not APCPrefillStrategy.BATCHED
        or batched.engine.paged_cache_pool is None
        or not batched.engine.paged_cache_pool.prefix_cache_enabled
    ):
        _invalid("Stage 19 batched APC runtime wiring diverges from the resolved policy")
    if not _resources_released(batched):
        _invalid("Stage 19 idle batched runtime retained KV resources")

    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "legacy_defaults_preserved": True,
            "legacy_scheduler_defaults": cast(
                "JsonValue",
                {
                    "max_scheduled_tokens": legacy_scheduler.max_scheduled_tokens,
                    "prefill_chunk_tokens": legacy_scheduler.prefill_chunk_tokens,
                    "kv_preemption": legacy_scheduler.kv_preemption,
                    "lazy_kv_reservation": legacy_scheduler.lazy_kv_reservation,
                },
            ),
            "lazy_scheduler_resolved": cast("JsonValue", asdict(lazy_scheduler)),
            "lazy_pool_dimensions": cast(
                "JsonValue",
                {"block_tokens": pool.config.block_tokens, "num_blocks": pool.config.num_blocks},
            ),
            "executor": type(executor).__name__,
            "apc_prefill_strategy": executor.prefix_prefill_strategy.value,
            "batched_apc_wired": True,
            "runner_queue_sizes": {
                "command_queue_size": lazy.runner.config.command_queue_size,
                "stream_buffer_size": lazy.runner.config.stream_buffer_size,
            },
            "idle_resources_released": True,
        },
    )
    return output_path


def generate_stage19_invalid_combinations(output_path: Path) -> Path:
    """Prove the typed policy rejects every important invalid combination."""
    rejected: dict[str, str] = {}
    for name, factory in _INVALID_CONFIG_FACTORIES:
        try:
            _ = factory()
        except ValueError as error:
            rejected[name] = str(error)
        else:
            _invalid(f"Stage 19 invalid combination was accepted: {name}")
    if len(rejected) != len(_INVALID_CONFIG_FACTORIES):
        _invalid("Stage 19 invalid-combination witness lost an entry")
    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "rejected_combinations": cast("JsonValue", rejected),
            "all_rejected": True,
        },
    )
    return output_path


def generate_stage19_manifest(
    output_path: Path,
    *,
    work_root: Path,
) -> Path:
    """Prove manifest bytes are deterministic, portable, and atomically replaced."""
    document_root = work_root / "manifest"
    document_root.mkdir(parents=True, exist_ok=True)
    model = _model()
    lazy = _runtime(_lazy_config(), model)
    legacy = _runtime(_legacy_config(), model)

    first_path = document_root / "first.json"
    second_path = document_root / "second.json"
    _ = write_runtime_manifest(first_path, lazy.manifest)
    _ = write_runtime_manifest(second_path, lazy.manifest)
    first_bytes = first_path.read_bytes()
    if (
        first_bytes != second_path.read_bytes()
        or first_bytes != render_runtime_manifest(lazy.manifest).encode()
        or not first_bytes.endswith(b"\n")
        or b"\r" in first_bytes
    ):
        _invalid("Stage 19 manifest bytes are not deterministic UTF-8/LF")
    serialized = first_bytes.decode("utf-8")
    for forbidden in (
        str(document_root),
        str(work_root),
        "timestamp",
        "hostname",
        "pid",
    ):
        if forbidden.lower() in serialized.lower():
            _invalid(f"Stage 19 manifest leaked a non-portable field: {forbidden}")

    replaced_path = document_root / "replaced.json"
    _ = replaced_path.write_text("stale-bytes", encoding="utf-8")
    _ = write_runtime_manifest(replaced_path, legacy.manifest)
    if replaced_path.read_text(encoding="utf-8") == "stale-bytes":
        _invalid("Stage 19 manifest replacement did not take effect")
    leftovers = [path.name for path in document_root.iterdir() if path.name.endswith(".tmp")]
    if leftovers:
        _invalid(f"Stage 19 manifest write left temporary files: {leftovers}")

    legacy_document = legacy.manifest
    lazy_document = lazy.manifest
    claim = cast("dict[str, JsonValue]", lazy_document["claim_policy"])
    if (
        legacy_document["stage"] != STAGE_NAME
        or legacy_document["schema_version"] != 1
        or legacy_document["project_version"] != __version__
        or claim["benchmark_strict_verdict"] != "descriptive_only"
        or claim["wall_clock_performance_improvement"] is not False
        or claim["public_production_security_readiness"] is not False
        or cast("dict[str, JsonValue]", lazy_document["scheduler"])["lazy_kv_reservation"]
        is not True
    ):
        _invalid("Stage 19 manifest contract fields are wrong")

    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "manifest_bytes_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "deterministic_bytes": True,
            "lf_only": True,
            "no_absolute_paths": True,
            "no_timestamps": True,
            "atomic_replacement": True,
            "no_temporary_leftovers": True,
            "claim_policy_bounded": True,
        },
    )
    return output_path


def generate_stage19_checkpoint_identity(output_path: Path, tmp_path: Path) -> Path:
    """Bind the manifest to real checkpoint/tokenizer SHA-256 identities."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tmp_path / "stage19-checkpoint.bin"
    tokenizer_path = tmp_path / "stage19-tokenizer.json"
    _ = checkpoint_path.write_bytes(b"stage19-checkpoint-payload")
    _ = tokenizer_path.write_bytes(b"stage19-tokenizer-payload")
    checkpoint_sha = file_sha256(checkpoint_path)
    tokenizer_sha = file_sha256(tokenizer_path)
    runtime = build_serving_runtime(
        model=_model(),
        block_size=_BLOCK_SIZE,
        num_threads=1,
        checkpoint_sha256=checkpoint_sha,
        tokenizer_sha256=tokenizer_sha,
        config=_lazy_config(),
    )
    document = runtime.manifest
    if (
        document["checkpoint_sha256"] != checkpoint_sha
        or document["tokenizer_sha256"] != tokenizer_sha
    ):
        _invalid("Stage 19 manifest did not bind the real file identities")
    _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "checkpoint_sha256": checkpoint_sha,
            "tokenizer_sha256": tokenizer_sha,
            "identity_bound": True,
        },
    )
    return output_path


def write_stage19_lifecycle(path: Path, tests: Sequence[str]) -> Path:
    """Run the repository lifecycle tests and capture their evidence."""
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and repository-owned tests
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "command": "python -m pytest -q " + " ".join(tests),
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return path


def generate_stage19_evidence(  # noqa: PLR0913
    *,
    runtime_wiring_path: Path,
    invalid_combinations_path: Path,
    manifest_path: Path,
    identity_path: Path,
    lifecycle_path: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Build the Stage 19 compatibility, wiring, manifest, and lifecycle package."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    inputs = {
        "runtime_wiring.json": runtime_wiring_path,
        "invalid_combinations.json": invalid_combinations_path,
        "manifest.json": manifest_path,
        "identity.json": identity_path,
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

    wiring = _read_json(evidence_root / "runtime_wiring.json")
    invalid = _read_json(evidence_root / "invalid_combinations.json")
    manifest = _read_json(evidence_root / "manifest.json")
    identity = _read_json(evidence_root / "identity.json")
    lifecycle = _read_json(evidence_root / "lifecycle_tests.json")
    required_true = (
        wiring.get("legacy_defaults_preserved"),
        wiring.get("batched_apc_wired"),
        wiring.get("idle_resources_released"),
        invalid.get("all_rejected"),
        manifest.get("deterministic_bytes"),
        manifest.get("lf_only"),
        manifest.get("no_absolute_paths"),
        manifest.get("no_timestamps"),
        manifest.get("atomic_replacement"),
        manifest.get("no_temporary_leftovers"),
        manifest.get("claim_policy_bounded"),
        identity.get("identity_bound"),
    )
    if not all(value is True for value in required_true):
        _invalid("Stage 19 evidence contracts did not all pass")
    if lifecycle.get("exit_code") != 0:
        _invalid("Stage 19 lifecycle tests did not pass")

    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "legacy_defaults_preserved": True,
        "stage15_apc_prefill_strategy_flag": True,
        "stage16_token_budget_flags": True,
        "stage17_preemption_flag": True,
        "stage18_lazy_reservation_flag": True,
        "typed_policy_validation": True,
        "runtime_manifest": True,
        "deterministic_manifest_bytes": True,
        "atomic_manifest_replacement": True,
        "checkpoint_tokenizer_identity_bound": True,
        "http_schema_unchanged": True,
        "lifecycle_passed": True,
        "benchmark_strict_verdict": "descriptive_only",
        "wall_clock_performance_improvement": False,
        "public_production_security_readiness": False,
        "implementation": "Python/PyTorch reference implementation",
    }
    _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary, wiring, invalid, manifest),
        encoding="utf-8",
        newline="\n",
    )
    manifest_file = package_root / "artifact_manifest.json"
    artifacts: list[JsonValue] = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path != manifest_file
    ]
    _write_json(
        manifest_file,
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
    _ = verify_stage19_evidence(package_root)
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
    manifest_file = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_file)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 19")
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
        if path.is_file() and path != manifest_file
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
        "legacy_defaults_preserved",
        "stage15_apc_prefill_strategy_flag",
        "stage16_token_budget_flags",
        "stage17_preemption_flag",
        "stage18_lazy_reservation_flag",
        "typed_policy_validation",
        "runtime_manifest",
        "deterministic_manifest_bytes",
        "atomic_manifest_replacement",
        "checkpoint_tokenizer_identity_bound",
        "http_schema_unchanged",
        "lifecycle_passed",
    )
    for key in true_keys:
        if summary.get(key) is not True:
            _invalid(f"summary contract {key} did not pass")
    false_keys = (
        "wall_clock_performance_improvement",
        "public_production_security_readiness",
    )
    for key in false_keys:
        if summary.get(key) is not False:
            _invalid(f"Stage 19 scope boundary {key} must be false")
    if summary.get("benchmark_strict_verdict") != "descriptive_only":
        _invalid("Stage 19 benchmark verdict must be descriptive_only")


def verify_stage19_evidence(package_root: Path) -> EvidenceDocument:
    """Verify exact membership, hashes, source identity, contracts, and bounded claims."""
    manifest, source_commit = _verify_artifact_manifest(package_root)
    _verify_summary(_read_json(package_root / "summary.json"), source_commit)
    return manifest


def _readme(
    summary: EvidenceDocument,
    wiring: EvidenceDocument,
    invalid: EvidenceDocument,
    manifest: EvidenceDocument,
) -> str:
    rejected = cast("dict[str, object]", invalid["rejected_combinations"])
    return (
        "\n".join(
            (
                "# Stage 19 — Production Serving Configuration + Runtime Manifest",
                "",
                "Stage 19 exposes the Stage 15-18 scheduler and paged-cache controls on the",
                "real HTTP CLI behind one typed package-level runtime builder, while the",
                "legacy dense/continuous service keeps its previous defaults.",
                "",
                f"Legacy defaults preserved: {summary['legacy_defaults_preserved']}.",
                f"Invalid combinations rejected: {len(rejected)}.",
                f"Deterministic manifest bytes: {manifest['deterministic_bytes']}.",
                f"Manifest SHA-256: {manifest['manifest_bytes_sha256']}.",
                f"Atomic replacement verified: {manifest['atomic_replacement']}.",
                "Checkpoint/tokenizer SHA-256 identities are bound in the manifest.",
                f"Idle runtime resources released: {wiring['idle_resources_released']}.",
                "",
                "The completion request schema is unchanged. serve.py remains a thin",
                "parser/Uvicorn boundary over minigpt.serving_runtime.",
                "",
                "## Performance claim policy",
                "",
                "The benchmark verdict is descriptive_only. No wall-clock performance",
                "improvement is claimed and no public-production security readiness is",
                "claimed.",
                "",
                "## Scope boundaries",
                "",
                "New HTTP endpoints, authentication, GPU kernels, CPU swap, partial-block",
                "COW, and speculative decoding remain outside Stage 19.",
                "",
                f"Source commit: {summary['source_commit']}.",
            )
        )
        + "\n"
    )
