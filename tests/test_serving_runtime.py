from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import serve
import torch
import yaml

import minigpt.serving_runtime as runtime_module
from minigpt import __version__
from minigpt.model import GPT
from minigpt.paged_kv_cache import KVCacheBackend
from minigpt.serving import APCPrefillStrategy, PagedAttentionExecutor
from minigpt.serving_runtime import (
    ServingExecutorName,
    ServingRuntime,
    ServingRuntimeConfig,
    build_serving_runtime,
    render_runtime_manifest,
    write_runtime_manifest,
)
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    import argparse


def _model() -> GPT:
    original = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(1919)
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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _argument(namespace: argparse.Namespace, name: str) -> object:
    values = cast("dict[str, object]", vars(namespace))
    return values[name]


def _runtime(config: ServingRuntimeConfig) -> ServingRuntime:
    return build_serving_runtime(
        model=_model(),
        block_size=8,
        num_threads=1,
        checkpoint_sha256=_digest("checkpoint"),
        tokenizer_sha256=_digest("tokenizer"),
        config=config,
    )


def test_legacy_runtime_defaults_preserve_continuous_dense_contract() -> None:
    config = ServingRuntimeConfig(
        executor=ServingExecutorName.CONTINUOUS,
        max_active_requests=4,
        max_cached_tokens=32,
    )

    runtime = _runtime(config)

    assert runtime.config is config
    assert runtime.engine.config.kv_cache_backend is KVCacheBackend.DENSE
    assert runtime.engine.config.scheduler.max_active_requests == 4
    assert runtime.engine.config.scheduler.max_cached_tokens == 32
    assert runtime.engine.config.scheduler.max_scheduled_tokens is None
    assert runtime.engine.paged_cache_pool is None
    assert runtime.runner.config.command_queue_size == 256
    assert runtime.manifest["executor"] == "continuous"
    assert runtime.manifest["paged_kv_cache"] is None
    prefix = cast("dict[str, object]", runtime.manifest["prefix_cache"])
    assert prefix == {"enabled": False, "apc_prefill_strategy": "sequential"}


def _typed_runtime(config: ServingRuntimeConfig) -> ServingRuntime:
    return build_serving_runtime(
        model=_model(),
        block_size=8,
        num_threads=1,
        checkpoint_sha256=_digest("checkpoint"),
        tokenizer_sha256=_digest("tokenizer"),
        config=config,
        tokenizer_type="bpe",
        model_family="story_forge",
    )


def test_runtime_manifest_records_tokenizer_type_and_family() -> None:
    # Given: a nominal runtime with explicit tokenizer identity labels.
    runtime = _typed_runtime(
        ServingRuntimeConfig(
            executor=ServingExecutorName.CONTINUOUS,
            max_active_requests=2,
            max_cached_tokens=16,
        )
    )

    # When: the tokenizer descriptor is read from the manifest.
    tokenizer = cast("dict[str, object]", runtime.manifest["tokenizer"])

    # Then: the identity is recorded without exposing any path.
    assert tokenizer["sha256"] == _digest("tokenizer")
    assert tokenizer["type"] == "bpe"
    assert tokenizer["model_family"] == "story_forge"


def test_runtime_manifest_omits_tokenizer_labels_when_unknown() -> None:
    # Given: a runtime built without tokenizer identity labels.
    runtime = _runtime(
        ServingRuntimeConfig(
            executor=ServingExecutorName.CONTINUOUS,
            max_active_requests=2,
            max_cached_tokens=16,
        )
    )

    # When: the tokenizer descriptor is read.
    tokenizer = cast("dict[str, object]", runtime.manifest["tokenizer"])

    # Then: only the SHA-256 identity is present, with no invented labels.
    assert tokenizer == {"sha256": _digest("tokenizer")}


InvalidRuntimeFactory = Callable[[], ServingRuntimeConfig]


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: ServingRuntimeConfig(
                executor=ServingExecutorName.PAGED_ATTENTION,
                max_active_requests=2,
                max_cached_tokens=8,
                kv_cache_backend=KVCacheBackend.DENSE,
            ),
            "paged KV-cache backend",
        ),
        (
            lambda: ServingRuntimeConfig(
                executor=ServingExecutorName.CONTINUOUS,
                max_active_requests=2,
                max_cached_tokens=8,
                kv_cache_backend=KVCacheBackend.PAGED,
                prefix_cache=True,
            ),
            "prefix_cache",
        ),
        (
            lambda: ServingRuntimeConfig(
                executor=ServingExecutorName.PAGED_ATTENTION,
                max_active_requests=2,
                max_cached_tokens=8,
                kv_cache_backend=KVCacheBackend.PAGED,
                apc_prefill_strategy=APCPrefillStrategy.BATCHED,
            ),
            "batched APC",
        ),
        (
            lambda: ServingRuntimeConfig(
                executor=ServingExecutorName.PAGED_ATTENTION,
                max_active_requests=2,
                max_cached_tokens=8,
                kv_cache_backend=KVCacheBackend.PAGED,
                max_scheduled_tokens=8,
            ),
            "configured together",
        ),
        (
            lambda: ServingRuntimeConfig(
                executor=ServingExecutorName.CONTINUOUS,
                max_active_requests=2,
                max_cached_tokens=8,
                max_scheduled_tokens=8,
                prefill_chunk_tokens=2,
            ),
            "direct paged_attention",
        ),
        (
            lambda: ServingRuntimeConfig(
                executor=ServingExecutorName.PAGED_ATTENTION,
                max_active_requests=2,
                max_cached_tokens=8,
                kv_cache_backend=KVCacheBackend.PAGED,
                kv_preemption=True,
            ),
            "token-budget",
        ),
        (
            lambda: ServingRuntimeConfig(
                executor=ServingExecutorName.PAGED_ATTENTION,
                max_active_requests=2,
                max_cached_tokens=8,
                kv_cache_backend=KVCacheBackend.PAGED,
                max_scheduled_tokens=8,
                prefill_chunk_tokens=2,
                lazy_kv_reservation=True,
            ),
            "requires kv_preemption",
        ),
        (
            lambda: ServingRuntimeConfig(
                executor=ServingExecutorName.CONTINUOUS,
                max_active_requests=2,
                max_cached_tokens=8,
                kv_overcommit_ratio=1.5,
            ),
            "must be 1.0",
        ),
        (
            lambda: ServingRuntimeConfig(
                executor=ServingExecutorName.CONTINUOUS,
                max_active_requests=2,
                max_cached_tokens=8,
                kv_overcommit_ratio=float("nan"),
            ),
            "finite",
        ),
    ],
)
def test_runtime_config_rejects_invalid_combinations(
    factory: InvalidRuntimeFactory,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = factory()


def test_real_lazy_paged_runtime_wires_scheduler_pool_and_batched_apc() -> None:
    config = ServingRuntimeConfig(
        executor=ServingExecutorName.PAGED_ATTENTION,
        max_active_requests=2,
        max_cached_tokens=8,
        kv_cache_backend=KVCacheBackend.PAGED,
        kv_block_tokens=2,
        kv_num_blocks=4,
        prefix_cache=True,
        apc_prefill_strategy=APCPrefillStrategy.BATCHED,
        max_scheduled_tokens=8,
        prefill_chunk_tokens=2,
        kv_preemption=True,
        lazy_kv_reservation=True,
        kv_overcommit_ratio=2.0,
        command_queue_size=7,
        stream_buffer_size=3,
    )

    runtime = _runtime(config)

    scheduler = runtime.engine.config.scheduler
    assert scheduler.lazy_kv_reservation is True
    assert scheduler.kv_preemption is True
    assert scheduler.kv_overcommit_ratio == 2.0
    assert scheduler.max_scheduled_tokens == 8
    assert scheduler.prefill_chunk_tokens == 2
    pool = runtime.engine.paged_cache_pool
    assert pool is not None
    assert pool.config.block_tokens == 2
    assert pool.config.num_blocks == 4
    executor = runtime.executor
    assert isinstance(executor, PagedAttentionExecutor)
    assert executor.prefix_prefill_strategy is APCPrefillStrategy.BATCHED
    assert runtime.runner.config.command_queue_size == 7
    assert runtime.runner.config.stream_buffer_size == 3
    scheduler_document = cast("dict[str, object]", runtime.manifest["scheduler"])
    assert scheduler_document["lazy_kv_reservation"] is True
    claim = cast("dict[str, object]", runtime.manifest["claim_policy"])
    assert claim["benchmark_strict_verdict"] == "descriptive_only"
    assert claim["wall_clock_performance_improvement"] is False


@pytest.mark.parametrize(
    ("chunk", "budget", "match"),
    [(3, 8, "align"), (2, 4, "too small")],
)
def test_engine_validation_rejects_chunk_alignment_or_budget(
    chunk: int,
    budget: int,
    match: str,
) -> None:
    config = ServingRuntimeConfig(
        executor=ServingExecutorName.PAGED_ATTENTION,
        max_active_requests=2,
        max_cached_tokens=8,
        kv_cache_backend=KVCacheBackend.PAGED,
        kv_block_tokens=2,
        kv_num_blocks=4,
        max_scheduled_tokens=budget,
        prefill_chunk_tokens=chunk,
    )

    with pytest.raises(ValueError, match=match):
        _ = _runtime(config)


def test_runtime_manifest_is_deterministic_portable_and_lf(tmp_path: Path) -> None:
    runtime = _runtime(
        ServingRuntimeConfig(
            executor=ServingExecutorName.CONTINUOUS,
            max_active_requests=2,
            max_cached_tokens=16,
        )
    )

    first = render_runtime_manifest(runtime.manifest)
    second = render_runtime_manifest(runtime.manifest)
    output = tmp_path / "runtime.json"
    _ = write_runtime_manifest(output, runtime.manifest)

    assert first == second == output.read_text(encoding="utf-8")
    assert output.read_bytes().endswith(b"\n")
    assert b"\r\n" not in output.read_bytes()
    assert str(tmp_path) not in first
    assert "timestamp" not in first.lower()
    assert "hostname" not in first.lower()
    document = cast("dict[str, object]", json.loads(first))
    assert document["schema_version"] == 1
    assert document["stage"] == "19"
    assert document["project_version"] == __version__
    assert document["checkpoint_sha256"] == _digest("checkpoint")
    assert document["tokenizer_sha256"] == _digest("tokenizer")


def test_runtime_manifest_failed_replace_leaves_no_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "runtime.json"
    _ = target.write_text("previous\n", encoding="utf-8", newline="\n")

    def fail_replace(_source: Path, _target: Path) -> None:
        reason = "injected atomic replace failure"
        raise OSError(reason)

    monkeypatch.setattr(runtime_module, "_atomic_replace", fail_replace)

    with pytest.raises(OSError, match="injected"):
        _ = write_runtime_manifest(target, {"schema_version": 1})

    assert target.read_text(encoding="utf-8") == "previous\n"
    assert not tuple(tmp_path.glob("*.tmp"))
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_serve_parser_exposes_stage19_flags_without_changing_defaults() -> None:
    parser = serve.build_parser()

    defaults = parser.parse_args(["--checkpoint", "model.pt", "--tokenizer", "tokenizer.json"])
    configured = parser.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--tokenizer",
            "tokenizer.json",
            "--executor",
            "paged_attention",
            "--kv-cache-backend",
            "paged",
            "--kv-block-tokens",
            "2",
            "--kv-num-blocks",
            "4",
            "--prefix-cache",
            "--apc-prefill-strategy",
            "batched",
            "--max-scheduled-tokens",
            "8",
            "--prefill-chunk-tokens",
            "2",
            "--kv-preemption",
            "--lazy-kv-reservation",
            "--kv-overcommit-ratio",
            "2",
            "--runtime-manifest",
            "runtime.json",
        ]
    )

    assert _argument(defaults, "executor") == "continuous"
    assert _argument(defaults, "kv_cache_backend") == "dense"
    assert _argument(defaults, "max_scheduled_tokens") is None
    assert _argument(defaults, "prefill_chunk_tokens") is None
    assert _argument(defaults, "kv_preemption") is False
    assert _argument(defaults, "lazy_kv_reservation") is False
    assert _argument(defaults, "kv_overcommit_ratio") == 1.0
    assert _argument(defaults, "apc_prefill_strategy") == "sequential"
    assert _argument(defaults, "runtime_manifest") is None
    assert _argument(configured, "apc_prefill_strategy") == "batched"
    assert _argument(configured, "max_scheduled_tokens") == 8
    assert _argument(configured, "prefill_chunk_tokens") == 2
    assert _argument(configured, "kv_preemption") is True
    assert _argument(configured, "lazy_kv_reservation") is True
    assert _argument(configured, "kv_overcommit_ratio") == 2.0
    assert _argument(configured, "runtime_manifest") == Path("runtime.json")


def test_canonical_http_lazy_kv_example_resolves() -> None:
    raw = cast(
        "dict[str, object]",
        yaml.safe_load(Path("configs/serving_http_lazy_kv.yaml").read_text(encoding="utf-8")),
    )

    config = ServingRuntimeConfig(
        executor=ServingExecutorName(cast("str", raw["executor"])),
        max_active_requests=cast("int", raw["max_active_requests"]),
        max_cached_tokens=cast("int", raw["max_cached_tokens"]),
        kv_cache_backend=KVCacheBackend(cast("str", raw["kv_cache_backend"])),
        kv_block_tokens=cast("int", raw["kv_block_tokens"]),
        kv_num_blocks=cast("int", raw["kv_num_blocks"]),
        max_scheduled_tokens=cast("int", raw["max_scheduled_tokens"]),
        prefill_chunk_tokens=cast("int", raw["prefill_chunk_tokens"]),
        kv_preemption=cast("bool", raw["kv_preemption"]),
        lazy_kv_reservation=cast("bool", raw["lazy_kv_reservation"]),
        kv_overcommit_ratio=cast("float", raw["kv_overcommit_ratio"]),
    )
    paged_cache = config.paged_cache()

    assert config.scheduler().lazy_kv_reservation is True
    assert paged_cache is not None
    assert paged_cache.num_blocks == 64
    assert paged_cache.block_tokens == 16
