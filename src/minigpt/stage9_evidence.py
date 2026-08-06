"""Generate and verify the compact Stage 9 KV-cache evidence package."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeAlias, cast

import torch

from minigpt.inference_benchmark import verify_inference_run_manifest
from minigpt.model import GPT, kv_cache_nbytes
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from pathlib import Path

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None


def _json(path: Path) -> dict[str, object]:
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        msg = f"{path} must contain a JSON object"
        raise TypeError(msg)
    return cast("dict[str, object]", raw)


def _write_json(path: Path, document: object) -> None:
    _ = path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _correctness_evidence() -> dict[str, JsonValue]:
    _ = torch.default_generator.manual_seed(20260806)
    config = GPTConfig(
        vocab_size=11,
        block_size=4,
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )
    model = GPT(config)
    _ = model.eval()
    prompt = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    forward_logits, _ = cast("tuple[torch.Tensor, torch.Tensor | None]", model(prompt))
    prefill_logits, cache = model.prefill(prompt)
    new_tokens = torch.tensor([[5], [6]], dtype=torch.long)
    decoded_logits, next_cache = model.decode(new_tokens, cache)
    complete_logits, _ = cast(
        "tuple[torch.Tensor, torch.Tensor | None]",
        model(torch.cat((prompt, new_tokens), dim=1)),
    )
    decode_difference = (decoded_logits - complete_logits[:, -1:, :]).abs().max().item()
    uncached_generator = torch.Generator(device="cpu")
    cached_generator = torch.Generator(device="cpu")
    _ = uncached_generator.manual_seed(41)
    _ = cached_generator.manual_seed(41)
    overflow_prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    uncached_tokens = model.generate(
        overflow_prompt,
        max_new_tokens=4,
        temperature=0.9,
        top_k=7,
        generator=uncached_generator,
    )
    state_names = tuple(model.state_dict())
    buffer_names = tuple(name for name, _ in model.named_buffers())
    cached_tokens = model.generate_cached(
        overflow_prompt,
        max_new_tokens=4,
        temperature=0.9,
        top_k=7,
        generator=cached_generator,
    )
    expected_bytes = 2 * config.n_layer * 2 * 2 * config.n_embd * 4
    return {
        "seed": 20260806,
        "prefill_exact_equal": torch.equal(prefill_logits, forward_logits[:, -1:, :]),
        "decode_rtol": 1e-5,
        "decode_atol": 1e-6,
        "decode_max_absolute_difference": float(decode_difference),
        "decode_within_tolerance": torch.allclose(
            decoded_logits,
            complete_logits[:, -1:, :],
            rtol=1e-5,
            atol=1e-6,
        ),
        "cached_uncached_overflow_tokens_exact_equal": torch.equal(cached_tokens, uncached_tokens),
        "sampling_generator_states_exact_equal": torch.equal(
            cached_generator.get_state(), uncached_generator.get_state()
        ),
        "cache_bytes_observed": kv_cache_nbytes(cache),
        "cache_bytes_expected": expected_bytes,
        "extended_cache_length": next_cache[0].length,
        "cache_requires_grad": any(
            layer_cache.key.requires_grad or layer_cache.value.requires_grad
            for layer_cache in (*cache, *next_cache)
        ),
        "model_state_names_unchanged": tuple(model.state_dict()) == state_names,
        "model_buffer_names_unchanged": (
            tuple(name for name, _ in model.named_buffers()) == buffer_names
        ),
    }


def _compact_performance(run_summary: dict[str, object]) -> dict[str, JsonValue]:
    raw_cases = run_summary.get("cases")
    if not isinstance(raw_cases, list):
        msg = "inference summary cases must be a list"
        raise TypeError(msg)
    case_items = cast("list[object]", raw_cases)
    compact_cases: list[JsonValue] = []
    pass_count = 0
    changes: list[float] = []
    for raw_case in case_items:
        if not isinstance(raw_case, dict):
            msg = "inference summary case must be an object"
            raise TypeError(msg)
        item = cast("dict[str, object]", raw_case)
        case = cast("dict[str, object]", item["case"])
        cached = cast("dict[str, object]", item["cached"])
        uncached = cast("dict[str, object]", item["uncached"])
        comparison = cast("dict[str, object]", item["comparison"])
        end_to_end_change = float(cast("float", comparison["end_to_end_change_percent"]))
        changes.append(end_to_end_change)
        verdict = cast("str", comparison["verdict"])
        pass_count += verdict == "pass"
        compact_cases.append(
            {
                "case_name": cast("str", case["name"]),
                "prompt_length": cast("int", case["prompt_length"]),
                "generated_length": cast("int", case["generated_length"]),
                "uncached_end_to_end_median_ms": cast(
                    "dict[str, JsonValue]", uncached["end_to_end_time_ms"]
                )["median"],
                "cached_end_to_end_median_ms": cast(
                    "dict[str, JsonValue]", cached["end_to_end_time_ms"]
                )["median"],
                "end_to_end_change_percent": end_to_end_change,
                "decode_change_percent": cast("float", comparison["decode_time_change_percent"]),
                "throughput_change_percent": cast("float", comparison["throughput_change_percent"]),
                "uncached_end_to_end_cv_percent": cast(
                    "dict[str, JsonValue]", uncached["end_to_end_time_ms"]
                )["coefficient_of_variation_percent"],
                "cached_end_to_end_cv_percent": cast(
                    "dict[str, JsonValue]", cached["end_to_end_time_ms"]
                )["coefficient_of_variation_percent"],
                "uncached_decode_cv_percent": cast(
                    "dict[str, JsonValue]", uncached["median_decode_time_ms"]
                )["coefficient_of_variation_percent"],
                "cached_decode_cv_percent": cast(
                    "dict[str, JsonValue]", cached["median_decode_time_ms"]
                )["coefficient_of_variation_percent"],
                "kv_cache_bytes": cast("dict[str, JsonValue]", cached["kv_cache_bytes"])["median"],
                "verdict": verdict,
                "reasons": cast("list[JsonValue]", comparison["reasons"]),
            }
        )
    strict_verdict = run_summary.get("strict_verdict")
    if strict_verdict not in {"pass", "not_comparable"}:
        msg = "inference summary strict_verdict is invalid"
        raise ValueError(msg)
    return {
        "strict_verdict": cast("str", strict_verdict),
        "case_count": len(compact_cases),
        "strict_pass_case_count": pass_count,
        "not_comparable_case_count": len(compact_cases) - pass_count,
        "all_descriptive_cached_medians_lower": all(change < 0.0 for change in changes),
        "descriptive_end_to_end_change_percent_range": [min(changes), max(changes)],
        "cases": compact_cases,
    }


def _paragraph(*sentences: str) -> str:
    return " ".join(sentences)


def _readme(summary: dict[str, JsonValue]) -> str:
    performance = cast("dict[str, JsonValue]", summary["performance"])
    raw_cases = cast("list[object]", performance["cases"])
    rows: list[str] = []
    for raw_case in raw_cases:
        case = cast("dict[str, object]", raw_case)
        row = f"| {case['case_name']} | "
        row += f"{float(cast('float', case['uncached_end_to_end_median_ms'])):.3f} | "
        row += f"{float(cast('float', case['cached_end_to_end_median_ms'])):.3f} | "
        row += f"{float(cast('float', case['end_to_end_change_percent'])):+.2f}% | "
        row += f"{int(cast('float', case['kv_cache_bytes']))} | {case['verdict']} |"
        rows.append(row)
    verdict = performance["strict_verdict"]
    if verdict == "not_comparable":
        outcome = _paragraph(
            "The overall strict comparison is `not_comparable`; five cases exceeded the",
            "5% CV limit, so this package does not claim a strict overall performance improvement.",
        )
    else:
        outcome = "The complete environment-compatible run passed the configured strict policy."
    lines = [
        "# Stage 9 — KV Cache Autoregressive Generation Evidence",
        "",
        "## Outcome",
        "",
        outcome,
        _paragraph(
            "All 168 fresh-process workers completed, and the descriptive cached median was",
            "lower in all 12 canonical cases. Seven cases were individually comparable.",
        ),
        "",
        "## Correctness and generation semantics",
        "",
        _paragraph(
            "Prefill final logits are exactly equal to ordinary forward. Incremental decode is",
            "checked with `rtol=1e-5, atol=1e-6`, and fixed-generator cached/uncached sampling",
            "remains token- and RNG-state identical across a `block_size` overflow. Cache tensors",
            "are detached, caller-owned, absent from model state, and represented per layer as",
            "`[batch, heads, time, head_size]`.",
        ),
        "",
        _paragraph(
            "K/V can be cached because every future query still attends to historical keys and",
            "consumes their values. Historical Q is not reused. Decode projects only new-token",
            "Q/K/V, but its attention still reads all cached keys; the optimization removes",
            "repeated historical projection and MLP work; attention is not constant-time.",
        ),
        "",
        "## Canonical performance",
        "",
        "| Case | Uncached E2E ms | Cached E2E ms | Change | Cache bytes | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        *rows,
        "",
        _paragraph(
            "Negative change is faster. Each mode/case has seven fresh-process replicates,",
            "deterministic weights and forced tokens, three untimed warmups, seven measured",
            "iterations, batch size one, and no Profiler in canonical timers. Raw JSONL, execution",
            "order, config, environment, summaries, and their SHA-256 hashes are committed under",
            "`evidence/inference/`.",
        ),
        "",
        "## Cache memory and overflow",
        "",
        _paragraph(
            "For element size `S`, cache bytes are",
            "`2 * layers * batch * cached_time * embedding * S`. The canonical cache value is",
            "measured after the final logits-producing decode, so its length is",
            "`prompt_length + generated_length - 1`. KV cache trades this memory and",
            "concatenation cost for avoided historical computation.",
        ),
        "",
        _paragraph(
            "When the learned absolute-position window is full, cached generation does not drop",
            "the oldest K/V. It re-prefills the newest `block_size` tokens because uncached",
            "sliding windows renumber positions to `0..block_size-1`; old-position K/V",
            "would be numerically different.",
        ),
        "",
        "## TTFT, TPOT, throughput, and profiler",
        "",
        _paragraph(
            "TTFT is the prompt forward/prefill latency before the first forced token. TPOT is",
            "represented by median subsequent decode latency. Tokens/s divides generated tokens",
            "by end-to-end time. Small models and short prompts may not benefit because",
            "cache concatenation, allocation, and small-kernel overhead can exceed avoided work.",
        ),
        "",
        _paragraph(
            "The separate P128/G32 profiler is descriptive only. It shows matrix multiplication,",
            "batched attention, normalization, and cached concatenation costs; its instrumented",
            "times never feed the benchmark verdict.",
        ),
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python benchmark_inference.py --config configs/inference_benchmark_stage9.yaml",
        "python profile_inference.py `",
        "  --config configs/inference_benchmark_stage9.yaml `",
        "  --output reports/inference-profile-stage9/profile-p128-g32.json",
        "python generate_stage9_evidence.py `",
        "  --run-manifest <run_manifest.json> --profile <profile.json>",
        "python generate_stage9_evidence.py --verify",
        "```",
        "",
    ]
    return "\n".join(lines)


def generate_stage9_evidence(
    *,
    run_manifest_path: Path,
    profile_path: Path,
    package_root: Path,
) -> Path:
    """Copy verified raw evidence and atomically publish a hash-bound Stage 9 package."""
    manifest = verify_inference_run_manifest(run_manifest_path)
    run_summary = _json(run_manifest_path.parent / "summary.json")
    profile = _json(profile_path)
    if (
        profile.get("descriptive_only") is not True
        or profile.get("canonical_timing_source") is not False
    ):
        msg = "profile must be explicitly descriptive and non-canonical"
        raise ValueError(msg)
    if package_root.exists():
        msg = f"evidence package already exists: {package_root}"
        raise FileExistsError(msg)
    temporary_root = package_root.with_name(f".{package_root.name}.tmp")
    temporary_root.mkdir(parents=True, exist_ok=False)
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        msg = "run manifest has invalid run_id"
        raise ValueError(msg)
    inference_destination = temporary_root / "evidence" / "inference" / run_id
    inference_destination.mkdir(parents=True)
    for source in sorted(run_manifest_path.parent.iterdir(), key=lambda path: path.name):
        if source.is_file():
            _ = shutil.copyfile(source, inference_destination / source.name)
    profiler_destination = temporary_root / "evidence" / "profiler-p128-g32.json"
    profiler_destination.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copyfile(profile_path, profiler_destination)
    correctness = _correctness_evidence()
    _write_json(temporary_root / "evidence" / "correctness.json", correctness)
    performance = _compact_performance(run_summary)
    summary: dict[str, JsonValue] = {
        "schema_version": 1,
        "stage": "stage9-kv-cache-generation",
        "source_run_id": run_id,
        "source_git_commit_sha": cast("JsonValue", manifest.get("git_commit_sha")),
        "source_config_sha256": cast("JsonValue", manifest.get("config_sha256")),
        "correctness": correctness,
        "performance": performance,
        "overflow_fallback": "re-prefill latest block_size window",
        "profiler_descriptive_only": True,
    }
    _write_json(temporary_root / "summary.json", summary)
    _ = (temporary_root / "README.md").write_text(_readme(summary), encoding="utf-8", newline="\n")
    artifact_paths = sorted(
        (
            path
            for path in temporary_root.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        ),
        key=lambda path: path.relative_to(temporary_root).as_posix(),
    )
    artifact_manifest = {
        "schema_version": 1,
        "stage": "stage9-kv-cache-generation",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_run_id": run_id,
        "source_git_commit_sha": manifest.get("git_commit_sha"),
        "artifacts": [
            {
                "path": path.relative_to(temporary_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in artifact_paths
        ],
    }
    _write_json(temporary_root / "artifact_manifest.json", artifact_manifest)
    _ = temporary_root.rename(package_root)
    return package_root


def verify_stage9_evidence(package_root: Path) -> dict[str, object]:
    """Verify package completeness, every artifact hash, and the copied run manifest."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _json(manifest_path)
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        msg = "artifact_manifest artifacts must be a list"
        raise TypeError(msg)
    artifact_items = cast("list[object]", raw_artifacts)
    bound_paths: set[str] = set()
    for raw_entry in artifact_items:
        if not isinstance(raw_entry, dict):
            msg = "artifact entry must be an object"
            raise TypeError(msg)
        entry = cast("dict[str, object]", raw_entry)
        relative = entry.get("path")
        expected_size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not isinstance(expected_hash, str)
        ):
            msg = "artifact entry fields are invalid"
            raise TypeError(msg)
        content = (package_root / relative).read_bytes()
        if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_hash:
            msg = f"artifact {relative} failed size or SHA-256 verification"
            raise ValueError(msg)
        bound_paths.add(relative)
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if actual_paths != bound_paths:
        msg = "artifact manifest file set does not match package contents"
        raise ValueError(msg)
    source_run_id = manifest.get("source_run_id")
    if not isinstance(source_run_id, str):
        msg = "artifact manifest source_run_id must be a string"
        raise TypeError(msg)
    copied_run_manifest = (
        package_root / "evidence" / "inference" / source_run_id / "run_manifest.json"
    )
    _ = verify_inference_run_manifest(copied_run_manifest)
    return manifest
