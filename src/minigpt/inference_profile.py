"""Create a separate descriptive CPU operator profile for Stage 9 inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from minigpt.model import GPT

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.inference_benchmark_config import InferenceBenchmarkConfig, JsonValue


class _ProfilerEvent(Protocol):
    """Describe the compact operator fields retained as evidence."""

    key: str
    count: int
    self_cpu_time_total: float
    cpu_time_total: float
    self_cpu_memory_usage: int
    input_shapes: str


class _Profiler(Protocol):
    """Describe the profiler aggregation method used by compact evidence."""

    def key_averages(self, *, group_by_input_shape: bool = False) -> list[_ProfilerEvent]:
        """Return aggregated operator events."""
        ...


@dataclass(frozen=True, slots=True)
class InferenceProfileArtifacts:
    """Locate one descriptive profile JSON document."""

    output_path: Path


@torch.no_grad()
def _profile_workload(
    model: GPT,
    prompt: Tensor,
    forced_tokens: Tensor,
    *,
    cached: bool,
) -> None:
    generated = prompt
    if cached:
        _, cache = model.prefill(prompt)
        for token_index in range(forced_tokens.shape[1]):
            next_token = forced_tokens[:, token_index : token_index + 1]
            generated = torch.cat((generated, next_token), dim=1)
            if token_index + 1 < forced_tokens.shape[1]:
                _, cache = model.decode(next_token, cache)
    else:
        for token_index in range(forced_tokens.shape[1]):
            _ = cast("tuple[Tensor, Tensor | None]", model(generated))
            next_token = forced_tokens[:, token_index : token_index + 1]
            generated = torch.cat((generated, next_token), dim=1)


def _operator_rows(profiler: _Profiler, *, limit: int) -> list[JsonValue]:
    events = profiler.key_averages(group_by_input_shape=True)
    ranked = sorted(events, key=lambda event: event.self_cpu_time_total, reverse=True)[:limit]
    return [
        {
            "operator": event.key,
            "count": event.count,
            "self_cpu_time_us": event.self_cpu_time_total,
            "total_cpu_time_us": event.cpu_time_total,
            "self_cpu_memory_bytes": event.self_cpu_memory_usage,
            "input_shapes": event.input_shapes,
        }
        for event in ranked
    ]


def run_inference_profile(
    config: InferenceBenchmarkConfig,
    output_path: Path,
    *,
    prompt_length: int = 128,
    generated_length: int = 32,
    operator_limit: int = 20,
) -> InferenceProfileArtifacts:
    """Profile cached and uncached forced-token inference outside canonical timers."""
    if prompt_length + generated_length > config.model.block_size:
        msg = "profile case exceeds block_size"
        raise ValueError(msg)
    _ = torch.default_generator.manual_seed(config.benchmark_seed)
    model = GPT(config.model)
    _ = model.eval()
    generator = torch.Generator(device="cpu")
    _ = generator.manual_seed(config.benchmark_seed + 1)
    prompt = torch.randint(
        config.vocab_size,
        (config.batch_size, prompt_length),
        generator=generator,
        dtype=torch.long,
    )
    forced_tokens = torch.randint(
        config.vocab_size,
        (config.batch_size, generated_length),
        generator=generator,
        dtype=torch.long,
    )
    modes: dict[str, JsonValue] = {}
    for name, cached in (("uncached", False), ("cached", True)):
        _profile_workload(model, prompt, forced_tokens, cached=cached)
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            _profile_workload(model, prompt, forced_tokens, cached=cached)
        modes[name] = {
            "top_operators": _operator_rows(cast("_Profiler", profiler), limit=operator_limit),
        }
    document: dict[str, JsonValue] = {
        "schema_version": 1,
        "stage": "stage9-kv-cache-generation",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "descriptive_only": True,
        "canonical_timing_source": False,
        "warning": "Profiler timings include instrumentation overhead and do not affect verdicts.",
        "case": {
            "batch_size": config.batch_size,
            "prompt_length": prompt_length,
            "generated_length": generated_length,
        },
        "modes": modes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return InferenceProfileArtifacts(output_path=output_path)
