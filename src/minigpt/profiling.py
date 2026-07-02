"""Capture CPU operator profiles and export human-readable evidence."""

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from torch.profiler import ProfilerActivity, profile

from minigpt.benchmark_types import BenchmarkCase
from minigpt.benchmark_workload import TrainingStepWorkload

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.benchmark_config import BenchmarkConfig


class _ProfilerEvent(Protocol):
    key: str
    count: int
    self_cpu_time_total: float
    cpu_time_total: float
    self_cpu_memory_usage: int
    input_shapes: str


@dataclass(frozen=True, slots=True)
class ProfileArtifacts:
    """Locate the operator table, Markdown report, and Chrome trace."""

    top_operators_csv: Path
    report_markdown: Path
    trace_json: Path


def _profile_case(config: BenchmarkConfig) -> BenchmarkCase:
    profile_settings = config.profile
    model = next(model for model in config.model_sizes if model.name == profile_settings.model_size)
    return BenchmarkCase(
        model_size=model.name,
        n_layer=model.n_layer,
        n_head=model.n_head,
        n_embd=model.n_embd,
        thread_count=profile_settings.thread_count,
        block_size=profile_settings.block_size,
        batch_size=profile_settings.batch_size,
    )


def _write_operator_csv(path: Path, events: list[_ProfilerEvent]) -> None:
    headers = [
        "rank",
        "operator",
        "count",
        "self_cpu_time_ms",
        "total_cpu_time_ms",
        "self_cpu_memory_bytes",
        "input_shapes",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for rank, event in enumerate(events, start=1):
            writer.writerow(
                [
                    rank,
                    event.key,
                    event.count,
                    event.self_cpu_time_total / 1_000,
                    event.cpu_time_total / 1_000,
                    event.self_cpu_memory_usage,
                    event.input_shapes,
                ]
            )


def _write_profile_report(
    path: Path,
    case: BenchmarkCase,
    active_steps: int,
    events: list[_ProfilerEvent],
    trace_path: Path,
) -> None:
    lines = [
        "# CPU Operator Profile",
        "",
        f"- Case: `{case.label}`",
        f"- Parameters: `{case.n_layer}` layers, `{case.n_head}` heads, `{case.n_embd}` embedding",
        f"- Active profiled steps: {active_steps}",
        "- Activities: CPU only",
        "- Shape recording: enabled",
        "- Memory profiling: enabled",
        "",
        "## High-level scopes",
        "",
        "- `data_preparation`: random window sampling and tensor construction.",
        "- `forward_backward`: model forward, cross entropy, backward, and gradient clipping.",
        "- `optimizer_step`: AdamW parameter update.",
        "",
        "## Operator interpretation",
        "",
        "- Attention: inspect `aten::matmul`, `aten::bmm`, `aten::softmax`, and masking ops.",
        "- MLP: inspect `aten::addmm`, `aten::mm`, and `aten::gelu`.",
        "- LayerNorm: this project is custom; inspect `aten::mean`, `aten::rsqrt`, mul, and add.",
        "- Optimizer: inspect the `optimizer_step` scope and pointwise parameter updates.",
        "- Data: compare `data_preparation` against `forward_backward`.",
        "",
        "## Top operators by self CPU time",
        "",
        "| Rank | Operator | Self CPU ms | Total CPU ms | Calls | Self memory bytes |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, event in enumerate(events[:20], start=1):
        lines.append(
            "".join(
                (
                    f"| {rank} | `{event.key}` | ",
                    f"{event.self_cpu_time_total / 1_000:.3f} | ",
                    f"{event.cpu_time_total / 1_000:.3f} | {event.count} | ",
                    f"{event.self_cpu_memory_usage} |",
                )
            )
        )
    lines.extend(
        [
            "",
            "## Chrome trace",
            "",
            f"Open `{trace_path.name}` in `chrome://tracing` or Perfetto.",
            "",
            "Profiler overhead makes these timings unsuitable for benchmark comparisons.",
        ]
    )
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_profile(config: BenchmarkConfig) -> ProfileArtifacts:
    """Profile one representative training case and export operator evidence."""
    case = _profile_case(config)
    workload = TrainingStepWorkload(
        case,
        seed=config.seed,
        vocab_size=config.vocab_size,
    )
    for _ in range(config.profile.warmup_steps):
        workload.step()

    output_dir = config.output_dir / "profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_json = output_dir / "trace.json"
    top_operators_csv = output_dir / "top_operators.csv"
    report_markdown = output_dir / "profile_report.md"
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        acc_events=True,
    ) as profiler:
        for _ in range(config.profile.active_steps):
            workload.profiled_step()
            profiler.step()
    profiler.export_chrome_trace(str(trace_json))

    averaged_events = cast(
        "list[_ProfilerEvent]",
        profiler.key_averages(group_by_input_shape=True),
    )
    all_events = sorted(
        averaged_events,
        key=lambda event: event.self_cpu_time_total,
        reverse=True,
    )
    selected_events = list(all_events[:50])
    selected_names = {event.key for event in selected_events}
    required_scopes = {"data_preparation", "forward_backward", "optimizer_step"}
    selected_events.extend(
        event
        for event in all_events
        if event.key in required_scopes and event.key not in selected_names
    )
    _write_operator_csv(top_operators_csv, selected_events)
    _write_profile_report(
        report_markdown,
        case,
        config.profile.active_steps,
        all_events,
        trace_json,
    )
    return ProfileArtifacts(top_operators_csv, report_markdown, trace_json)
