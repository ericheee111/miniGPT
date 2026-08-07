"""Generate canonical Stage 13B correctness and descriptive performance evidence."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from minigpt.paged_attention_benchmark import (
    PagedAttentionBenchmarkConfig,
    write_paged_attention_benchmark,
)
from minigpt.stage13b_evidence import (
    generate_stage13b_correctness,
    generate_stage13b_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run Stage 13B tests, correctness simulations, benchmark, and packaging."""
    if argv:
        reason = "generate_stage13b_evidence.py does not accept arguments"
        raise ValueError(reason)
    project_root = Path(__file__).resolve().parent
    test_files = [
        "tests/test_model.py",
        "tests/test_paged_kv_cache.py",
        "tests/test_paged_serving.py",
        "tests/test_serving_simulator.py",
        "tests/test_paged_attention_benchmark.py",
        "tests/test_serve_subprocess.py",
    ]
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", *test_files, "-q"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        _ = sys.stderr.write(completed.stdout)
        _ = sys.stderr.write(completed.stderr)
        return completed.returncode
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lifecycle = {
        "schema_version": 1,
        "exit_code": completed.returncode,
        "test_files": test_files,
        "covered_contracts": [
            "paged_logits_and_kv_delta",
            "no_dense_materialization_guard",
            "three_strategy_logical_equivalence",
            "overflow_dense_reprefill",
            "cancellation_and_zero_leaks",
            "http_cli_direct_executor",
            "descriptive_cpu_benchmark",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="minigpt-stage13b-") as temporary:
        temporary_root = Path(temporary)
        lifecycle_path = temporary_root / "lifecycle_tests.json"
        _ = lifecycle_path.write_text(
            json.dumps(lifecycle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        correctness_path = generate_stage13b_correctness(
            config_root=project_root / "configs",
            work_root=temporary_root / "correctness-runs",
            output_path=temporary_root / "correctness.json",
        )
        benchmark_path = write_paged_attention_benchmark(
            temporary_root / "benchmark.json",
            config=PagedAttentionBenchmarkConfig(),
        )
        package = generate_stage13b_evidence(
            correctness_path=correctness_path,
            benchmark_path=benchmark_path,
            lifecycle_path=lifecycle_path,
            package_root=project_root / "docs" / "results" / "paged-attention",
            source_commit=source_commit,
        )
    _ = sys.stdout.write(f"{package}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
