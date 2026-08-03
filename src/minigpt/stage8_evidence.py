"""Regenerate the committed Stage 8 evidence package from bound artifacts."""

# ruff: noqa: E501 - the generated Markdown template is intentionally reviewable as prose.

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import TYPE_CHECKING, cast

from minigpt.batcher_benchmark_evidence import (
    LoadedBatcherBenchmarkRun,
    compare_batcher_benchmarks,
    load_batcher_benchmark_run,
    write_batcher_comparison,
)
from minigpt.benchmark_v2_comparison_policy import load_comparison_policy

if TYPE_CHECKING:
    from pathlib import Path

    from minigpt.benchmark_v2_config import JsonValue

_CANDIDATE_GIT_SHA = "3d6ae83e5c0b37a423d60abfb73377cf02fab51f"
_LOWER_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_RUN_LABELS = ("baseline-a", "candidate-a", "baseline-b", "candidate-b")
_COMPARISONS = (
    ("baseline-a", "candidate-a"),
    ("baseline-a", "candidate-b"),
    ("baseline-b", "candidate-a"),
    ("baseline-b", "candidate-b"),
    ("baseline-a", "baseline-b"),
    ("candidate-a", "candidate-b"),
)


def _json_document(path: Path) -> dict[str, JsonValue]:
    """Load a generated or retained JSON object."""
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        msg = f"{path} must contain a JSON object"
        raise TypeError(msg)
    raw = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw):
        msg = f"{path} must contain a string-keyed JSON object"
        raise ValueError(msg)
    return cast("dict[str, JsonValue]", raw)


def _write_json(path: Path, document: dict[str, JsonValue]) -> None:
    """Write deterministic finite JSON."""
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Hash one committed artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_manifest_path(result_directory: Path, label: str) -> Path:
    """Resolve the one run package committed under a stable label."""
    candidates = tuple(
        (result_directory / "evidence" / "batch-only" / label).glob("*/run_manifest.json")
    )
    if len(candidates) != 1:
        msg = f"expected exactly one run manifest for {label}, found {len(candidates)}"
        raise ValueError(msg)
    return candidates[0]


def _comparison_path(result_directory: Path, baseline: str, candidate: str) -> Path:
    """Return the deterministic path for one comparison."""
    return (
        result_directory
        / "evidence"
        / "batch-only"
        / "comparisons"
        / f"{baseline}-vs-{candidate}.json"
    )


def _comparison_cases(document: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """Return the typed case list from one generated comparison."""
    cases = document["cases"]
    if not isinstance(cases, list) or any(not isinstance(item, dict) for item in cases):
        msg = "comparison cases must be a list of objects"
        raise ValueError(msg)
    return cast("list[dict[str, JsonValue]]", cases)


def _format_percent(value: JsonValue) -> str:
    """Format one comparison percentage without hand-entered report values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):+.2f}%"


def _format_number(value: JsonValue, digits: int) -> str:
    """Format one finite comparison number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _cases_by_name(document: dict[str, JsonValue]) -> dict[str, dict[str, JsonValue]]:
    """Index generated comparison cases by display name."""
    return {cast("str", case["case_name"]): case for case in _comparison_cases(document)}


def _write_role_manifest(
    result_directory: Path,
    role: str,
    runs: dict[str, LoadedBatcherBenchmarkRun],
) -> None:
    """Replace legacy filename-only batch manifests with hash-bound run indexes."""
    selected_labels = tuple(label for label in _RUN_LABELS if label.startswith(role))
    selected = tuple(runs[label] for label in selected_labels)
    git_shas = {run.git_commit_sha for run in selected}
    config_shas = {run.config_sha256 for run in selected}
    if len(git_shas) != 1 or len(config_shas) != 1:
        msg = f"{role} runs do not share one implementation and config identity"
        raise ValueError(msg)
    document: dict[str, JsonValue] = {
        "schema_version": 2,
        "methodology_version": 2,
        "role": role,
        "git_commit_sha": git_shas.pop(),
        "config_sha256": config_shas.pop(),
        "runs": [
            {
                "label": label,
                "run_id": runs[label].run_id,
                "run_manifest_path": (
                    runs[label].manifest_path.relative_to(result_directory).as_posix()
                ),
                "run_manifest_sha256": runs[label].manifest_sha256,
            }
            for label in selected_labels
        ],
    }
    _write_json(result_directory / "evidence" / f"batcher-{role}-manifest.json", document)


def _build_batch_summary(
    comparisons: dict[str, dict[str, JsonValue]],
    runs: dict[str, LoadedBatcherBenchmarkRun],
) -> dict[str, JsonValue]:
    """Build machine conclusions exclusively from strict runs and comparison artifacts."""
    cross_names = (
        "baseline-a-vs-candidate-a",
        "baseline-a-vs-candidate-b",
        "baseline-b-vs-candidate-a",
        "baseline-b-vs-candidate-b",
    )
    cross_changes = [
        cast("float", case["batch_time_change_percent"])
        for name in cross_names
        for case in _comparison_cases(comparisons[name])
    ]
    all_candidate_lower = all(change < 0.0 for change in cross_changes)
    same_code: dict[str, JsonValue] = {}
    for name in ("baseline-a-vs-baseline-b", "candidate-a-vs-candidate-b"):
        changes = [
            cast("float", case["batch_time_change_percent"])
            for case in _comparison_cases(comparisons[name])
        ]
        same_code[name] = {
            "verdict": comparisons[name]["verdict"],
            "batch_time_change_percent_range": [min(changes), max(changes)],
        }
    return {
        "methodology_version": 2,
        "config_sha256": runs["baseline-a"].config_sha256,
        "execution_order": list(_RUN_LABELS),
        "runs": {
            label: {
                "run_id": run.run_id,
                "git_commit_sha": run.git_commit_sha,
                "run_manifest_sha256": run.manifest_sha256,
                "successful_replicates": sum(summary.success_count for summary in run.summaries),
                "stable_cases": sum(summary.stability == "stable" for summary in run.summaries),
                "cases": [cast("dict[str, JsonValue]", asdict(item)) for item in run.summaries],
            }
            for label, run in runs.items()
        },
        "comparisons": {
            name: {
                "artifact": f"evidence/batch-only/comparisons/{name}.json",
                "verdict": document["verdict"],
                "environment_mismatches": document["environment_mismatches"],
            }
            for name, document in comparisons.items()
        },
        "same_code_drift": same_code,
        "all_candidate_medians_lower_than_all_baseline_medians": all_candidate_lower,
        "strict_improvement_detected": all(
            comparisons[name]["verdict"] == "pass" for name in cross_names
        ),
        "cross_comparison_batch_time_change_percent_range": [
            min(cross_changes),
            max(cross_changes),
        ],
    }


def _build_report(
    comparisons: dict[str, dict[str, JsonValue]],
    summary: dict[str, JsonValue],
) -> str:
    """Render report tables from generated machine-readable comparisons."""
    aa = comparisons["baseline-a-vs-candidate-a"]
    bb = comparisons["baseline-b-vs-candidate-b"]
    aa_cases = _cases_by_name(aa)
    bb_cases = _cases_by_name(bb)
    case_order = tuple(cast("str", case["case_name"]) for case in _comparison_cases(aa))
    run_rows: list[str] = []
    for case_name in case_order:
        first = aa_cases[case_name]
        second = bb_cases[case_name]
        run_rows.append(
            "| "
            + " | ".join(
                (
                    case_name.replace("char-gpt-", "").upper().replace("-", " / "),
                    _format_number(first["baseline_median_batch_time_ms"], 6),
                    _format_number(first["baseline_cv_percent"], 2),
                    _format_number(first["candidate_median_batch_time_ms"], 6),
                    _format_number(first["candidate_cv_percent"], 2),
                    _format_number(second["baseline_median_batch_time_ms"], 6),
                    _format_number(second["baseline_cv_percent"], 2),
                    _format_number(second["candidate_median_batch_time_ms"], 6),
                    _format_number(second["candidate_cv_percent"], 2),
                )
            )
            + " |"
        )
    cross_rows: list[str] = []
    for baseline in ("baseline-a", "baseline-b"):
        for candidate in ("candidate-a", "candidate-b"):
            name = f"{baseline}-vs-{candidate}"
            document = comparisons[name]
            changes = [
                cast("float", case["batch_time_change_percent"])
                for case in _comparison_cases(document)
            ]
            cross_rows.append(
                "".join(
                    (
                        f"| {baseline} → {candidate} | ",
                        f"{_format_percent(min(changes))} to ",
                        f"{_format_percent(max(changes))} | {document['verdict']} |",
                    )
                )
            )
    drift_rows: list[str] = []
    for name in ("baseline-a-vs-baseline-b", "candidate-a-vs-candidate-b"):
        document = comparisons[name]
        drift_rows.extend(
            "".join(
                (
                    f"| {name} | {case['case_name']} | ",
                    f"{_format_percent(case['batch_time_change_percent'])} | ",
                    f"{_format_number(case['baseline_cv_percent'], 2)}% → ",
                    f"{_format_number(case['candidate_cv_percent'], 2)}% |",
                )
            )
            for case in _comparison_cases(document)
        )
    reference = cast("dict[str, JsonValue]", summary["reference_benchmark"])
    run_table = "\n".join(run_rows)
    cross_table = "\n".join(cross_rows)
    drift_table = "\n".join(drift_rows)
    return f"""# Stage 8 — TokenBatcher / mmap Optimization Evidence

## Outcome

Stage 8 removes the construction-time full-corpus `uint16` to `int64` copy and the Python row loop. The candidate keeps the read-only mmap, gathers selected windows, performs one local `int64` conversion, and returns overlapping read-only contract views from one tensor owner.

The candidate used `{_CANDIDATE_GIT_SHA}`. The new alternating batch-only rerun completed all 112 requested worker replicates, and both candidate runs had lower medians than both baseline runs in every case. However, at least one side of every comparison exceeded the 5% CV limit, so all six strict comparison verdicts are `not_comparable`. This package therefore supports only a descriptive same-host observation, not a stable performance claim.

The earlier B4/T128 `-10.10%` and B16/T256 candidate `4.97%` CV values are not carried forward as conclusions: their raw replicates were not committed and the new same-code runs did not reproduce that stability.

## Batch-only method and provenance

Runs used the host-specific i7-14700 config, a 1,000,000-token read-only `uint16` mmap, 500 warmup batches, 20,000 measured batches, 7 fresh worker processes per case, and the order `baseline A → candidate A → baseline B → candidate B`. Each run manifest binds raw JSONL, resolved config, environment, execution order, CSV and Markdown summaries by SHA-256 and byte size. The strict loader recomputes median, MAD, population CV, tokens/s, worker-lifetime peak RSS and stability before comparison.

Comparison policy SHA-256: `{aa["policy_sha256"]}`. Environment mismatches: `{aa["environment_mismatches"]}` for baseline A versus candidate A; all other committed comparisons also report an empty mismatch list.

## Four-run batch-only results

Every numeric cell below is rendered from the committed comparison JSON. Times are medians in milliseconds; CV is population CV in percent.

| Case | Base A ms | Base A CV | Cand A ms | Cand A CV | Base B ms | Base B CV | Cand B ms | Cand B CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{run_table}

| Comparison | Descriptive batch-time change range | Strict verdict |
| --- | ---: | --- |
{cross_table}

Negative change is faster. All 16 candidate-versus-baseline case medians are lower, but `not_comparable` prevents converting that consistency into a pass or an improvement claim.

## Same-code cross-run drift

| Comparison | Case | Median change | CV change |
| --- | --- | ---: | ---: |
{drift_table}

Baseline drift is materially larger than candidate drift in several cases, which confirms that this host/run method was not stable enough for a strict verdict.

## Reference training-step evidence and preconditioning

The retained historical training-step comparisons report `{reference["verdicts"]}` against the two baselines. Those runs predate Benchmark v2 schema 3: their resolved config did not bind the reported external 120-second preconditioning step. They are therefore retained as legacy descriptive evidence only, and this report no longer claims that their exact resolved config covered that manual operation.

Benchmark v2 schema 3 now records preconditioning in the resolved config, methodology identity and run manifest. The i7-14700 Stage 8 config enables a 120-second training-step precondition, so the reproduction command actually performs and records it.

## Reproduction

```powershell
python -m pip install -e ".[dev,report]"
python benchmark_batcher.py --config configs/batcher_benchmark_i7_14700_stage8.yaml
python compare_batcher_benchmarks.py `
  --baseline <baseline-run-manifest> `
  --candidate <candidate-run-manifest> `
  --policy configs/benchmark_v2_comparison.yaml
python benchmark_v2.py --config configs/benchmark_v2_i7_14700_stage8.yaml
python profile_benchmark_v2.py --config configs/benchmark_v2_stage8_profile.yaml
python generate_stage8_evidence.py
```

`benchmark_v2_reference.yaml` and `batcher_benchmark_reference.yaml` remain portable templates with no host affinity. The explicitly named i7-14700 configs carry this host's affinity and Stage 8 measurement settings.

## Evidence and limitations

The committed batch-only package contains all four small raw JSONL files and every artifact needed to independently recompute raw → summary → comparison → report. Worker-lifetime peak RSS includes imports, mmap construction, warmup and measurement. The microbenchmark is not a model benchmark, and no result establishes statistical significance, cross-machine superiority, or GPU behavior.

Stage 8 is not ready for a performance-evidence PR claim: the batch-only reruns failed the stability policy, and the historical end-to-end preconditioning was not bound by its resolved config. The implementation and evidence machinery can still be reviewed independently of that withdrawn claim.
"""


def generate_stage8_evidence(
    result_directory: Path,
    policy_path: Path,
    *,
    generated_by_git_sha: str,
) -> None:
    """Regenerate comparisons, indexes, report, summary, and the package hash manifest."""
    if _LOWER_GIT_SHA.fullmatch(generated_by_git_sha) is None:
        msg = "generated_by_git_sha must be 40 lowercase hexadecimal characters"
        raise ValueError(msg)
    runs = {
        label: load_batcher_benchmark_run(_run_manifest_path(result_directory, label))
        for label in _RUN_LABELS
    }
    candidate_shas = {runs[label].git_commit_sha for label in ("candidate-a", "candidate-b")}
    if candidate_shas != {_CANDIDATE_GIT_SHA}:
        msg = f"candidate run identities differ from {_CANDIDATE_GIT_SHA}"
        raise ValueError(msg)
    policy = load_comparison_policy(policy_path)
    comparison_directory = result_directory / "evidence" / "batch-only" / "comparisons"
    comparison_directory.mkdir(parents=True, exist_ok=True)
    comparisons: dict[str, dict[str, JsonValue]] = {}
    for baseline, candidate in _COMPARISONS:
        name = f"{baseline}-vs-{candidate}"
        path = _comparison_path(result_directory, baseline, candidate)
        comparison = compare_batcher_benchmarks(
            runs[baseline].manifest_path,
            runs[candidate].manifest_path,
            policy,
        )
        write_batcher_comparison(comparison, path)
        comparisons[name] = _json_document(path)
    _write_role_manifest(result_directory, "baseline", runs)
    _write_role_manifest(result_directory, "candidate", runs)

    reference_candidate = _json_document(
        result_directory / "evidence" / "reference-candidate-manifest.json"
    )
    reference_git = cast("dict[str, JsonValue]", reference_candidate["git"])
    if reference_git["commit_sha"] != _CANDIDATE_GIT_SHA:
        msg = "reference candidate Git SHA differs from the batch-only candidate"
        raise ValueError(msg)
    reference_comparisons = tuple(
        _json_document(
            result_directory / "evidence" / f"reference-comparison-baseline-{index}.json"
        )
        for index in (1, 2)
    )
    batch_summary = _build_batch_summary(comparisons, runs)
    summary: dict[str, JsonValue] = {
        "schema_version": 2,
        "stage": "stage8-batcher-mmap-optimization",
        "baseline_git_commit_sha": runs["baseline-a"].git_commit_sha,
        "candidate_git_commit_sha": _CANDIDATE_GIT_SHA,
        "comparison_policy_sha256": policy.sha256,
        "batch_only": batch_summary,
        "reference_benchmark": {
            "legacy_preconditioning_provenance_bound": False,
            "config_sha256": reference_candidate["config_sha256"],
            "candidate_run_id": reference_candidate["run_id"],
            "verdicts": [document["verdict"] for document in reference_comparisons],
        },
        "conclusions": {
            "batch_only_descriptive_all_candidate_medians_lower": batch_summary[
                "all_candidate_medians_lower_than_all_baseline_medians"
            ],
            "batch_only_strict_improvement_detected": False,
            "end_to_end_improvement_detected": False,
            "cross_machine_claim": False,
            "performance_evidence_pr_ready": False,
        },
    }
    _write_json(result_directory / "summary.json", summary)
    _ = (result_directory / "README.md").write_text(
        _build_report(comparisons, summary),
        encoding="utf-8",
    )

    artifact_paths = sorted(
        (
            path
            for path in result_directory.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        ),
        key=lambda path: path.relative_to(result_directory).as_posix(),
    )
    manifest: dict[str, JsonValue] = {
        "schema_version": 2,
        "package": "stage8-batcher-mmap-optimization",
        "generated_by_git_sha": generated_by_git_sha,
        "candidate_git_commit_sha": _CANDIDATE_GIT_SHA,
        "source_identities": {
            "batcher_config_sha256": runs["baseline-a"].config_sha256,
            "comparison_policy_sha256": policy.sha256,
            "reference_config_sha256": reference_candidate["config_sha256"],
        },
        "artifacts": [
            {
                "path": path.relative_to(result_directory).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    _write_json(result_directory / "artifact_manifest.json", manifest)
