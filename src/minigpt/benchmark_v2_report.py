"""Atomically write and load durable CPU Benchmark v2 run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import psutil
import torch
import yaml

from minigpt.benchmark_v2_config import (
    JsonValue,
    case_identity,
    load_resolved_benchmark_v2_config,
    resolved_config_document,
    resolved_config_sha256,
)
from minigpt.benchmark_v2_statistics import BenchmarkV2Summary, summarize_replicates

if TYPE_CHECKING:
    from collections.abc import Iterable

    from minigpt.benchmark_v2_types import BenchmarkV2Config

RunStatus = Literal["complete", "partial", "failed"]
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "status",
        "started_at_utc",
        "ended_at_utc",
        "config_sha256",
        "git",
        "expected_task_count",
        "completed_task_count",
        "successful_task_count",
        "failed_task_count",
        "case_identities",
        "artifacts",
    }
)
_ARTIFACT_ENTRY_KEYS = frozenset({"path", "size_bytes", "sha256"})
_SHA256_HEX_LENGTH = 64
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")
_ENVIRONMENT_SCHEMA_VERSION = 2
_GIT_COMMIT_SHA_LENGTHS = frozenset({40, 64})
_GIT_IDENTITY_KEYS = frozenset({"commit_sha", "branch", "dirty"})
_ENVIRONMENT_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "run_status",
        "config_sha256",
        "run_environment",
        "worker_environments",
    }
)
_REQUIRED_BOUND_ARTIFACTS = frozenset(
    {
        "environment.json",
        "resolved_config.yaml",
        "raw_replicates.jsonl",
        "summary.csv",
        "summary.md",
        "execution_order.json",
    }
)
_FINALIZED_RUN_ENTRIES = _REQUIRED_BOUND_ARTIFACTS | frozenset({"run_manifest.json"})


class _PriorityProcess(Protocol):
    """Describe the psutil priority method absent from the installed type stubs."""

    def nice(self) -> int | str:
        """Return the current process priority without changing it."""
        ...


class _RawReplicateLike(Protocol):
    """Describe the raw record fields reporting consumes without importing orchestration."""

    @property
    def status(self) -> Literal["ok", "error"]: ...

    @property
    def case_identity(self) -> str: ...

    @property
    def case_name(self) -> str: ...

    @property
    def replicate_index(self) -> int: ...

    @property
    def worker_pid(self) -> int | None: ...

    @property
    def started_at_utc(self) -> str | None: ...

    @property
    def ended_at_utc(self) -> str | None: ...

    @property
    def return_code(self) -> int | None: ...

    @property
    def error_type(self) -> str | None: ...

    @property
    def message(self) -> str | None: ...

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...

    @property
    def worker_response(self) -> dict[str, JsonValue] | None: ...


class _BenchmarkCaseLike(Protocol):
    """Expose the case label needed by execution-order rendering."""

    @property
    def name(self) -> str: ...


class _BenchmarkTaskLike(Protocol):
    """Describe task identity without creating a reporting/orchestration import cycle."""

    @property
    def case(self) -> _BenchmarkCaseLike: ...

    @property
    def case_identity(self) -> str: ...

    @property
    def replicate_index(self) -> int: ...

    @property
    def benchmark_seed(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ArtifactManifestEntry:
    """Bind one run-relative artifact path to its exact bytes."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    """Bind one regular file's bytes to metadata observed before and after its single read."""

    content: bytes
    metadata: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Represent the self-excluded durable identity of one benchmark invocation."""

    schema_version: int
    run_id: str
    status: RunStatus
    started_at_utc: str
    ended_at_utc: str
    config_sha256: str
    git: dict[str, JsonValue]
    expected_task_count: int
    completed_task_count: int
    successful_task_count: int
    failed_task_count: int
    case_identities: tuple[dict[str, str], ...]
    artifacts: tuple[ArtifactManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class RunArtifactPaths:
    """Expose every final artifact path for a completed, partial, or failed run."""

    run_directory: Path
    run_manifest_path: Path
    environment_path: Path
    resolved_config_path: Path
    raw_replicates_path: Path
    summary_csv_path: Path
    summary_markdown_path: Path
    execution_order_path: Path


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Durably replace one file using a unique sibling temporary file."""
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("xb") as stream:
            _ = stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _ = temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _json_bytes(document: JsonValue) -> bytes:
    """Encode a human-readable, deterministic JSON artifact."""
    return (
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(documents: Iterable[dict[str, JsonValue]]) -> bytes:
    """Encode each raw record as one compact, independently parseable JSON line."""
    return b"".join(
        (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for document in documents
    )


def _yaml_bytes(document: dict[str, JsonValue]) -> bytes:
    """Encode the fully resolved configuration without Python-specific YAML tags."""
    rendered = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
    return rendered.encode("utf-8")


def _csv_bytes(summaries: Iterable[BenchmarkV2Summary]) -> bytes:
    """Render compact tabular summaries without rounding away raw precision."""
    fieldnames = tuple(BenchmarkV2Summary.__dataclass_fields__)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for summary in summaries:
        writer.writerow(asdict(summary))
    return stream.getvalue().encode("utf-8")


def _raw_replicate_document(record: _RawReplicateLike) -> dict[str, JsonValue]:
    """Serialize every raw worker field without filtering evidence or outliers."""
    return {
        "status": record.status,
        "case_identity": record.case_identity,
        "case_name": record.case_name,
        "replicate_index": record.replicate_index,
        "worker_pid": record.worker_pid,
        "started_at_utc": record.started_at_utc,
        "ended_at_utc": record.ended_at_utc,
        "return_code": record.return_code,
        "error_type": record.error_type,
        "message": record.message,
        "stdout": record.stdout,
        "stderr": record.stderr,
        "worker_response": record.worker_response,
    }


def _worker_environment(record: _RawReplicateLike) -> dict[str, JsonValue] | None:
    """Extract one successful worker's already validated environment evidence."""
    if record.status != "ok" or record.worker_response is None:
        return None
    environment = record.worker_response.get("environment")
    if not isinstance(environment, dict):
        msg = "successful worker response has no environment"
        raise TypeError(msg)
    return cast("dict[str, JsonValue]", environment)


def _git_identity() -> dict[str, str | bool | None]:
    """Capture Git identity when the report is generated inside a checkout."""
    executable = shutil.which("git")
    if executable is None:
        return {"commit_sha": None, "branch": None, "dirty": None}

    def run_git(*arguments: str) -> str | None:
        completed = subprocess.run(  # noqa: S603 - arguments are fixed Git subcommands.
            [executable, *arguments],
            capture_output=True,
            check=False,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    dirty_output = run_git("status", "--porcelain")
    return {
        "commit_sha": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(dirty_output) if dirty_output is not None else None,
    }


def _power_scheme_evidence() -> dict[str, str | None]:
    """Capture the active Windows power scheme when the native utility is available."""
    if sys.platform != "win32":
        return {"value": None, "reason": "not_windows"}
    executable = shutil.which("powercfg")
    if executable is None:
        return {"value": None, "reason": "powercfg_unavailable"}
    completed = subprocess.run(  # noqa: S603 - command and arguments are fixed.
        [executable, "/getactivescheme"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return {"value": None, "reason": f"powercfg_exit_{completed.returncode}"}
    return {"value": (completed.stdout or "").strip() or None, "reason": None}


def _nonempty_cpu_text(value: str | None) -> str | None:
    """Normalize one optional CPU identity value without inventing unavailable evidence."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _linux_cpu_name_from_cpuinfo(cpuinfo_text: str | None) -> str | None:
    """Select a Linux CPU brand deterministically from a supplied procfs document."""
    if cpuinfo_text is None:
        return None
    values: dict[str, str] = {}
    for line in cpuinfo_text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            normalized = _nonempty_cpu_text(value)
            if normalized is not None:
                _ = values.setdefault(key.strip().casefold(), normalized)
    for key in ("model name", "hardware", "processor"):
        if key in values:
            return values[key]
    return None


def resolve_cpu_name(
    *,
    platform_processor: str | None,
    uname_processor: str | None,
    system: str,
    linux_cpuinfo_text: str | None,
    windows_processor_identifier: str | None,
) -> str | None:
    """Resolve CPU identity from platform APIs before platform-specific evidence fallbacks."""
    for value in (platform_processor, uname_processor):
        identity = _nonempty_cpu_text(value)
        if identity is not None:
            return identity
    if system.casefold() == "linux":
        return _linux_cpu_name_from_cpuinfo(linux_cpuinfo_text)
    if system.casefold() == "windows":
        return _nonempty_cpu_text(windows_processor_identifier)
    return None


def _linux_cpuinfo_text(system: str) -> str | None:
    """Read Linux CPU identity evidence only when the active platform exposes procfs."""
    if system.casefold() != "linux":
        return None
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _captured_cpu_name() -> str | None:
    """Capture CPU identity while preserving an explicit unavailable discovery result."""
    system = platform.system()
    return resolve_cpu_name(
        platform_processor=platform.processor(),
        uname_processor=platform.uname().processor,
        system=system,
        linux_cpuinfo_text=_linux_cpuinfo_text(system),
        windows_processor_identifier=os.environ.get("PROCESSOR_IDENTIFIER"),
    )


def capture_run_environment(config: BenchmarkV2Config) -> dict[str, JsonValue]:
    """Capture immutable parent-process environment evidence before the first worker starts."""
    process = cast("_PriorityProcess", cast("object", psutil.Process()))
    return cast(
        "dict[str, JsonValue]",
        {
            "captured_before_first_worker": True,
            "git": cast("JsonValue", _git_identity()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_name": _captured_cpu_name(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "parent_torch_num_threads": torch.get_num_threads(),
            "parent_torch_num_interop_threads": torch.get_num_interop_threads(),
            "configured_torch_num_interop_threads": config.torch_num_interop_threads,
            "configured_cpu_affinity": list(config.cpu_affinity) if config.cpu_affinity else None,
            "relevant_environment_variables": {
                name: os.environ.get(name) for name in config.relevant_environment_variables
            },
            "process_priority": str(process.nice()),
            "power_scheme": cast("JsonValue", _power_scheme_evidence()),
        },
    )


def _environment_document(
    *,
    config: BenchmarkV2Config,
    run_id: str,
    status: RunStatus,
    raw_replicates: tuple[_RawReplicateLike, ...],
    environment_snapshot: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Preserve run identity and every successful worker environment without pretending sameness."""
    environments = [
        environment
        for record in raw_replicates
        if (environment := _worker_environment(record)) is not None
    ]
    return cast(
        "dict[str, JsonValue]",
        {
            "schema_version": 2,
            "run_id": run_id,
            "run_status": status,
            "config_sha256": resolved_config_sha256(config),
            "run_environment": environment_snapshot,
            "worker_environments": environments,
        },
    )


def _execution_order_document(
    tasks: tuple[_BenchmarkTaskLike, ...], raw_replicates: tuple[_RawReplicateLike, ...]
) -> list[JsonValue]:
    """Render fixed execution order together with eventual process and result state."""
    records_by_task = {
        (record.case_identity, record.replicate_index): record for record in raw_replicates
    }

    def result_fields(task: _BenchmarkTaskLike) -> tuple[int | None, str]:
        record = records_by_task.get((task.case_identity, task.replicate_index))
        return (record.worker_pid, record.status) if record is not None else (None, "pending")

    return [
        {
            "execution_index": index,
            "task_id": f"{task.case_identity}:{task.replicate_index}",
            "case_name": task.case.name,
            "case_identity": task.case_identity,
            "replicate_index": task.replicate_index,
            "worker_seed": task.benchmark_seed,
            "worker_pid": result_fields(task)[0],
            "status": result_fields(task)[1],
        }
        for index, task in enumerate(tasks)
    ]


def _summary_markdown(
    *,
    run_id: str,
    status: RunStatus,
    config: BenchmarkV2Config,
    summaries: tuple[BenchmarkV2Summary, ...],
) -> str:
    """Explain results and methodology without presenting partial evidence as success."""
    heading = (
        "Benchmark v2 run complete."
        if status == "complete"
        else (
            "This run is partial and does not claim benchmark success."
            if status == "partial"
            else "This run failed and does not claim benchmark success."
        )
    )
    rows = "\n".join(_summary_row(summary) for summary in summaries)
    return f"""# Benchmark v2 summary: {run_id}

{heading}

## Results

| Case | Successful/Raw replicates | Median step time (ms) | CV (%) | Stability |
| --- | ---: | ---: | ---: | --- |
{rows}

## Methodology

Timer boundary: one `time.perf_counter()` interval around the complete measurement loop: batch
acquisition, `optimizer.zero_grad(set_to_none=True)`, model forward and cross-entropy loss,
backward, gradient clipping, and optimizer step. It excludes worker startup/imports, environment
and thread setup, model/optimizer/batcher construction, warmup, pre-timer garbage collection,
post-timer memory/environment reads, JSON transport, logging/report writes, profiler
instrumentation, checkpointing, validation, and text generation.

Memory method: `final_rss_mib` is read immediately after the canonical loop. `peak_rss_mib` is an
OS-native process-lifetime high-water mark (Windows peak working set or Linux getrusage
ru_maxrss), with no sampling thread and `peak_rss_sampling_interval_ms: null`; it includes
imports, construction, warmup, and measurement and is not model-only memory.

Stability threshold: `insufficient_samples` means fewer than {config.minimum_replicates} successful
replicates; otherwise `unstable` means population-CV strictly greater than {config.max_cv_percent}%,
and `stable` means CV is at or below that threshold. Raw replicates and outliers are preserved;
none are automatically deleted.

This report is not a shared-runner performance gate. Shared CI may validate correctness, but it
must not enforce performance thresholds.
"""


def _summary_row(summary: BenchmarkV2Summary) -> str:
    """Render one Markdown result row from unrounded summary values."""
    fields = (
        summary.case_name,
        f"{summary.success_count}/{summary.replicate_count}",
        str(summary.median_step_time_ms),
        str(summary.coefficient_of_variation_percent),
        summary.stability,
    )
    return "| " + " | ".join(fields) + " |"


def _artifact_entry(run_directory: Path, artifact_path: Path) -> ArtifactManifestEntry:
    """Hash one output only after its atomic replacement is complete."""
    relative_path = artifact_path.relative_to(run_directory).as_posix()
    content = artifact_path.read_bytes()
    return ArtifactManifestEntry(
        path=relative_path,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _case_identity_documents(
    tasks: tuple[_BenchmarkTaskLike, ...],
) -> tuple[dict[str, str], ...]:
    """List each case identity once in first execution-order appearance."""
    documents: list[dict[str, str]] = []
    observed: set[str] = set()
    for task in tasks:
        if task.case_identity not in observed:
            observed.add(task.case_identity)
            documents.append({"case_name": task.case.name, "case_identity": task.case_identity})
    return tuple(documents)


def _manifest_document(manifest: RunManifest) -> dict[str, JsonValue]:
    """Serialize the strict manifest schema while deliberately omitting its own hash."""
    return cast(
        "dict[str, JsonValue]",
        {
            "schema_version": manifest.schema_version,
            "run_id": manifest.run_id,
            "status": manifest.status,
            "started_at_utc": manifest.started_at_utc,
            "ended_at_utc": manifest.ended_at_utc,
            "config_sha256": manifest.config_sha256,
            "git": manifest.git,
            "expected_task_count": manifest.expected_task_count,
            "completed_task_count": manifest.completed_task_count,
            "successful_task_count": manifest.successful_task_count,
            "failed_task_count": manifest.failed_task_count,
            "case_identities": list(manifest.case_identities),
            "artifacts": [asdict(entry) for entry in manifest.artifacts],
        },
    )


def _validate_complete_task_identities(
    status: RunStatus,
    tasks: tuple[_BenchmarkTaskLike, ...],
    raw_replicates: tuple[_RawReplicateLike, ...],
) -> None:
    """Require a complete status to bind each expected task identity exactly once."""
    if status != "complete":
        return
    expected = {(task.case_identity, task.replicate_index) for task in tasks}
    observed = {(record.case_identity, record.replicate_index) for record in raw_replicates}
    if (
        len(expected) != len(tasks)
        or len(raw_replicates) != len(tasks)
        or len(observed) != len(raw_replicates)
        or observed != expected
        or any(record.status != "ok" for record in raw_replicates)
    ):
        msg = "a complete run requires exactly one successful raw record for every task identity"
        raise ValueError(msg)


def write_run_artifacts(  # noqa: PLR0913
    *,
    config: BenchmarkV2Config,
    run_directory: Path,
    run_id: str,
    status: RunStatus,
    tasks: tuple[_BenchmarkTaskLike, ...],
    raw_replicates: tuple[_RawReplicateLike, ...],
    started_at_utc: str,
    ended_at_utc: str,
    environment_snapshot: dict[str, JsonValue] | None = None,
) -> RunArtifactPaths:
    """Atomically finalize a unique run directory from all preserved raw evidence."""
    if not run_directory.is_dir():
        msg = f"run directory does not exist: {run_directory}"
        raise FileNotFoundError(msg)
    _validate_complete_task_identities(status, tasks, raw_replicates)
    grouped_records: dict[str, list[_RawReplicateLike]] = {}
    for record in raw_replicates:
        grouped_records.setdefault(record.case_identity, []).append(record)
    case_identities_in_order = tuple(dict.fromkeys(task.case_identity for task in tasks))
    summaries = tuple(
        summarize_replicates(
            grouped_records[case_identity],
            minimum_replicates=config.minimum_replicates,
            max_cv_percent=config.max_cv_percent,
        )
        for case_identity in case_identities_in_order
        if case_identity in grouped_records
    )
    return _write_final_artifacts(
        config=config,
        run_directory=run_directory,
        run_id=run_id,
        status=status,
        tasks=tasks,
        raw_replicates=raw_replicates,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        summaries=summaries,
        environment_snapshot=environment_snapshot or capture_run_environment(config),
    )


def _write_final_artifacts(  # noqa: PLR0913
    *,
    config: BenchmarkV2Config,
    run_directory: Path,
    run_id: str,
    status: RunStatus,
    tasks: tuple[_BenchmarkTaskLike, ...],
    raw_replicates: tuple[_RawReplicateLike, ...],
    started_at_utc: str,
    ended_at_utc: str,
    summaries: tuple[BenchmarkV2Summary, ...],
    environment_snapshot: dict[str, JsonValue],
) -> RunArtifactPaths:
    """Write bound non-manifest artifacts first, then write the self-excluded manifest."""
    environment_path = run_directory / "environment.json"
    resolved_config_path = run_directory / "resolved_config.yaml"
    raw_replicates_path = run_directory / "raw_replicates.jsonl"
    summary_csv_path = run_directory / "summary.csv"
    summary_markdown_path = run_directory / "summary.md"
    execution_order_path = run_directory / "execution_order.json"
    run_manifest_path = run_directory / "run_manifest.json"
    _atomic_write_bytes(
        environment_path,
        _json_bytes(
            _environment_document(
                config=config,
                run_id=run_id,
                status=status,
                raw_replicates=raw_replicates,
                environment_snapshot=environment_snapshot,
            )
        ),
    )
    _atomic_write_bytes(resolved_config_path, _yaml_bytes(resolved_config_document(config)))
    _atomic_write_bytes(
        raw_replicates_path,
        _jsonl_bytes(_raw_replicate_document(record) for record in raw_replicates),
    )
    _atomic_write_bytes(summary_csv_path, _csv_bytes(summaries))
    _atomic_write_bytes(
        summary_markdown_path,
        _summary_markdown(run_id=run_id, status=status, config=config, summaries=summaries).encode(
            "utf-8"
        ),
    )
    _atomic_write_bytes(
        execution_order_path,
        _json_bytes(_execution_order_document(tasks, raw_replicates)),
    )
    entries = tuple(
        _artifact_entry(run_directory, artifact_path)
        for artifact_path in (
            environment_path,
            resolved_config_path,
            raw_replicates_path,
            summary_csv_path,
            summary_markdown_path,
            execution_order_path,
        )
    )
    manifest = RunManifest(
        schema_version=2,
        run_id=run_id,
        status=status,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        config_sha256=resolved_config_sha256(config),
        git=cast("dict[str, JsonValue]", environment_snapshot["git"]),
        expected_task_count=len(tasks),
        completed_task_count=len(raw_replicates),
        successful_task_count=sum(record.status == "ok" for record in raw_replicates),
        failed_task_count=sum(record.status == "error" for record in raw_replicates),
        case_identities=_case_identity_documents(tasks),
        artifacts=entries,
    )
    _atomic_write_bytes(run_manifest_path, _json_bytes(_manifest_document(manifest)))
    return RunArtifactPaths(
        run_directory=run_directory,
        run_manifest_path=run_manifest_path,
        environment_path=environment_path,
        resolved_config_path=resolved_config_path,
        raw_replicates_path=raw_replicates_path,
        summary_csv_path=summary_csv_path,
        summary_markdown_path=summary_markdown_path,
        execution_order_path=execution_order_path,
    )


def _require_mapping(value: object, context: str) -> dict[str, object]:
    """Validate one strict JSON object with text keys."""
    if not isinstance(value, dict):
        msg = f"{context} must be an object with string keys"
        raise TypeError(msg)
    mapping = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        msg = f"{context} must be an object with string keys"
        raise ValueError(msg)
    return cast("dict[str, object]", mapping)


def _require_sha256(value: object, context: str) -> str:
    """Validate a lowercase hexadecimal SHA-256 identity before it is compared or trusted."""
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in _SHA256_HEX_DIGITS for character in value)
    ):
        msg = f"{context} must be a lowercase SHA-256 digest"
        raise ValueError(msg)
    return value


def _require_git_identity(value: object, context: str) -> dict[str, JsonValue]:
    """Validate the complete Git identity schema emitted in immutable environment evidence."""
    git = _require_mapping(value, context)
    if frozenset(git) != _GIT_IDENTITY_KEYS:
        msg = f"{context} has an invalid field set"
        raise ValueError(msg)
    commit_sha, branch, dirty = git["commit_sha"], git["branch"], git["dirty"]
    if commit_sha is not None and (
        not isinstance(commit_sha, str)
        or len(commit_sha) not in _GIT_COMMIT_SHA_LENGTHS
        or any(character not in _SHA256_HEX_DIGITS for character in commit_sha)
    ):
        msg = f"{context} has an invalid commit_sha"
        raise ValueError(msg)
    if branch is not None and not isinstance(branch, str):
        msg = f"{context} has an invalid branch"
        raise ValueError(msg)
    if dirty is not None and not isinstance(dirty, bool):
        msg = f"{context} has an invalid dirty flag"
        raise ValueError(msg)
    return cast("dict[str, JsonValue]", git)


def _regular_file_metadata(stat_result: os.stat_result, context: str) -> tuple[int, int, int, int]:
    """Return stable identity metadata while rejecting symlinks and every non-regular file."""
    if not stat.S_ISREG(stat_result.st_mode):
        msg = f"{context} must be a regular file"
        raise ValueError(msg)
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _snapshot_regular_file(path: Path) -> _FileSnapshot:
    """Read a regular file exactly once and reject replacements or mutations around that read."""
    before_path = _regular_file_metadata(path.lstat(), str(path))
    with path.open("rb") as stream:
        before_handle = _regular_file_metadata(os.fstat(stream.fileno()), str(path))
        if before_path != before_handle:
            msg = f"file changed before snapshot read: {path.name}"
            raise ValueError(msg)
        content = stream.read()
        after_handle = _regular_file_metadata(os.fstat(stream.fileno()), str(path))
    after_path = _regular_file_metadata(path.lstat(), str(path))
    if before_path != after_handle or before_path != after_path or len(content) != before_path[2]:
        msg = f"file changed during snapshot read: {path.name}"
        raise ValueError(msg)
    return _FileSnapshot(content=content, metadata=before_path)


def _require_finalized_run_entries(run_directory: Path) -> None:
    """Require the exact finalized seven-entry package and no lifecycle or temporary residue."""
    if frozenset(entry.name for entry in run_directory.iterdir()) != _FINALIZED_RUN_ENTRIES:
        msg = "finalized run directory has unexpected filesystem entries"
        raise ValueError(msg)


def _snapshot_finalized_run(run_directory: Path) -> dict[str, _FileSnapshot]:
    """Take a stable single-read snapshot of every entry in one finalized run package."""
    _require_finalized_run_entries(run_directory)
    snapshots = {
        entry_name: _snapshot_regular_file(run_directory / entry_name)
        for entry_name in _FINALIZED_RUN_ENTRIES
    }
    _require_finalized_run_entries(run_directory)
    for entry_name, snapshot in snapshots.items():
        if (
            _regular_file_metadata((run_directory / entry_name).lstat(), entry_name)
            != snapshot.metadata
        ):
            msg = f"file changed after snapshot read: {entry_name}"
            raise ValueError(msg)
    return snapshots


def load_run_manifest(path: Path) -> RunManifest:  # noqa: C901, PLR0912, PLR0915
    """Load a strict self-excluded manifest without trusting malformed artifact metadata."""
    if path.name != "run_manifest.json":
        msg = "manifest path must be the finalized run_manifest.json entry"
        raise ValueError(msg)
    resolved_run_directory = path.parent.resolve()
    snapshots = _snapshot_finalized_run(resolved_run_directory)
    raw_document = cast(
        "object", json.loads(snapshots["run_manifest.json"].content.decode("utf-8"))
    )
    document = _require_mapping(raw_document, "run manifest")
    if frozenset(document) != _MANIFEST_KEYS:
        msg = "run manifest has an invalid field set"
        raise ValueError(msg)
    status = document["status"]
    if status not in {"complete", "partial", "failed"}:
        msg = "run manifest has an invalid status"
        raise ValueError(msg)
    raw_artifacts = document["artifacts"]
    if not isinstance(raw_artifacts, list):
        msg = "run manifest artifacts must be a list"
        raise TypeError(msg)
    artifacts: list[ArtifactManifestEntry] = []
    for raw_entry in cast("list[object]", raw_artifacts):
        entry = _require_mapping(raw_entry, "artifact entry")
        if frozenset(entry) != _ARTIFACT_ENTRY_KEYS:
            msg = "artifact entry has an invalid field set"
            raise ValueError(msg)
        raw_path, size_bytes, sha256 = entry["path"], entry["size_bytes"], entry["sha256"]
        artifact_path = Path(raw_path) if isinstance(raw_path, str) else None
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or artifact_path is None
            or artifact_path.drive
            or artifact_path.is_absolute()
            or "\\" in raw_path
            or ".." in artifact_path.parts
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or len(sha256) != _SHA256_HEX_LENGTH
            or any(character not in _SHA256_HEX_DIGITS for character in sha256)
        ):
            msg = "artifact entry has invalid values"
            raise ValueError(msg)
        artifacts.append(ArtifactManifestEntry(raw_path, size_bytes, sha256))
    artifact_paths = tuple(entry.path for entry in artifacts)
    if len(artifact_paths) != len(set(artifact_paths)):
        msg = "run manifest artifact paths must be unique"
        raise ValueError(msg)
    if frozenset(artifact_paths) != _REQUIRED_BOUND_ARTIFACTS:
        msg = "run manifest artifact paths must be the required self-excluded set"
        raise ValueError(msg)
    for artifact in artifacts:
        bound_path = (resolved_run_directory / artifact.path).resolve()
        if not bound_path.is_relative_to(resolved_run_directory):
            msg = f"bound artifact escapes run directory: {artifact.path}"
            raise ValueError(msg)
        content = snapshots[artifact.path].content
        if (
            len(content) != artifact.size_bytes
            or hashlib.sha256(content).hexdigest() != artifact.sha256
        ):
            msg = f"bound artifact hash or size mismatch: {artifact.path}"
            raise ValueError(msg)
    raw_case_identities = document["case_identities"]
    if not isinstance(raw_case_identities, list):
        msg = "run manifest case_identities must be a list"
        raise TypeError(msg)
    case_identities: list[dict[str, str]] = []
    for raw_case in cast("list[object]", raw_case_identities):
        case = _require_mapping(raw_case, "case identity")
        if frozenset(case) != frozenset({"case_name", "case_identity"}) or any(
            not isinstance(value, str) or not value for value in case.values()
        ):
            msg = "run manifest has an invalid case identity"
            raise ValueError(msg)
        case_identities.append(cast("dict[str, str]", case))
    integer_fields = (
        "schema_version",
        "expected_task_count",
        "completed_task_count",
        "successful_task_count",
        "failed_task_count",
    )
    if any(
        not isinstance(document[field], int) or isinstance(document[field], bool)
        for field in integer_fields
    ):
        msg = "run manifest has an invalid integer field"
        raise ValueError(msg)
    string_fields = ("run_id", "started_at_utc", "ended_at_utc")
    if any(not isinstance(document[field], str) or not document[field] for field in string_fields):
        msg = "run manifest has an invalid string field"
        raise ValueError(msg)
    config_sha256 = _require_sha256(document["config_sha256"], "run manifest config_sha256")
    try:
        resolved_config = load_resolved_benchmark_v2_config(
            snapshots["resolved_config.yaml"].content,
            resolved_run_directory / "resolved_config.yaml",
        )
    except ValueError as error:
        msg = f"resolved config is invalid: {error}"
        raise ValueError(msg) from error
    recomputed_config_sha256 = resolved_config_sha256(resolved_config)
    if config_sha256 != recomputed_config_sha256:
        msg = "run manifest config_sha256 does not match snapshotted resolved config"
        raise ValueError(msg)
    expected_case_identities = {
        (case.name, case_identity(resolved_config, case)) for case in resolved_config.cases
    }
    observed_case_identities = {
        (case["case_name"], case["case_identity"]) for case in case_identities
    }
    if (
        len(case_identities) != len(observed_case_identities)
        or observed_case_identities != expected_case_identities
    ):
        msg = "run manifest case identities do not match snapshotted resolved config"
        raise ValueError(msg)
    if path.parent.name != document["run_id"]:
        msg = "run manifest run_id does not match its run directory"
        raise ValueError(msg)
    git = _require_git_identity(document["git"], "run manifest git")
    raw_environment = cast(
        "object",
        json.loads(snapshots["environment.json"].content.decode("utf-8")),
    )
    environment = _require_mapping(raw_environment, "environment artifact")
    if frozenset(environment) != _ENVIRONMENT_DOCUMENT_KEYS:
        msg = "environment artifact has an invalid field set"
        raise ValueError(msg)
    environment_schema_version = environment["schema_version"]
    if (
        not isinstance(environment_schema_version, int)
        or isinstance(environment_schema_version, bool)
        or environment_schema_version != _ENVIRONMENT_SCHEMA_VERSION
    ):
        msg = "environment artifact has an invalid schema_version"
        raise ValueError(msg)
    environment_status = environment["run_status"]
    if environment_status not in {"complete", "partial", "failed"}:
        msg = "environment artifact has an invalid run_status"
        raise ValueError(msg)
    if environment_status != status:
        msg = "environment artifact run_status does not match the manifest"
        raise ValueError(msg)
    if not isinstance(environment["worker_environments"], list):
        msg = "environment artifact worker_environments must be a list"
        raise TypeError(msg)
    if environment.get("run_id") != document["run_id"]:
        msg = "environment artifact run_id does not match the manifest"
        raise ValueError(msg)
    environment_config_sha256 = _require_sha256(
        environment["config_sha256"], "environment artifact config_sha256"
    )
    if environment_config_sha256 != config_sha256:
        msg = "environment artifact config_sha256 does not match the manifest"
        raise ValueError(msg)
    if environment_config_sha256 != recomputed_config_sha256:
        msg = "environment artifact config_sha256 does not match snapshotted resolved config"
        raise ValueError(msg)
    run_environment = _require_mapping(
        environment.get("run_environment"), "environment run_environment"
    )
    environment_git = _require_git_identity(run_environment.get("git"), "environment artifact git")
    if environment_git != git:
        msg = "environment artifact git identity does not match the manifest"
        raise ValueError(msg)
    return RunManifest(
        schema_version=cast("int", document["schema_version"]),
        run_id=cast("str", document["run_id"]),
        status=cast("RunStatus", status),
        started_at_utc=cast("str", document["started_at_utc"]),
        ended_at_utc=cast("str", document["ended_at_utc"]),
        config_sha256=config_sha256,
        git=git,
        expected_task_count=cast("int", document["expected_task_count"]),
        completed_task_count=cast("int", document["completed_task_count"]),
        successful_task_count=cast("int", document["successful_task_count"]),
        failed_task_count=cast("int", document["failed_task_count"]),
        case_identities=tuple(case_identities),
        artifacts=tuple(artifacts),
    )
