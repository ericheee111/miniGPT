"""Record immutable environment and process segments for reference training."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

import numpy as np
import psutil
import torch
from typing_extensions import override

from minigpt.checkpoint import (
    DatasetFingerprints,
    compute_dataset_fingerprints,
    load_checkpoint_metadata,
)

if TYPE_CHECKING:
    from minigpt.settings import ExperimentConfig

_SCHEMA_VERSION: Final = 1
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_UNKNOWN_CPU: Final = "unknown"
_POWERSHELL_CPU_COMMAND: Final = (
    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()"
)
_STATUS_VALUES: Final = frozenset({"running", "completed", "failed"})
_GIT_UNAVAILABLE_REASON: Final = "Git executable is unavailable"
_PREVIOUS_SEGMENT_REASON: Final = "previous segment did not complete"
_RESUME_REQUIRED_REASON: Final = "subsequent segment requires a resume checkpoint"
_COMMIT_CHANGED_REASON: Final = "Git commit changed"
_CONFIG_PATH_CHANGED_REASON: Final = "config path changed"
_CONFIG_HASH_CHANGED_REASON: Final = "config SHA-256 changed"
_ENVIRONMENT_CHANGED_REASON: Final = "software or CPU environment changed"
_RESUME_CHECKPOINT_REASON: Final = "resume checkpoint is not the previous segment output"
_FIRST_RESUME_REASON: Final = "first segment cannot be a resume"
_CURRENT_SEGMENT_REASON: Final = "segment is not the current running invocation"
_COMPLETED_STEP_REASON: Final = "checkpoint completed step differs from training result"
_CHECKPOINT_DATA_REASON: Final = "checkpoint dataset fingerprints changed"
_RESOLVED_CONFIG_REASON: Final = "resolved config SHA-256 changed"

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
SegmentStatus: TypeAlias = Literal["running", "completed", "failed"]


@dataclass(slots=True)
class DirtyReferenceRunError(RuntimeError):
    """Reject reference training from an uncommitted repository."""

    repository_root: Path

    @override
    def __str__(self) -> str:
        """Render the repository whose state is not reproducible."""
        return f"reference training requires a clean Git tree: {self.repository_root}"


@dataclass(slots=True)
class IncompatibleRunProvenanceError(ValueError):
    """Reject a segment that does not belong to the recorded experiment."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the mismatched experiment identity."""
        return f"incompatible reference run provenance: {self.reason}"


@dataclass(slots=True)
class RunProvenanceFormatError(ValueError):
    """Report an invalid persisted provenance document."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the invalid sidecar path and reason."""
        return f"invalid run provenance {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class GitIdentity:
    """Identify the exact clean source revision used by a run."""

    commit_sha: str
    branch: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Capture automatically discoverable software and CPU environment facts."""

    operating_system: str
    machine: str
    python_version: str
    pytorch_version: str
    numpy_version: str
    cpu_name: str
    physical_cores: int | None
    logical_cores: int | None
    torch_num_threads: int
    cuda_available: bool


@dataclass(frozen=True, slots=True)
class RunSegment:
    """Describe one actual training-process invocation."""

    started_at_utc: str
    ended_at_utc: str | None
    status: SegmentStatus
    argv: tuple[str, ...]
    run_until_step: int | None
    resume_checkpoint_sha256: str | None
    checkpoint_sha256: str | None
    final_completed_step: int | None


@dataclass(frozen=True, slots=True)
class RunInvocation:
    """Hold process-specific values outside the immutable experiment identity."""

    argv: tuple[str, ...]
    run_until_step: int | None
    resume_path: Path | None


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Bind all process segments to one immutable experiment identity."""

    schema_version: int
    experiment_name: str
    config_path: str
    source_config_sha256: str
    resolved_config_sha256: str | None
    git: GitIdentity
    environment: EnvironmentSnapshot
    dataset_fingerprints: DatasetFingerprints
    started_at_utc: str
    ended_at_utc: str | None
    segments: tuple[RunSegment, ...]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _git_executable() -> str:
    executable = shutil.which("git")
    if executable is None:
        raise IncompatibleRunProvenanceError(_GIT_UNAVAILABLE_REASON)
    return executable


def _run_git(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        [_git_executable(), "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def find_repository_root(path: Path) -> Path:
    """Find the Git worktree containing a reference configuration."""
    candidate = path if path.is_dir() else path.parent
    root = _run_git(candidate, "rev-parse", "--show-toplevel")
    return Path(root).resolve()


def _read_git_identity(repository_root: Path) -> GitIdentity:
    commit_sha = _run_git(repository_root, "rev-parse", "HEAD")
    branch = _run_git(repository_root, "branch", "--show-current")
    dirty = bool(_run_git(repository_root, "status", "--porcelain"))
    return GitIdentity(
        commit_sha=commit_sha,
        branch=branch or "<detached>",
        dirty=dirty,
    )


def _windows_cpu_name() -> str | None:
    if sys.platform != "win32":
        return None
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return None
    completed = subprocess.run(  # noqa: S603
        [powershell, "-NoProfile", "-Command", _POWERSHELL_CPU_COMMAND],
        check=False,
        capture_output=True,
        text=True,
    )
    name = completed.stdout.strip()
    return name or None


def _linux_cpu_name() -> str | None:
    if not sys.platform.startswith("linux"):
        return None
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "model name":
            name = value.strip()
            return name or None
    return None


def _cpu_name() -> str:
    return _windows_cpu_name() or _linux_cpu_name() or platform.processor().strip() or _UNKNOWN_CPU


def _capture_environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        operating_system=platform.platform(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        pytorch_version=str(torch.__version__),
        numpy_version=np.__version__,
        cpu_name=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        torch_num_threads=torch.get_num_threads(),
        cuda_available=torch.cuda.is_available(),
    )


def _relative_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as error:
        reason = f"config path is outside repository: {path}"
        raise IncompatibleRunProvenanceError(reason) from error


def _fingerprints_document(fingerprints: DatasetFingerprints) -> dict[str, JsonValue]:
    return {
        "tokenizer_sha256": fingerprints.tokenizer_sha256,
        "train_sha256": fingerprints.train_sha256,
        "val_sha256": fingerprints.val_sha256,
    }


def _environment_document(environment: EnvironmentSnapshot) -> dict[str, JsonValue]:
    return {
        "operating_system": environment.operating_system,
        "machine": environment.machine,
        "python_version": environment.python_version,
        "pytorch_version": environment.pytorch_version,
        "numpy_version": environment.numpy_version,
        "cpu_name": environment.cpu_name,
        "physical_cores": environment.physical_cores,
        "logical_cores": environment.logical_cores,
        "torch_num_threads": environment.torch_num_threads,
        "cuda_available": environment.cuda_available,
    }


def _segment_document(segment: RunSegment) -> dict[str, JsonValue]:
    return {
        "started_at_utc": segment.started_at_utc,
        "ended_at_utc": segment.ended_at_utc,
        "status": segment.status,
        "argv": list(segment.argv),
        "run_until_step": segment.run_until_step,
        "resume_checkpoint_sha256": segment.resume_checkpoint_sha256,
        "checkpoint_sha256": segment.checkpoint_sha256,
        "final_completed_step": segment.final_completed_step,
    }


def _provenance_document(provenance: RunProvenance) -> dict[str, JsonValue]:
    return {
        "schema_version": provenance.schema_version,
        "experiment_name": provenance.experiment_name,
        "config_path": provenance.config_path,
        "source_config_sha256": provenance.source_config_sha256,
        "resolved_config_sha256": provenance.resolved_config_sha256,
        "git": {
            "commit_sha": provenance.git.commit_sha,
            "branch": provenance.git.branch,
            "dirty": provenance.git.dirty,
        },
        "environment": _environment_document(provenance.environment),
        "dataset_fingerprints": _fingerprints_document(provenance.dataset_fingerprints),
        "started_at_utc": provenance.started_at_utc,
        "ended_at_utc": provenance.ended_at_utc,
        "segments": [_segment_document(segment) for segment in provenance.segments],
    }


def _write_run_provenance(path: Path, provenance: RunProvenance) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    _ = temporary_path.write_text(
        json.dumps(
            _provenance_document(provenance),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _ = temporary_path.replace(path)


def _mapping(value: JsonValue, path: Path, field: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise RunProvenanceFormatError(path, f"{field} must be an object")
    return value


def _string(document: dict[str, JsonValue], key: str, path: Path) -> str:
    value = document.get(key)
    if not isinstance(value, str):
        raise RunProvenanceFormatError(path, f"{key} must be a string")
    return value


def _optional_string(document: dict[str, JsonValue], key: str, path: Path) -> str | None:
    value = document.get(key)
    if value is not None and not isinstance(value, str):
        raise RunProvenanceFormatError(path, f"{key} must be a string or null")
    return value


def _integer(document: dict[str, JsonValue], key: str, path: Path) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunProvenanceFormatError(path, f"{key} must be an integer")
    return value


def _optional_integer(document: dict[str, JsonValue], key: str, path: Path) -> int | None:
    value = document.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise RunProvenanceFormatError(path, f"{key} must be an integer or null")
    return value


def _boolean(document: dict[str, JsonValue], key: str, path: Path) -> bool:
    value = document.get(key)
    if not isinstance(value, bool):
        raise RunProvenanceFormatError(path, f"{key} must be a boolean")
    return value


def _load_git(document: dict[str, JsonValue], path: Path) -> GitIdentity:
    git = _mapping(document.get("git"), path, "git")
    return GitIdentity(
        commit_sha=_string(git, "commit_sha", path),
        branch=_string(git, "branch", path),
        dirty=_boolean(git, "dirty", path),
    )


def _load_environment(document: dict[str, JsonValue], path: Path) -> EnvironmentSnapshot:
    environment = _mapping(document.get("environment"), path, "environment")
    return EnvironmentSnapshot(
        operating_system=_string(environment, "operating_system", path),
        machine=_string(environment, "machine", path),
        python_version=_string(environment, "python_version", path),
        pytorch_version=_string(environment, "pytorch_version", path),
        numpy_version=_string(environment, "numpy_version", path),
        cpu_name=_string(environment, "cpu_name", path),
        physical_cores=_optional_integer(environment, "physical_cores", path),
        logical_cores=_optional_integer(environment, "logical_cores", path),
        torch_num_threads=_integer(environment, "torch_num_threads", path),
        cuda_available=_boolean(environment, "cuda_available", path),
    )


def _load_fingerprints(document: dict[str, JsonValue], path: Path) -> DatasetFingerprints:
    fingerprints = _mapping(
        document.get("dataset_fingerprints"),
        path,
        "dataset_fingerprints",
    )
    return DatasetFingerprints(
        tokenizer_sha256=_string(fingerprints, "tokenizer_sha256", path),
        train_sha256=_string(fingerprints, "train_sha256", path),
        val_sha256=_string(fingerprints, "val_sha256", path),
    )


def _load_segment(value: JsonValue, path: Path) -> RunSegment:
    document = _mapping(value, path, "segment")
    raw_status = _string(document, "status", path)
    if raw_status not in _STATUS_VALUES:
        raise RunProvenanceFormatError(path, "segment status is unsupported")
    raw_argv = document.get("argv")
    if not isinstance(raw_argv, list) or not all(isinstance(item, str) for item in raw_argv):
        raise RunProvenanceFormatError(path, "segment argv must be a string array")
    argv = tuple(cast("str", item) for item in raw_argv)
    return RunSegment(
        started_at_utc=_string(document, "started_at_utc", path),
        ended_at_utc=_optional_string(document, "ended_at_utc", path),
        status=cast("SegmentStatus", raw_status),
        argv=argv,
        run_until_step=_optional_integer(document, "run_until_step", path),
        resume_checkpoint_sha256=_optional_string(
            document,
            "resume_checkpoint_sha256",
            path,
        ),
        checkpoint_sha256=_optional_string(document, "checkpoint_sha256", path),
        final_completed_step=_optional_integer(document, "final_completed_step", path),
    )


def load_run_provenance(path: Path) -> RunProvenance:
    """Load and validate a persisted reference-run journal."""
    try:
        raw_document = cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise RunProvenanceFormatError(path, str(error)) from error
    document = _mapping(raw_document, path, "top-level value")
    schema_version = _integer(document, "schema_version", path)
    if schema_version != _SCHEMA_VERSION:
        raise RunProvenanceFormatError(path, "unsupported schema version")
    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list):
        raise RunProvenanceFormatError(path, "segments must be an array")
    segments = tuple(_load_segment(item, path) for item in raw_segments)
    if not segments:
        raise RunProvenanceFormatError(path, "segments must not be empty")
    return RunProvenance(
        schema_version=schema_version,
        experiment_name=_string(document, "experiment_name", path),
        config_path=_string(document, "config_path", path),
        source_config_sha256=_string(document, "source_config_sha256", path),
        resolved_config_sha256=_optional_string(document, "resolved_config_sha256", path),
        git=_load_git(document, path),
        environment=_load_environment(document, path),
        dataset_fingerprints=_load_fingerprints(document, path),
        started_at_utc=_string(document, "started_at_utc", path),
        ended_at_utc=_optional_string(document, "ended_at_utc", path),
        segments=segments,
    )


def _environment_identity(environment: EnvironmentSnapshot) -> tuple[str | int | bool | None, ...]:
    return (
        environment.operating_system,
        environment.machine,
        environment.python_version,
        environment.pytorch_version,
        environment.numpy_version,
        environment.cpu_name,
        environment.physical_cores,
        environment.logical_cores,
        environment.cuda_available,
    )


@dataclass(frozen=True, slots=True)
class _CurrentRunIdentity:
    git: GitIdentity
    config_path: str
    source_config_sha256: str
    fingerprints: DatasetFingerprints
    environment: EnvironmentSnapshot


def _validate_existing_identity(
    provenance: RunProvenance,
    current: _CurrentRunIdentity,
    resume_path: Path | None,
) -> str:
    if any(segment.status != "completed" for segment in provenance.segments):
        raise IncompatibleRunProvenanceError(_PREVIOUS_SEGMENT_REASON)
    if resume_path is None:
        raise IncompatibleRunProvenanceError(_RESUME_REQUIRED_REASON)
    if provenance.git.commit_sha != current.git.commit_sha:
        raise IncompatibleRunProvenanceError(_COMMIT_CHANGED_REASON)
    if provenance.config_path != current.config_path:
        raise IncompatibleRunProvenanceError(_CONFIG_PATH_CHANGED_REASON)
    if provenance.source_config_sha256 != current.source_config_sha256:
        raise IncompatibleRunProvenanceError(_CONFIG_HASH_CHANGED_REASON)
    if provenance.dataset_fingerprints != current.fingerprints:
        fields = (
            (
                "tokenizer",
                provenance.dataset_fingerprints.tokenizer_sha256,
                current.fingerprints.tokenizer_sha256,
            ),
            (
                "train",
                provenance.dataset_fingerprints.train_sha256,
                current.fingerprints.train_sha256,
            ),
            (
                "validation",
                provenance.dataset_fingerprints.val_sha256,
                current.fingerprints.val_sha256,
            ),
        )
        changed = next(name for name, expected, actual in fields if expected != actual)
        reason = f"{changed} data fingerprint changed"
        raise IncompatibleRunProvenanceError(reason)
    if _environment_identity(provenance.environment) != _environment_identity(current.environment):
        raise IncompatibleRunProvenanceError(_ENVIRONMENT_CHANGED_REASON)
    resume_sha256 = _sha256_file(resume_path)
    previous_checkpoint_sha256 = provenance.segments[-1].checkpoint_sha256
    if resume_sha256 != previous_checkpoint_sha256:
        raise IncompatibleRunProvenanceError(_RESUME_CHECKPOINT_REASON)
    return resume_sha256


def begin_run_segment(
    path: Path,
    *,
    config_path: Path,
    config: ExperimentConfig,
    invocation: RunInvocation,
) -> RunSegment:
    """Validate clean provenance and atomically record a running segment."""
    repository_root = find_repository_root(config_path)
    git = _read_git_identity(repository_root)
    if git.dirty:
        raise DirtyReferenceRunError(repository_root)
    relative_config_path = _relative_path(config_path, repository_root)
    source_config_sha256 = _sha256_file(config_path)
    fingerprints = compute_dataset_fingerprints(config.data)
    environment = _capture_environment()
    started_at_utc = _utc_now()
    current = _CurrentRunIdentity(
        git=git,
        config_path=relative_config_path,
        source_config_sha256=source_config_sha256,
        fingerprints=fingerprints,
        environment=environment,
    )
    resume_checkpoint_sha256 = None
    if path.exists():
        provenance = load_run_provenance(path)
        resume_checkpoint_sha256 = _validate_existing_identity(
            provenance,
            current,
            invocation.resume_path,
        )
    else:
        if invocation.resume_path is not None:
            raise IncompatibleRunProvenanceError(_FIRST_RESUME_REASON)
        provenance = RunProvenance(
            schema_version=_SCHEMA_VERSION,
            experiment_name=config_path.stem,
            config_path=relative_config_path,
            source_config_sha256=source_config_sha256,
            resolved_config_sha256=None,
            git=git,
            environment=environment,
            dataset_fingerprints=fingerprints,
            started_at_utc=started_at_utc,
            ended_at_utc=None,
            segments=(),
        )
    segment = RunSegment(
        started_at_utc=started_at_utc,
        ended_at_utc=None,
        status="running",
        argv=invocation.argv,
        run_until_step=invocation.run_until_step,
        resume_checkpoint_sha256=resume_checkpoint_sha256,
        checkpoint_sha256=None,
        final_completed_step=None,
    )
    _write_run_provenance(
        path,
        replace(
            provenance,
            ended_at_utc=None,
            segments=(*provenance.segments, segment),
        ),
    )
    return segment


def _require_running_last_segment(
    provenance: RunProvenance,
    segment: RunSegment,
) -> None:
    if provenance.segments[-1] != segment or segment.status != "running":
        raise IncompatibleRunProvenanceError(_CURRENT_SEGMENT_REASON)


def complete_run_segment(
    path: Path,
    *,
    segment: RunSegment,
    checkpoint_path: Path,
    final_step: int,
) -> None:
    """Replace the running segment with validated completion metadata."""
    provenance = load_run_provenance(path)
    _require_running_last_segment(provenance, segment)
    metadata = load_checkpoint_metadata(checkpoint_path)
    if metadata.completed_step != final_step:
        raise IncompatibleRunProvenanceError(_COMPLETED_STEP_REASON)
    if metadata.dataset_fingerprints != provenance.dataset_fingerprints:
        raise IncompatibleRunProvenanceError(_CHECKPOINT_DATA_REASON)
    resolved_config_sha256 = _sha256_text(metadata.config.to_yaml())
    if (
        provenance.resolved_config_sha256 is not None
        and provenance.resolved_config_sha256 != resolved_config_sha256
    ):
        raise IncompatibleRunProvenanceError(_RESOLVED_CONFIG_REASON)
    ended_at_utc = _utc_now()
    completed = replace(
        segment,
        ended_at_utc=ended_at_utc,
        status="completed",
        checkpoint_sha256=_sha256_file(checkpoint_path),
        final_completed_step=final_step,
    )
    _write_run_provenance(
        path,
        replace(
            provenance,
            resolved_config_sha256=resolved_config_sha256,
            environment=replace(
                provenance.environment,
                torch_num_threads=torch.get_num_threads(),
            ),
            ended_at_utc=ended_at_utc,
            segments=(*provenance.segments[:-1], completed),
        ),
    )


def fail_run_segment(path: Path, *, segment: RunSegment) -> None:
    """Record an ended failed segment without claiming a completed step."""
    provenance = load_run_provenance(path)
    _require_running_last_segment(provenance, segment)
    ended_at_utc = _utc_now()
    failed = replace(
        segment,
        ended_at_utc=ended_at_utc,
        status="failed",
    )
    _write_run_provenance(
        path,
        replace(
            provenance,
            ended_at_utc=ended_at_utc,
            segments=(*provenance.segments[:-1], failed),
        ),
    )
