"""Verify miniGPT packaging, configuration, evidence, ancestry, and release invariants."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from minigpt._version import __version__
from minigpt.evidence_registry import EvidencePackage, evidence_registry
from minigpt.model import GPT
from minigpt.paged_kv_cache import KVCacheBackend
from minigpt.serving import APCPrefillStrategy
from minigpt.serving_runtime import ServingExecutorName, ServingRuntimeConfig, build_serving_runtime
from minigpt.serving_simulator import load_simulator_config, run_simulation
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

_DISTRIBUTION_NAME = "minitrain-gpt"
_SCHEMA_VERSION = 1


class DoctorMode(StrEnum):
    """Select the depth of deterministic repository verification."""

    QUICK = "quick"
    CI = "ci"
    RELEASE = "release"


class CheckStatus(StrEnum):
    """Describe one project-doctor check result."""

    PASS = "pass"  # noqa: S105 - verification status, not a credential
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Record one stable, path-independent verification result."""

    name: str
    status: CheckStatus
    detail: str

    def to_document(self) -> dict[str, object]:
        """Render the result without machine-specific paths or timings."""
        return {"name": self.name, "status": self.status.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """Summarize all checks for one deterministic verification mode."""

    mode: DoctorMode
    project_version: str
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """Return whether every requested check passed."""
        return all(item.status is CheckStatus.PASS for item in self.checks)

    def to_document(self) -> dict[str, object]:
        """Return a stable JSON-compatible report."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "mode": self.mode.value,
            "project_version": self.project_version,
            "repository_root": ".",
            "passed": self.passed,
            "checks": [item.to_document() for item in self.checks],
        }


class ProjectDoctorError(RuntimeError):
    """Report a malformed invocation or unavailable repository contract."""


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.FAIL, detail=detail)


def _pass(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.PASS, detail=detail)


def _installed_version() -> str:
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError as error:
        reason = "minitrain-gpt distribution metadata is unavailable; install the project first"
        raise ProjectDoctorError(reason) from error


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - repository-owned fixed git arguments
        ["git", "-C", str(root), *arguments],  # noqa: S607 - git resolved by development toolchain
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _check_version() -> CheckResult:
    installed = _installed_version()
    if installed != __version__:
        return _fail(
            "package-version",
            f"metadata version {installed!r} differs from source version {__version__!r}",
        )
    return _pass("package-version", f"source and installed metadata equal {__version__}")


def _check_docs(root: Path) -> CheckResult:
    required = {
        root / "README.md": ("PROJECT_OVERVIEW.md", "Stage 19", "Stage 20"),
        root / "AGENTS.md": ("Stage 19", "Stage 20"),
        root / "docs" / "PROJECT_OVERVIEW.md": ("Stage 19", "Stage 20"),
    }
    missing: list[str] = []
    for path, needles in required.items():
        if not path.is_file():
            missing.append(path.relative_to(root).as_posix())
            continue
        text = path.read_text(encoding="utf-8")
        missing.extend(
            f"{path.relative_to(root).as_posix()}::{needle}"
            for needle in needles
            if needle not in text
        )
    if missing:
        return _fail("release-documentation", "missing: " + ", ".join(sorted(missing)))
    return _pass("release-documentation", "README, AGENTS, and project overview cover Stage 19-20")


def _source_commit(result: dict[str, object], package_root: Path) -> str | None:
    direct = result.get("source_commit")
    if isinstance(direct, str) and direct:
        return direct
    for name in ("artifact_manifest.json", "summary.json"):
        path = package_root / name
        if not path.is_file():
            continue
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
        if isinstance(raw, dict):
            source = cast("dict[object, object]", raw).get("source_commit")
            if isinstance(source, str) and source:
                return source
    return None


def _check_ancestry(  # noqa: PLR0911
    root: Path,
    source_commit: str,
    *,
    merged_commit: str | None = None,
) -> str | None:
    shallow = _run_git(root, "rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0:
        return "git repository metadata is unavailable"
    if shallow.stdout.strip() == "true":
        return "repository is shallow; fetch full history before verifying evidence ancestry"
    source_exists = _run_git(root, "cat-file", "-e", f"{source_commit}^{{commit}}")
    if source_exists.returncode != 0:
        return f"source commit {source_commit} is unavailable in the full repository history"
    completed = _run_git(root, "merge-base", "--is-ancestor", source_commit, "HEAD")
    if completed.returncode == 0:
        return None
    if merged_commit is None:
        return f"source commit {source_commit} is not an ancestor of HEAD"
    merged_exists = _run_git(root, "cat-file", "-e", f"{merged_commit}^{{commit}}")
    if merged_exists.returncode != 0:
        return f"declared squash-merge commit {merged_commit} is unavailable"
    merged = _run_git(root, "merge-base", "--is-ancestor", merged_commit, "HEAD")
    if merged.returncode != 0:
        return f"declared squash-merge commit {merged_commit} is not an ancestor of HEAD"
    return None


def _check_evidence_package(root: Path, package: EvidencePackage) -> CheckResult:
    package_root = root / package.relative_root
    name = f"evidence-stage-{package.stage}"
    if not package_root.is_dir():
        return _fail(name, f"missing package {package.relative_root.as_posix()}")
    try:
        result = package.verifier(package_root)
        source = _source_commit(result, package_root)
        if source is not None:
            ancestry_error = _check_ancestry(
                root,
                source,
                merged_commit=package.merged_commit,
            )
            if ancestry_error is not None:
                return _fail(name, ancestry_error)
    except (OSError, UnicodeError, TypeError, ValueError, RuntimeError) as error:
        return _fail(name, f"{type(error).__name__}: {error}")
    source_detail = "legacy contract" if source is None else f"source {source[:12]}"
    return _pass(name, f"{package.slug}: hashes/contracts valid; {source_detail}")


def _check_evidence(root: Path, packages: tuple[EvidencePackage, ...]) -> list[CheckResult]:
    return [_check_evidence_package(root, package) for package in packages]


def _check_canonical_configs(root: Path) -> CheckResult:
    try:
        config = load_simulator_config(root / "configs" / "serving_lazy_kv_reservation.yaml")
        if not config.scheduler.lazy_kv_reservation:
            return _fail("canonical-configs", "Stage 18 canonical config did not enable lazy KV")
        _ = ServingRuntimeConfig(
            executor=ServingExecutorName.PAGED_ATTENTION,
            max_active_requests=2,
            max_cached_tokens=8,
            kv_cache_backend=KVCacheBackend.PAGED,
            kv_block_tokens=2,
            kv_num_blocks=4,
            prefix_cache=True,
            apc_prefill_strategy=APCPrefillStrategy.SEQUENTIAL,
            max_scheduled_tokens=8,
            prefill_chunk_tokens=2,
            kv_preemption=True,
            lazy_kv_reservation=True,
            kv_overcommit_ratio=2.0,
            command_queue_size=16,
            stream_buffer_size=8,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        return _fail("canonical-configs", f"{type(error).__name__}: {error}")
    return _pass("canonical-configs", "Stage 18 simulation and Stage 19 runtime configs validate")


def _tiny_model() -> GPT:
    original = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(20260826)
        return GPT(
            GPTConfig(
                vocab_size=23,
                block_size=8,
                n_layer=1,
                n_head=1,
                n_embd=8,
                dropout=0.0,
                bias=False,
            )
        ).eval()
    finally:
        torch.set_rng_state(original)


def _check_runtime_smoke(root: Path) -> CheckResult:
    try:
        simulator = load_simulator_config(root / "configs" / "serving_lazy_kv_reservation.yaml")
        with tempfile.TemporaryDirectory(prefix="minigpt-doctor-") as raw_output:
            result = run_simulation(simulator, output_dir=Path(raw_output) / "simulation")
        if result.metrics.completed_requests != len(simulator.requests):
            return _fail("runtime-smoke", "Stage 18 canonical simulation did not complete")
        model = _tiny_model()
        runtime = build_serving_runtime(
            model=model,
            block_size=model.config.block_size,
            num_threads=1,
            config=ServingRuntimeConfig(
                executor=ServingExecutorName.PAGED_ATTENTION,
                max_active_requests=2,
                max_cached_tokens=8,
                kv_cache_backend=KVCacheBackend.PAGED,
                kv_block_tokens=2,
                kv_num_blocks=4,
                prefix_cache=False,
                apc_prefill_strategy=APCPrefillStrategy.SEQUENTIAL,
                max_scheduled_tokens=8,
                prefill_chunk_tokens=2,
                kv_preemption=True,
                lazy_kv_reservation=True,
                kv_overcommit_ratio=2.0,
                command_queue_size=16,
                stream_buffer_size=8,
            ),
            checkpoint_sha256="0" * 64,
            tokenizer_sha256="1" * 64,
        )
        if not runtime.engine.metrics().lazy_kv_reservation_enabled:
            return _fail("runtime-smoke", "Stage 19 runtime did not wire lazy reservation")
        runtime.engine.release_all_cache_resources()
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        return _fail("runtime-smoke", f"{type(error).__name__}: {error}")
    return _pass("runtime-smoke", "canonical simulation and real runtime wiring completed")


def _check_cli_subprocess(root: Path) -> CheckResult:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-m", "minigpt", "--version"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != __version__:
        return _fail("installed-cli", "python -m minigpt --version failed or drifted")
    return _pass("installed-cli", f"python -m minigpt reports {__version__}")


def _check_clean(root: Path) -> CheckResult:
    completed = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    if completed.returncode != 0:
        return _fail("clean-worktree", "git status failed")
    dirty = [line for line in completed.stdout.splitlines() if line.strip()]
    if dirty:
        return _fail("clean-worktree", f"worktree has {len(dirty)} changed path(s)")
    return _pass("clean-worktree", "tracked and untracked worktree state is clean")


def verify_project(
    root: Path,
    *,
    mode: DoctorMode = DoctorMode.QUICK,
    require_clean: bool = False,
    packages: tuple[EvidencePackage, ...] | None = None,
) -> DoctorReport:
    """Run deterministic package, evidence, configuration, and runtime checks."""
    resolved = root.resolve()
    selected_packages = evidence_registry() if packages is None else packages
    checks: list[CheckResult] = [_check_version(), _check_docs(resolved)]
    checks.extend(_check_evidence(resolved, selected_packages))
    checks.append(_check_canonical_configs(resolved))
    if mode in {DoctorMode.CI, DoctorMode.RELEASE}:
        checks.extend((_check_runtime_smoke(resolved), _check_cli_subprocess(resolved)))
    if require_clean:
        checks.append(_check_clean(resolved))
    return DoctorReport(mode=mode, project_version=__version__, checks=tuple(checks))


def build_parser() -> argparse.ArgumentParser:
    """Create the project-doctor command-line parser."""
    parser = argparse.ArgumentParser(description="Verify miniGPT release and evidence contracts.")
    _ = parser.add_argument("--root", type=Path, default=Path.cwd())
    _ = parser.add_argument("--mode", choices=tuple(DoctorMode), default=DoctorMode.QUICK)
    _ = parser.add_argument("--require-clean", action="store_true")
    _ = parser.add_argument("--output", type=Path)
    return parser


def _write_report(path: Path, document: dict[str, object]) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(payload, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected verification mode and emit stable JSON."""
    arguments = build_parser().parse_args(argv)
    try:
        report = verify_project(
            cast("Path", arguments.root),
            mode=DoctorMode(cast("str", arguments.mode)),
            require_clean=cast("bool", arguments.require_clean),
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        document: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "mode": cast("str", arguments.mode),
            "project_version": __version__,
            "repository_root": ".",
            "passed": False,
            "checks": [_fail("project-doctor", f"{type(error).__name__}: {error}").to_document()],
        }
        output = cast("Path | None", arguments.output)
        if output is None:
            print(json.dumps(document, indent=2, sort_keys=True))  # noqa: T201
        else:
            _write_report(output, document)
        return 1
    document = report.to_document()
    output = cast("Path | None", arguments.output)
    if output is None:
        print(json.dumps(document, indent=2, sort_keys=True))  # noqa: T201
    else:
        _write_report(output, document)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
