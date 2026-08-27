"""Generate and verify the hash-bound miniGPT v1.0 capstone evidence package."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Never, TypeAlias, cast

import torch
from typing_extensions import override

from minigpt import __version__
from minigpt.data import JsonValue
from minigpt.evidence_registry import evidence_registry
from minigpt.model import GPT
from minigpt.paged_kv_cache import KVCacheBackend
from minigpt.project_doctor import DoctorMode, verify_project
from minigpt.release_validation import validate_release_artifacts
from minigpt.serving import APCPrefillStrategy
from minigpt.serving_runtime import ServingExecutorName, ServingRuntimeConfig, build_serving_runtime
from minigpt.serving_simulator import load_simulator_config, run_simulation
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME = "21"
_RELEASE_VERSION = "1.0.0"
_SHA256_HEX_LENGTH = 64
_KV_OVERCOMMIT_RATIO = 2.0


@dataclass(slots=True)
class Stage21EvidenceVerificationError(ValueError):
    """Report invalid v1 release evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the capstone evidence failure."""
        return f"invalid Stage 21 evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise Stage21EvidenceVerificationError(reason)


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
    document = cast("dict[object, object]", raw)
    if any(not isinstance(key, str) for key in document):
        _invalid(f"{path} must contain string keys")
    return cast("EvidenceDocument", document)


def _write_json(path: Path, document: EvidenceDocument) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _gate_entries(document: EvidenceDocument, label: str) -> tuple[EvidenceDocument, ...]:
    if document.get("exit_code") != 0:
        _invalid(f"{label} evidence did not pass")
    raw_entries = document.get("results", document.get("partitions"))
    if not isinstance(raw_entries, list) or not raw_entries:
        _invalid(f"{label} evidence must contain non-empty results or partitions")
    entries: list[EvidenceDocument] = []
    for raw_entry in cast("list[object]", raw_entries):
        if not isinstance(raw_entry, dict):
            _invalid(f"{label} evidence entries must be objects")
        entry = cast("dict[object, object]", raw_entry)
        if any(not isinstance(key, str) for key in entry):
            _invalid(f"{label} evidence entries must use string keys")
        document_entry = cast("EvidenceDocument", entry)
        if document_entry.get("exit_code") != 0:
            _invalid(f"{label} evidence contains a failed command")
        command = document_entry.get("command")
        valid_command = isinstance(command, str) and bool(command.strip())
        if isinstance(command, list):
            valid_command = bool(command) and all(
                isinstance(item, str) and bool(item) for item in command
            )
        if not valid_command:
            _invalid(f"{label} evidence command is missing or malformed")
        entries.append(document_entry)
    return tuple(entries)


def _gate_command_text(entries: tuple[EvidenceDocument, ...]) -> str:
    commands: list[str] = []
    for entry in entries:
        command = entry["command"]
        if isinstance(command, str):
            commands.append(command)
        else:
            commands.append(" ".join(cast("list[str]", command)))
    return "\n".join(commands).replace("\\", "/")


def _verify_gate_documents(
    exact_resume: EvidenceDocument,
    lifecycle: EvidenceDocument,
    full_tests: EvidenceDocument,
    quality: EvidenceDocument,
) -> None:
    exact_entries = _gate_entries(exact_resume, "exact resume")
    lifecycle_entries = _gate_entries(lifecycle, "lifecycle")
    full_entries = _gate_entries(full_tests, "full tests")
    quality_entries = _gate_entries(quality, "quality gates")

    passed = full_tests.get("passed")
    failed = full_tests.get("failed")
    skipped = full_tests.get("skipped")
    if (
        isinstance(passed, bool)
        or not isinstance(passed, int)
        or passed <= 0
        or isinstance(failed, bool)
        or not isinstance(failed, int)
        or failed != 0
        or isinstance(skipped, bool)
        or not isinstance(skipped, int)
        or skipped < 0
    ):
        _invalid("full-test evidence counts are invalid")

    exact_text = _gate_command_text(exact_entries)
    for required in (
        "tests/test_checkpoint.py",
        "tests/test_trainer.py",
        "tests/test_training_components.py",
    ):
        if required not in exact_text:
            _invalid(f"exact-resume evidence omitted {required}")

    lifecycle_text = _gate_command_text(lifecycle_entries)
    for required in (
        "tests/test_serving_runtime.py",
        "tests/test_serve_subprocess.py",
        "tests/test_cli.py",
        "tests/test_project_doctor.py",
        "tests/test_release_validation.py",
        "tests/test_stage19_evidence.py",
        "tests/test_stage20_evidence.py",
        "tests/test_stage21_evidence.py",
    ):
        if required not in lifecycle_text:
            _invalid(f"lifecycle evidence omitted {required}")

    full_text = _gate_command_text(full_entries)
    if "pytest" not in full_text or "tests/" not in full_text:
        _invalid("full-test evidence does not describe pytest test-file partitions")

    quality_text = _gate_command_text(quality_entries)
    for required in (
        "pip check",
        "ruff format --check src tests",
        "ruff check src tests",
        "basedpyright",
    ):
        if required not in quality_text:
            _invalid(f"quality-gate evidence omitted {required}")


def _absolute_command_token(token: str) -> bool:
    return PurePosixPath(token).is_absolute() or PureWindowsPath(token).is_absolute()


def _verify_portable_external_document(value: JsonValue) -> None:
    """Reject host-specific absolute command tokens before packaging external gate output."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "command":
                if isinstance(child, str):
                    tokens = (child,)
                elif isinstance(child, list) and all(isinstance(item, str) for item in child):
                    tokens = tuple(cast("list[str]", child))
                else:
                    _invalid("external evidence command must be a string or string list")
                if any(_absolute_command_token(token) for token in tokens):
                    _invalid("external evidence command contains an absolute host path")
            _verify_portable_external_document(child)
    elif isinstance(value, list):
        for item in value:
            _verify_portable_external_document(item)


def _run(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed release-evidence commands
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _installed_console_script() -> Path:
    scripts_root = Path(sys.executable).parent
    return scripts_root / "minigpt.exe" if sys.platform == "win32" else scripts_root / "minigpt"


def _generated_metadata_untracked(root: Path) -> bool:
    result = _run(("git", "ls-files", "*.egg-info", "**/*.egg-info/**"), cwd=root)
    return result.returncode == 0 and not result.stdout.strip()


def generate_stage21_version_evidence(root: Path, output_path: Path) -> Path:
    """Prove source, metadata, module CLI, console CLI, and dynamic build version agree."""
    source_version = str(__version__)
    pyproject = cast(
        "dict[str, object]",
        tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")),
    )
    project = cast("dict[str, object]", pyproject["project"])
    setuptools = cast("dict[str, object]", pyproject["tool"])
    module_result = _run((sys.executable, "-m", "minigpt", "--version"), cwd=root)
    console_path = _installed_console_script()
    if not console_path.is_file():
        _invalid("installed minigpt console script is unavailable")
    console_result = _run((str(console_path), "--version"), cwd=root)
    installed = importlib.metadata.version("minitrain-gpt")
    dynamic = project.get("dynamic")
    tool_setuptools = cast("dict[str, object]", setuptools["setuptools"])
    dynamic_config = cast("dict[str, object]", tool_setuptools["dynamic"])
    version_config = cast("dict[str, object]", dynamic_config["version"])
    version_attr = version_config.get("attr")
    passed = (
        source_version == _RELEASE_VERSION
        and installed == _RELEASE_VERSION
        and module_result.returncode == 0
        and module_result.stdout.strip() == _RELEASE_VERSION
        and console_result.returncode == 0
        and console_result.stdout.strip() == _RELEASE_VERSION
        and dynamic == ["version"]
        and version_attr == "minigpt._version.__version__"
        and "version" not in project
        and _generated_metadata_untracked(root)
    )
    if not passed:
        _invalid("v1 source/distribution/CLI version contract did not pass")
    return _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "release_version": _RELEASE_VERSION,
            "source_version": source_version,
            "installed_metadata_version": installed,
            "module_cli_version": module_result.stdout.strip(),
            "console_cli_version": console_result.stdout.strip(),
            "dynamic_version_metadata": True,
            "generated_egg_info_untracked": True,
            "version_contract_passed": True,
        },
    )


def generate_stage21_registry_evidence(root: Path, output_path: Path) -> Path:
    """Verify the complete non-self-referential Stage 7A-20 release registry."""
    registry = evidence_registry()
    report = verify_project(root, mode=DoctorMode.QUICK, packages=registry)
    stages = tuple(item.stage for item in registry)
    if not report.passed or stages[0] != "7A" or stages[-1] != "20":
        _invalid("v1 release registry did not verify Stage 7A-20")
    evidence_checks = tuple(
        item for item in report.checks if item.name.startswith("evidence-stage-")
    )
    if len(evidence_checks) != len(registry):
        _invalid("project doctor did not report every registered evidence package")
    return _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "registry_first_stage": stages[0],
                "registry_last_stage": stages[-1],
                "registry_packages": len(registry),
                "registered_stages": list(stages),
                "all_packages_passed": True,
                "all_modern_sources_are_ancestors": True,
                "non_shallow_history_required": True,
                "doctor_report": cast("JsonValue", report.to_document()),
            },
        ),
    )


def generate_stage21_release_evidence(root: Path, output_path: Path) -> Path:
    """Run the release doctor and independently capture wheel/sdist artifact hashes."""
    report = verify_project(root, mode=DoctorMode.RELEASE)
    if not report.passed:
        _invalid("release doctor did not pass")
    artifacts = validate_release_artifacts(root)
    if artifacts.project_version != _RELEASE_VERSION:
        _invalid("release artifacts do not report v1.0.0")
    return _write_json(
        output_path,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "release_doctor_passed": True,
            "release_version": artifacts.project_version,
            "wheel_built": artifacts.wheel_built,
            "sdist_built": artifacts.sdist_built,
            "wheel_sha256": artifacts.wheel_sha256,
            "sdist_sha256": artifacts.sdist_sha256,
            "required_modules_present": artifacts.required_modules_present,
            "fresh_install_passed": artifacts.fresh_install_passed,
            "pip_check_passed": artifacts.pip_check_passed,
            "metadata_version_passed": artifacts.metadata_version_passed,
            "wheel_import_isolated": artifacts.wheel_import_isolated,
            "module_entrypoint_passed": artifacts.module_entrypoint_passed,
            "module_help_passed": artifacts.module_help_passed,
            "console_script_passed": artifacts.console_script_passed,
            "console_help_passed": artifacts.console_help_passed,
            "fresh_quick_doctor_passed": artifacts.quick_doctor_passed,
            "release_report": cast("JsonValue", report.to_document()),
        },
    )


def _tiny_model() -> GPT:
    original = torch.get_rng_state()
    try:
        _ = torch.default_generator.manual_seed(20260826)
        return GPT(
            GPTConfig(
                vocab_size=31,
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


def generate_stage21_runtime_evidence(root: Path, output_path: Path) -> Path:
    """Run the Stage 18 canonical simulation and construct the Stage 19 real runtime."""
    simulator = load_simulator_config(root / "configs" / "serving_lazy_kv_reservation.yaml")
    with tempfile.TemporaryDirectory(prefix="minigpt-v1-runtime-") as raw_temp:
        simulation = run_simulation(simulator, output_dir=Path(raw_temp) / "simulation")
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
            prefix_cache=True,
            apc_prefill_strategy=APCPrefillStrategy.SEQUENTIAL,
            max_scheduled_tokens=8,
            prefill_chunk_tokens=2,
            kv_preemption=True,
            lazy_kv_reservation=True,
            kv_overcommit_ratio=_KV_OVERCOMMIT_RATIO,
            command_queue_size=16,
            stream_buffer_size=8,
        ),
        checkpoint_sha256="0" * 64,
        tokenizer_sha256="1" * 64,
    )
    metrics = runtime.engine.metrics()
    passed = (
        simulation.metrics.completed_requests == len(simulator.requests)
        and simulation.metrics.failed_requests == 0
        and metrics.lazy_kv_reservation_enabled
        and metrics.kv_overcommit_ratio == _KV_OVERCOMMIT_RATIO
    )
    runtime.engine.release_all_cache_resources()
    if not passed:
        _invalid("Stage 18 simulation or Stage 19 runtime capstone did not pass")
    return _write_json(
        output_path,
        cast(
            "EvidenceDocument",
            {
                "schema_version": 1,
                "stage": STAGE_NAME,
                "stage18_canonical_simulation_passed": True,
                "simulation_completed_requests": simulation.metrics.completed_requests,
                "simulation_failed_requests": simulation.metrics.failed_requests,
                "stage19_real_runtime_passed": True,
                "runtime_executor": ServingExecutorName.PAGED_ATTENTION.value,
                "runtime_lazy_kv_reservation": True,
                "runtime_kv_overcommit_ratio": _KV_OVERCOMMIT_RATIO,
                "runtime_prefix_cache": True,
                "runtime_resources_released": True,
            },
        ),
    )


def generate_stage21_evidence(  # noqa: PLR0913
    *,
    version_path: Path,
    registry_path: Path,
    release_path: Path,
    runtime_path: Path,
    exact_resume_path: Path,
    lifecycle_path: Path,
    full_tests_path: Path,
    quality_path: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Build and immediately verify the v1.0 capstone evidence package."""
    if not source_commit:
        _invalid("source_commit must be non-empty")
    inputs = {
        "version.json": version_path,
        "registry.json": registry_path,
        "release.json": release_path,
        "runtime.json": runtime_path,
        "exact_resume_tests.json": exact_resume_path,
        "lifecycle_tests.json": lifecycle_path,
        "full_tests.json": full_tests_path,
        "quality_gates.json": quality_path,
    }
    for path in inputs.values():
        if not path.is_file():
            _invalid(f"evidence input does not exist: {path}")
    package_root.mkdir(parents=True, exist_ok=True)
    evidence_root = package_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    for name, path in inputs.items():
        document = _read_json(path)
        _verify_portable_external_document(cast("JsonValue", document))
        _ = _write_json(evidence_root / name, document)
    version = _read_json(evidence_root / "version.json")
    registry = _read_json(evidence_root / "registry.json")
    release = _read_json(evidence_root / "release.json")
    runtime = _read_json(evidence_root / "runtime.json")
    exact_resume = _read_json(evidence_root / "exact_resume_tests.json")
    lifecycle = _read_json(evidence_root / "lifecycle_tests.json")
    full_tests = _read_json(evidence_root / "full_tests.json")
    quality = _read_json(evidence_root / "quality_gates.json")
    required = (
        version.get("version_contract_passed"),
        version.get("dynamic_version_metadata"),
        registry.get("all_packages_passed"),
        registry.get("all_modern_sources_are_ancestors"),
        release.get("release_doctor_passed"),
        release.get("wheel_built"),
        release.get("sdist_built"),
        release.get("fresh_install_passed"),
        release.get("pip_check_passed"),
        release.get("metadata_version_passed"),
        release.get("wheel_import_isolated"),
        release.get("module_help_passed"),
        release.get("console_help_passed"),
        release.get("fresh_quick_doctor_passed"),
        runtime.get("stage18_canonical_simulation_passed"),
        runtime.get("stage19_real_runtime_passed"),
        runtime.get("runtime_resources_released"),
    )
    if not all(value is True for value in required):
        _invalid("v1 capstone evidence contracts did not all pass")
    _verify_gate_documents(exact_resume, lifecycle, full_tests, quality)
    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "release_version": _RELEASE_VERSION,
        "project_complete": True,
        "single_version_source": True,
        "stage7a_through_20_registry_passed": True,
        "all_modern_sources_are_ancestors": True,
        "wheel_built": True,
        "sdist_built": True,
        "fresh_install_passed": True,
        "release_doctor_passed": True,
        "stage18_simulation_passed": True,
        "stage19_runtime_passed": True,
        "checkpoint_v2_exact_resume_passed": True,
        "full_test_suite_passed": True,
        "quality_gates_passed": True,
        "lifecycle_passed": True,
        "benchmark_strict_verdict": "descriptive_only",
        "wall_clock_performance_improvement": False,
        "production_scale_performance_claim": False,
        "gpu_parity_claim": False,
    }
    _ = _write_json(package_root / "summary.json", summary)
    _ = (package_root / "README.md").write_text(
        _readme(summary, registry, release, runtime),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = package_root / "artifact_manifest.json"
    artifacts: list[JsonValue] = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _ = _write_json(
        manifest_path,
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
    _ = verify_stage21_evidence(package_root)
    return package_root


def _manifest_entries(entries: list[object]) -> dict[str, tuple[int, str]]:
    expected: dict[str, tuple[int, str]] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            _invalid("manifest artifact entries must be objects")
        entry = cast("dict[object, object]", raw)
        relative = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != _SHA256_HEX_LENGTH
        ):
            _invalid("manifest artifact entry fields are invalid")
        if relative in expected:
            _invalid(f"duplicate manifest path {relative}")
        expected[relative] = (size, digest)
    return expected


def verify_stage21_evidence(  # noqa: C901, PLR0912
    package_root: Path,
) -> EvidenceDocument:
    """Verify capstone membership, hashes, internal contracts, and bounded claims."""
    manifest_path = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != STAGE_NAME:
        _invalid("manifest stage must be 21")
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
        if path.is_file() and path != manifest_path
    }
    if actual != set(expected):
        _invalid("artifact membership differs from manifest")
    for relative, (size, digest) in expected.items():
        path = package_root / relative
        if path.stat().st_size != size or _sha256(path) != digest:
            _invalid(f"artifact hash mismatch for {relative}")
    summary = _read_json(package_root / "summary.json")
    if summary.get("source_commit") != source_commit:
        _invalid("summary source_commit differs from manifest")
    true_keys = (
        "project_complete",
        "single_version_source",
        "stage7a_through_20_registry_passed",
        "all_modern_sources_are_ancestors",
        "wheel_built",
        "sdist_built",
        "fresh_install_passed",
        "release_doctor_passed",
        "stage18_simulation_passed",
        "stage19_runtime_passed",
        "checkpoint_v2_exact_resume_passed",
        "full_test_suite_passed",
        "quality_gates_passed",
        "lifecycle_passed",
    )
    if any(summary.get(key) is not True for key in true_keys):
        _invalid("summary required release contract did not pass")
    false_keys = (
        "wall_clock_performance_improvement",
        "production_scale_performance_claim",
        "gpu_parity_claim",
    )
    if any(summary.get(key) is not False for key in false_keys):
        _invalid("summary contains an unsupported release claim")
    if summary.get("release_version") != _RELEASE_VERSION:
        _invalid("summary release version must equal 1.0.0")
    if summary.get("benchmark_strict_verdict") != "descriptive_only":
        _invalid("v1 capstone benchmark verdict must remain descriptive_only")
    version = _read_json(package_root / "evidence" / "version.json")
    release = _read_json(package_root / "evidence" / "release.json")
    registry = _read_json(package_root / "evidence" / "registry.json")
    runtime = _read_json(package_root / "evidence" / "runtime.json")
    if any(document.get("release_version") != _RELEASE_VERSION for document in (version, release)):
        _invalid("release version differs across internal evidence")
    release_contracts = (
        "release_doctor_passed",
        "wheel_built",
        "sdist_built",
        "fresh_install_passed",
        "pip_check_passed",
        "metadata_version_passed",
        "wheel_import_isolated",
        "module_entrypoint_passed",
        "module_help_passed",
        "console_script_passed",
        "console_help_passed",
        "fresh_quick_doctor_passed",
    )
    if any(release.get(key) is not True for key in release_contracts):
        _invalid("release artifact contract did not pass")
    if registry.get("registry_first_stage") != "7A" or registry.get("registry_last_stage") != "20":
        _invalid("capstone registry boundary must be Stage 7A-20")
    if runtime.get("runtime_lazy_kv_reservation") is not True:
        _invalid("capstone runtime did not enable lazy KV reservation")
    _verify_gate_documents(
        _read_json(package_root / "evidence" / "exact_resume_tests.json"),
        _read_json(package_root / "evidence" / "lifecycle_tests.json"),
        _read_json(package_root / "evidence" / "full_tests.json"),
        _read_json(package_root / "evidence" / "quality_gates.json"),
    )
    return manifest


def _readme(
    summary: EvidenceDocument,
    registry: EvidenceDocument,
    release: EvidenceDocument,
    runtime: EvidenceDocument,
) -> str:
    return (
        "\n".join(
            (
                "# miniGPT v1.0.0 Capstone Evidence",
                "",
                "Stage 21 closes the planned CPU-first miniGPT project with installable",
                "artifacts, release verification, runtime smoke, exact resume, and a full",
                "non-self-referential evidence roll-up.",
                "",
                f"Release version: {summary['release_version']}.",
                f"Registered Stage 7A-20 packages: {registry['registry_packages']}.",
                f"Fresh wheel install passed: {release['fresh_install_passed']}.",
                (
                    "Stage 18 simulation completed requests: "
                    f"{runtime['simulation_completed_requests']}."
                ),
                f"Stage 19 real runtime passed: {runtime['stage19_real_runtime_passed']}.",
                "Checkpoint v2 exact-resume and the full test suite passed.",
                "",
                "The capstone verdict is descriptive_only. v1.0 makes no universal",
                "wall-clock improvement, production-scale throughput, or GPU parity claim.",
                "",
                f"Source commit: {summary['source_commit']}.",
            )
        )
        + "\n"
    )
