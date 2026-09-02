"""Generate and verify hash-bound Story Forge v1.1 product evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Never, TypeAlias, cast

from typing_extensions import override

from minigpt import __version__

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None

EVIDENCE_DIRECTORY: Final = Path("docs/results/story-forge-product")
_MANIFEST_NAME: Final = "manifest.json"
_ARTIFACT_NAMES: Final = (
    "README.md",
    "product_summary.json",
    "runtime_contract.json",
    "verification.json",
    "web_assets.json",
)
_SOURCE_COMMIT_PATTERN: Final = re.compile(r"[0-9a-f]{40}")
_SHA256_HEX_LENGTH: Final = 64
_MODEL_CHECKPOINT_SHA256: Final = "9abcf022471df6766d4676fc93a3ecc6a8eaa9ec95c79a391227c2906b85e710"
_MODEL_TOKENIZER_SHA256: Final = "7c897e0d51d135f6bb24bdbd18b0e40db88c0e62defcff9d23f3a204e092585c"
_MODEL_PARAMETER_COUNT: Final = 4_928_144
_SOURCE_PATHS: Final = (
    "configs/story_forge_public_demo.yaml",
    "scripts/build_public_demo_site.py",
    "scripts/build_systems_lab_assets.py",
    "scripts/start_story_forge_demo.ps1",
    "scripts/stop_story_forge_demo.ps1",
    "scripts/validate_story_forge_model.py",
    "src/minigpt/engine_runner.py",
    "src/minigpt/prediction.py",
    "src/minigpt/public_demo.py",
    "src/minigpt/serving.py",
    "src/minigpt/story_forge_product.py",
    "src/minigpt/story_forge_systems.py",
    "web/app.js",
    "web/index.html",
    "web/styles.css",
    "web/data/automatic_prefix_cache.json",
    "web/data/continuous_batching.json",
    "web/data/kv_preemption.json",
    "web/data/lazy_reservation.json",
)


@dataclass(frozen=True, slots=True)
class StoryForgeProductEvidenceError(RuntimeError):
    """Report an invalid or unverifiable Story Forge evidence package."""

    reason: str

    @override
    def __str__(self) -> str:
        return f"Story Forge product evidence error: {self.reason}"


def _fail(reason: str) -> Never:
    raise StoryForgeProductEvidenceError(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(document: Mapping[str, JsonValue]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            _ = stream.write(content)
        _ = temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, document: Mapping[str, JsonValue]) -> None:
    _write_bytes(path, _canonical_json(document))


def _load_json_object(path: Path) -> dict[str, JsonValue]:
    try:
        parsed = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"cannot read {path}: {error}")
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) for key in cast("dict[object, object]", parsed)
    ):
        _fail(f"{path} must contain one JSON object")
    return cast("dict[str, JsonValue]", parsed)


def _git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed git invocation
        ["git", *arguments],  # noqa: S607 - git resolved by development toolchain
        cwd=repository_root,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _validate_source_commit(repository_root: Path, source_commit: str) -> None:
    if _SOURCE_COMMIT_PATTERN.fullmatch(source_commit) is None:
        _fail("source_commit must be one lowercase 40-character Git SHA")
    object_check = _git(
        repository_root,
        ["cat-file", "-e", f"{source_commit}^{{commit}}"],
        check=False,
    )
    if object_check.returncode != 0:
        _fail(f"source commit {source_commit} does not exist")
    ancestry = _git(
        repository_root,
        ["merge-base", "--is-ancestor", source_commit, "HEAD"],
        check=False,
    )
    if ancestry.returncode != 0:
        _fail(f"source commit {source_commit} is not an ancestor of HEAD")


def _validate_verification(document: Mapping[str, JsonValue]) -> None:
    expected = {"schema_version", "quality", "tests", "fresh_worktree", "runtime_smoke"}
    if set(document) != expected:
        _fail("verification input has unexpected or missing keys")
    if document.get("schema_version") != 1:
        _fail("verification schema_version must equal 1")
    for key in ("quality", "fresh_worktree", "runtime_smoke"):
        value = document.get(key)
        if not isinstance(value, dict) or value.get("status") != "pass":
            _fail(f"verification {key} must have pass status")
    tests = document.get("tests")
    if not isinstance(tests, dict):
        _fail("verification tests must be an object")
    tests_map = cast("dict[str, JsonValue]", tests)
    for key in ("passed", "skipped", "failed"):
        value = tests_map.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"verification tests.{key} must be a non-negative integer")
    if tests_map.get("failed") != 0:
        _fail("verification cannot publish failed tests")


def _committed_source_bytes(
    repository_root: Path,
    source_commit: str,
    relative: str,
) -> bytes:
    """Read one exact source artifact from the declared review commit."""
    result = subprocess.run(  # noqa: S603 - fixed git invocation
        ["git", "show", f"{source_commit}:{relative}"],  # noqa: S607
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        _fail(f"source commit does not contain required artifact: {relative}")
    return result.stdout


def _source_file_records(repository_root: Path, source_commit: str) -> list[JsonValue]:
    """Bind source bytes from the declared commit, never from a dirty worktree."""
    records: list[JsonValue] = []
    for relative in _SOURCE_PATHS:
        content = _committed_source_bytes(repository_root, source_commit, relative)
        records.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return records


def _web_asset_records(repository_root: Path, source_commit: str) -> list[JsonValue]:
    """Bind committed Systems Lab scenario assets with their bytes and hashes."""
    records: list[JsonValue] = []
    for name in (
        "automatic_prefix_cache",
        "continuous_batching",
        "kv_preemption",
        "lazy_reservation",
    ):
        relative = f"web/data/{name}.json"
        content = _committed_source_bytes(repository_root, source_commit, relative)
        records.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return records


def _product_summary(source_commit: str) -> dict[str, JsonValue]:
    """Build the bounded product summary with the v1.1 claim policy."""
    return {
        "schema_version": 1,
        "project_version": __version__,
        "source_commit": source_commit,
        "model": {
            "checkpoint_sha256": _MODEL_CHECKPOINT_SHA256,
            "tokenizer_sha256": _MODEL_TOKENIZER_SHA256,
            "parameter_count": _MODEL_PARAMETER_COUNT,
        },
        "features": ["story_forge", "prediction_lab", "systems_lab"],
        "claim_policy": {
            "verdict": "descriptive_only",
            "general_chat_claim": False,
            "semantic_understanding_claim": False,
            "authorship_detection_claim": False,
            "production_ready_claim": False,
            "public_cutover_claim": False,
        },
    }


def _runtime_contract() -> dict[str, JsonValue]:
    """Describe the bounded Story Forge API contract for auditability."""
    return {
        "schema_version": 1,
        "endpoints": {
            "story_branches": "/demo/story/branches",
            "predict_next": "/demo/predict/next",
            "predict_score": "/demo/predict/score",
        },
        "story": {
            "branch_count": 3,
            "max_branch_tokens": 64,
            "max_rounds": 4,
            "stable_derived_seeds": True,
            "bpe_safe_snapshot_sse": True,
        },
        "prediction": {
            "max_top_k": 10,
            "owner_thread_only": True,
            "no_sampling": True,
            "no_rng_advance": True,
            "no_kv_mutation": True,
        },
        "privacy": {
            "no_prompt_logging": True,
            "no_path_leakage": True,
            "text_content_only": True,
        },
    }


def generate_story_forge_product_evidence(
    *,
    repository_root: Path,
    source_commit: str,
    verification_path: Path,
    output_directory: Path,
) -> Path:
    """Assemble and self-verify the hash-bound Story Forge product evidence package.

    The package binds the reviewed source commit, the selected external
    checkpoint/tokenizer hashes and parameter count, the bounded API contract,
    the recorded Systems Lab assets, the deterministic site source, and an
    external verification report. It never embeds checkpoint/data bytes or
    absolute paths.
    """
    _validate_source_commit(repository_root, source_commit)
    verification = _validate_verification(_load_json_object(verification_path))
    del verification

    output_directory.mkdir(parents=True, exist_ok=True)
    summary = _product_summary(source_commit)
    _write_json(output_directory / "product_summary.json", summary)
    _write_json(output_directory / "runtime_contract.json", _runtime_contract())
    _write_json(
        output_directory / "verification.json",
        _load_json_object(verification_path),
    )
    _write_json(
        output_directory / "web_assets.json",
        {
            "schema_version": 1,
            "source_files": _source_file_records(repository_root, source_commit),
            "systems_lab_assets": _web_asset_records(repository_root, source_commit),
        },
    )
    _ = (output_directory / "README.md").write_text(
        _readme(source_commit),
        encoding="utf-8",
        newline="\n",
    )

    manifest_path = output_directory / _MANIFEST_NAME
    artifacts: list[JsonValue] = [
        {
            "path": path.relative_to(output_directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "source_commit": source_commit,
            "artifacts": artifacts,
        },
    )
    _ = verify_story_forge_product_evidence(
        repository_root=repository_root,
        evidence_directory=output_directory,
    )
    return output_directory


def _manifest_entries(entries: list[object]) -> dict[str, tuple[int, str]]:
    expected: dict[str, tuple[int, str]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            _fail("manifest artifact entries must be objects")
        entry = cast("dict[str, object]", raw_entry)
        raw_path = entry.get("path")
        raw_size = entry.get("bytes")
        raw_digest = entry.get("sha256")
        if not isinstance(raw_path, str) or Path(raw_path).is_absolute():
            _fail("manifest artifact entry fields are invalid")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            _fail("manifest artifact entry fields are invalid")
        if not isinstance(raw_digest, str) or len(raw_digest) != _SHA256_HEX_LENGTH:
            _fail("manifest artifact entry fields are invalid")
        relative = raw_path
        size = raw_size
        digest = raw_digest
        if relative in expected:
            _fail(f"duplicate manifest path {relative}")
        expected[relative] = (size, digest)
    return expected


def verify_story_forge_product_evidence(  # noqa: C901, PLR0912
    *,
    repository_root: Path,
    evidence_directory: Path,
) -> dict[str, JsonValue]:
    """Verify exact membership, hashes, source identity, and bounded claims."""
    manifest_path = evidence_directory / _MANIFEST_NAME
    if not manifest_path.is_file():
        _fail(f"missing manifest: {manifest_path}")
    manifest = _load_json_object(manifest_path)
    raw_source_commit = manifest.get("source_commit")
    if not isinstance(raw_source_commit, str) or not raw_source_commit:
        _fail("manifest source_commit must be non-empty")
    source_commit = raw_source_commit
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        _fail("manifest artifacts must be a list")
    expected = _manifest_entries(cast("list[object]", entries))
    actual = {
        path.relative_to(evidence_directory).as_posix()
        for path in evidence_directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    required_artifacts = set(_ARTIFACT_NAMES)
    if actual != required_artifacts or set(expected) != required_artifacts:
        _fail("artifact membership differs from manifest")
    for relative, (size, digest) in expected.items():
        path = evidence_directory / relative
        if path.stat().st_size != size or _sha256(path) != digest:
            _fail(f"artifact hash mismatch for {relative}")

    summary = _load_json_object(evidence_directory / "product_summary.json")
    if summary.get("source_commit") != source_commit:
        _fail("summary source_commit differs from manifest")
    if summary.get("project_version") != __version__:
        _fail("summary project_version does not match the authored version")
    model = cast("dict[str, JsonValue]", summary.get("model"))
    if model.get("checkpoint_sha256") != _MODEL_CHECKPOINT_SHA256:
        _fail("summary checkpoint hash is not the reviewed story_forge_5m artifact")
    if model.get("tokenizer_sha256") != _MODEL_TOKENIZER_SHA256:
        _fail("summary tokenizer hash is not the reviewed story_forge artifact")
    if model.get("parameter_count") != _MODEL_PARAMETER_COUNT:
        _fail("summary parameter count differs from the reviewed 5M value")

    claim_policy = cast("dict[str, JsonValue]", summary.get("claim_policy"))
    if claim_policy.get("verdict") != "descriptive_only":
        _fail("claim policy verdict must be descriptive_only")
    for key in (
        "general_chat_claim",
        "semantic_understanding_claim",
        "authorship_detection_claim",
        "production_ready_claim",
        "public_cutover_claim",
    ):
        if claim_policy.get(key) is not False:
            _fail(f"claim policy {key} must be false")

    if _load_json_object(evidence_directory / "runtime_contract.json") != _runtime_contract():
        _fail("runtime contract differs from the reviewed product contract")
    verification = _load_json_object(evidence_directory / "verification.json")
    _validate_verification(verification)
    web_assets = _load_json_object(evidence_directory / "web_assets.json")
    expected_web_asset_keys = {"schema_version", "source_files", "systems_lab_assets"}
    if set(web_assets) != expected_web_asset_keys or web_assets.get("schema_version") != 1:
        _fail("web_assets document has an invalid schema")
    if web_assets.get("source_files") != _source_file_records(repository_root, source_commit):
        _fail("source file records do not match the declared source commit")
    if web_assets.get("systems_lab_assets") != _web_asset_records(
        repository_root,
        source_commit,
    ):
        _fail("Systems Lab records do not match the declared source commit")

    _validate_source_commit(repository_root, source_commit)
    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "project_version": __version__,
        "verified": True,
    }


def _readme(source_commit: str) -> str:
    return (
        "\n".join(
            (
                "# Story Forge — product, Prediction Lab, and Systems Lab evidence",
                "",
                "This package links the post-v1 miniGPT Story Forge product to its",
                "reviewed source commit, the selected external 5M checkpoint and",
                "tokenizer hashes, the bounded story/prediction API contracts, the",
                "recorded Systems Lab scenario assets, and an external verification",
                "report. It is a presentation- and release-hardening extension and does",
                "not alter historical Stage 7A-21 Evidence.",
                "",
                f"Source commit: {source_commit}.",
                "Claim policy: descriptive_only. No general chat, semantic",
                "understanding, authorship detection, production readiness, or public",
                "cutover claim is made.",
            )
        )
        + "\n"
    )
