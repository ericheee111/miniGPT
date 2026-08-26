"""Declare the explicit, stage-aware evidence registry used by project verification."""

from __future__ import annotations

import hashlib
import importlib
import json
import string
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

EvidenceDocument = dict[str, object]
EvidenceVerifier = Callable[[Path], EvidenceDocument]
_SHA256_HEX_LENGTH = 64
_SQUASH_MERGED_COMMITS = {
    "13A": "6c46cf4ee0087333dad1e82ae7a388a8dabadfd7",
    "13B": "6c46cf4ee0087333dad1e82ae7a388a8dabadfd7",
    "14": "93beeb453946eda9affb7cd5462185ae3910a9df",
    "15": "e2032bdc5db1bd00809080693324f376f3376268",
    "16": "e2032bdc5db1bd00809080693324f376f3376268",
}


@dataclass(frozen=True, slots=True)
class EvidencePackage:
    """Bind one project stage to its committed package and verifier contract."""

    stage: str
    slug: str
    relative_root: Path
    verifier: EvidenceVerifier
    merged_commit: str | None = None


def _read_document(path: Path) -> EvidenceDocument:
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        reason = f"{path} must contain a JSON object with string keys"
        raise TypeError(reason)
    document = cast("dict[object, object]", raw)
    if any(not isinstance(key, str) for key in document):
        reason = f"{path} must contain a JSON object with string keys"
        raise ValueError(reason)
    return cast("EvidenceDocument", document)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entries(document: EvidenceDocument) -> list[EvidenceDocument]:
    raw_entries = document.get("artifacts", document.get("files"))
    if not isinstance(raw_entries, list):
        reason = "artifact manifest must contain an artifacts or files list"
        raise TypeError(reason)
    entries: list[EvidenceDocument] = []
    for raw in cast("list[object]", raw_entries):
        if not isinstance(raw, dict):
            reason = "artifact manifest entries must be objects with string keys"
            raise TypeError(reason)
        entry = cast("dict[object, object]", raw)
        if any(not isinstance(key, str) for key in entry):
            reason = "artifact manifest entries must be objects with string keys"
            raise TypeError(reason)
        entries.append(cast("EvidenceDocument", entry))
    return entries


def _entry_path(entry: EvidenceDocument) -> str:
    for key in ("path", "relative_path", "artifact_path"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    reason = "artifact manifest entry is missing a relative path"
    raise ValueError(reason)


def _entry_size(entry: EvidenceDocument) -> int | None:
    for key in ("bytes", "size_bytes"):
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            reason = f"artifact manifest field {key!r} must be a non-negative integer"
            raise TypeError(reason)
        if value < 0:
            reason = f"artifact manifest field {key!r} must be a non-negative integer"
            raise ValueError(reason)
        return value
    return None


def _entry_digest(entry: EvidenceDocument) -> str:
    value = entry.get("sha256")
    if not isinstance(value, str):
        reason = "artifact manifest entry sha256 must be a hexadecimal string"
        raise TypeError(reason)
    if len(value) != _SHA256_HEX_LENGTH or any(char not in string.hexdigits for char in value):
        reason = "artifact manifest entry sha256 must be a 64-character hexadecimal string"
        raise ValueError(reason)
    return value.lower()


def _repository_root(package_root: Path) -> Path:
    try:
        return package_root.parents[2]
    except IndexError as error:
        reason = "evidence package root is not nested beneath a repository"
        raise ValueError(reason) from error


def _normalize_package_path(package_root: Path, relative: str) -> str:
    normalized = relative.replace("\\", "/").removeprefix("./")
    repository_root = _repository_root(package_root)
    package_relative = package_root.relative_to(repository_root).as_posix()
    return normalized.removeprefix(package_relative + "/")


def _safe_candidate(package_root: Path, relative: str) -> Path:
    candidate = (package_root / relative).resolve()
    root = package_root.resolve()
    try:
        _ = candidate.relative_to(root)
    except ValueError as error:
        reason = f"artifact path escapes package root: {relative}"
        raise ValueError(reason) from error
    return candidate


def _stage7_external_checkpoint(relative: str, entry: EvidenceDocument) -> bool:
    normalized = relative.replace("\\", "/").lower()
    explicitly_external = entry.get("committed") is False or entry.get("external") is True
    checkpoint_path = normalized.startswith("checkpoints/") or normalized.endswith((".pt", ".pth"))
    return explicitly_external or checkpoint_path


def _stage7_declared_checkpoint(  # noqa: C901
    package_root: Path,
    manifest: EvidenceDocument,
) -> int:
    raw_sources = manifest.get("sources")
    if raw_sources is None:
        return 0
    if not isinstance(raw_sources, dict):
        reason = "Stage 7A manifest sources must be an object"
        raise TypeError(reason)
    sources = cast("dict[object, object]", raw_sources)
    raw_checkpoint = sources.get("checkpoint")
    if raw_checkpoint is None:
        return 0
    if not isinstance(raw_checkpoint, dict):
        reason = "Stage 7A checkpoint source must be an object"
        raise TypeError(reason)
    checkpoint = cast("dict[object, object]", raw_checkpoint)
    if any(not isinstance(key, str) for key in checkpoint):
        reason = "Stage 7A checkpoint source must use string keys"
        raise TypeError(reason)
    document = cast("EvidenceDocument", checkpoint)
    relative = _entry_path(document).replace("\\", "/")
    path = Path(relative)
    invalid_checkpoint = (
        path.is_absolute()
        or ".." in path.parts
        or not _stage7_external_checkpoint(relative, document)
    )
    if invalid_checkpoint:
        reason = "Stage 7A external checkpoint path must be a repository-relative checkpoint"
        raise ValueError(reason)
    size = _entry_size(document)
    digest = _entry_digest(document)
    repository_root = _repository_root(package_root)
    tracked = subprocess.run(  # noqa: S603 - fixed Git metadata query
        ["git", "-C", str(repository_root), "ls-files", "--error-unmatch", "--", relative],  # noqa: S607
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if tracked.returncode == 0:
        reason = "Stage 7A checkpoint source must remain outside the committed evidence package"
        raise ValueError(reason)
    candidate = repository_root / relative
    if candidate.is_file():
        if size is not None and candidate.stat().st_size != size:
            reason = "Stage 7A external checkpoint size differs from its declaration"
            raise ValueError(reason)
        if _sha256(candidate) != digest:
            reason = "Stage 7A external checkpoint hash differs from its declaration"
            raise ValueError(reason)
    return 1


def _verify_legacy_manifest(  # noqa: C901
    package_root: Path,
    *,
    stage: str,
    allow_external_checkpoint: bool,
) -> EvidenceDocument:
    manifest_path = package_root / "artifact_manifest.json"
    if not manifest_path.is_file():
        reason = f"missing artifact manifest: {manifest_path}"
        raise ValueError(reason)
    manifest = _read_document(manifest_path)
    verified_paths: set[str] = set()
    external_entries = (
        _stage7_declared_checkpoint(package_root, manifest) if allow_external_checkpoint else 0
    )
    for entry in _manifest_entries(manifest):
        raw_relative = _entry_path(entry)
        external_checkpoint = allow_external_checkpoint and _stage7_external_checkpoint(
            raw_relative, entry
        )
        raw_path = Path(raw_relative)
        if raw_path.is_absolute() or ".." in raw_path.parts:
            if external_checkpoint:
                _ = _entry_size(entry)
                _ = _entry_digest(entry)
                external_entries += 1
                continue
            reason = f"manifest artifact path must stay inside the package: {raw_relative}"
            raise ValueError(reason)
        relative = _normalize_package_path(package_root, raw_relative)
        candidate = _safe_candidate(package_root, relative)
        if not candidate.is_file():
            if external_checkpoint:
                _ = _entry_size(entry)
                _ = _entry_digest(entry)
                external_entries += 1
                continue
            reason = f"manifest artifact does not exist: {relative}"
            raise ValueError(reason)
        size = _entry_size(entry)
        if size is not None and candidate.stat().st_size != size:
            reason = f"artifact size mismatch: {relative}"
            raise ValueError(reason)
        if _sha256(candidate) != _entry_digest(entry):
            reason = f"artifact hash mismatch: {relative}"
            raise ValueError(reason)
        verified_paths.add(candidate.relative_to(package_root.resolve()).as_posix())
    actual_paths = {
        item.relative_to(package_root).as_posix()
        for item in package_root.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if actual_paths != verified_paths:
        reason = "artifact membership differs from the legacy manifest"
        raise ValueError(reason)
    if not verified_paths:
        reason = f"Stage {stage} manifest did not verify any committed artifacts"
        raise ValueError(reason)
    source = next(
        (
            value
            for key in (
                "source_commit",
                "git_commit",
                "candidate_commit",
                "candidate_git_commit_sha",
            )
            if isinstance((value := manifest.get(key)), str) and value
        ),
        None,
    )
    return {
        "stage": stage,
        "source_commit": source,
        "verified_artifacts": len(verified_paths),
        "external_artifacts": external_entries,
    }


def _stage7_verifier(package_root: Path) -> EvidenceDocument:
    return _verify_legacy_manifest(
        package_root,
        stage="7A",
        allow_external_checkpoint=True,
    )


def _stage8_verifier(package_root: Path) -> EvidenceDocument:
    return _verify_legacy_manifest(
        package_root,
        stage="8",
        allow_external_checkpoint=False,
    )


def _module_verifier(module_name: str, function_name: str) -> EvidenceVerifier:
    def verify(package_root: Path) -> EvidenceDocument:
        module = importlib.import_module(module_name)
        raw_target = cast("object", getattr(module, function_name, None))
        if not callable(raw_target):
            reason = f"{module_name}.{function_name} is not callable"
            raise TypeError(reason)
        target = cast("Callable[[Path], object]", raw_target)
        result = target(package_root)
        if not isinstance(result, dict):
            reason = f"{module_name}.{function_name} must return a mapping"
            raise TypeError(reason)
        document = cast("dict[object, object]", result)
        if any(not isinstance(key, str) for key in document):
            reason = f"{module_name}.{function_name} must return string keys"
            raise TypeError(reason)
        return cast("EvidenceDocument", document)

    return verify


def evidence_registry() -> tuple[EvidencePackage, ...]:
    """Return the immutable Stage 7A-20 evidence registry in release order."""
    modern = (
        ("9", "kv-cache-generation", "stage9_evidence", "verify_stage9_evidence"),
        ("10", "serving-control-plane", "stage10_evidence", "verify_stage10_evidence"),
        ("11A", "decode-continuous-batching", "stage11a_evidence", "verify_stage11a_evidence"),
        ("11B", "batched-prefill", "stage11b_evidence", "verify_stage11b_evidence"),
        ("12", "http-serving", "stage12_evidence", "verify_stage12_evidence"),
        ("13A", "paged-kv-cache-manager", "stage13a_evidence", "verify_stage13a_evidence"),
        ("13B", "paged-attention", "stage13b_evidence", "verify_stage13b_evidence"),
        ("14", "automatic-prefix-caching", "stage14_evidence", "verify_stage14_evidence"),
        ("15", "cache-aware-batched-prefill", "stage15_evidence", "verify_stage15_evidence"),
        ("16", "chunked-prefill-token-budget", "stage16_evidence", "verify_stage16_evidence"),
        ("17", "kv-pressure-preemption", "stage17_evidence", "verify_stage17_evidence"),
        ("18", "lazy-kv-reservation", "stage18_evidence", "verify_stage18_evidence"),
        (
            "19",
            "serving-runtime-configuration",
            "stage19_evidence",
            "verify_stage19_evidence",
        ),
        (
            "20",
            "project-doctor",
            "stage20_evidence",
            "verify_stage20_evidence",
        ),
    )
    packages: list[EvidencePackage] = [
        EvidencePackage(
            stage="7A",
            slug="reference-training",
            relative_root=Path("docs/results/reference-training"),
            verifier=_stage7_verifier,
        ),
        EvidencePackage(
            stage="8",
            slug="batcher-optimization",
            relative_root=Path("docs/results/batcher-optimization"),
            verifier=_stage8_verifier,
        ),
    ]
    packages.extend(
        EvidencePackage(
            stage=stage,
            slug=slug,
            relative_root=Path("docs/results") / slug,
            verifier=_module_verifier(f"minigpt.{module}", function_name),
            merged_commit=_SQUASH_MERGED_COMMITS.get(stage),
        )
        for stage, slug, module, function_name in modern
    )
    return tuple(packages)
