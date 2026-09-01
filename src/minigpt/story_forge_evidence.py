"""Hash-bound Story Forge model, evaluation, and serving evidence.

The Story Forge evidence package is intentionally **outside** the Stage 7A-20
doctor registry (mirroring the Stage 21 capstone) so it can verify its own
training/evaluation/evidence loop without self-registration.

It consumes three reporter documents produced by the ``.[story]`` toolchain and
a serving-smoke document, then binds them to:

- the exact source commit,
- the canonical tokenizer/data hashes,
- the model parameter count,
- the bounded training trajectory summary,
- the evaluator outputs and samples,
- the local serving smoke result, and
- a bounded ``descriptive_only`` claim policy.

No production, general-chat, semantic-understanding, or universal speedup claim
is made. Checkpoint and canonical data remain external/ignored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Never, TypeAlias, cast

from typing_extensions import override

from minigpt.data import JsonValue
from minigpt.model import expected_gpt_parameter_count
from minigpt.settings import GPTConfig

if TYPE_CHECKING:
    from pathlib import Path

EvidenceDocument: TypeAlias = dict[str, JsonValue]
STAGE_NAME: Final = "story-forge-model"
_STORY_MODEL_PARAMETER_COUNT: Final = 4_928_144
_STORY_VOCAB_SIZE: Final = 4096
_STORY_BLOCK_SIZE: Final = 512

# Canonical data identity (byte-bound SHA-256 of the ignored artifacts).
CANONICAL_TOKENIZER_SHA256: Final = (
    "7c897e0d51d135f6bb24bdbd18b0e40db88c0e62defcff9d23f3a204e092585c"
)
CANONICAL_TRAIN_SHA256: Final = "d7d979d1c55c6eb3596b6b36cfda344cfdc8db0856291d7421a92e657288a42e"
CANONICAL_VAL_SHA256: Final = "e40251b9d47484306b35ab714dedc541807a920a9e4fc59d0564a073d69a4b77"


@dataclass(slots=True)
class StoryForgeEvidenceVerificationError(ValueError):
    """Report invalid Story Forge evidence membership, hashes, or claims."""

    reason: str

    @override
    def __str__(self) -> str:
        """Render the evidence failure."""
        return f"invalid Story Forge evidence: {self.reason}"


def _invalid(reason: str) -> Never:
    raise StoryForgeEvidenceVerificationError(reason)


def sha256_file(path: Path) -> str:
    """Return the byte-bound SHA-256 of one artifact path."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> EvidenceDocument:
    if not path.is_file():
        _invalid(f"missing evidence document: {path}")
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        _invalid(f"{path} must contain a JSON object")
    return cast("EvidenceDocument", raw)


def _write_json(path: Path, document: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _copy_into(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _ = destination.write_bytes(source.read_bytes())


def _identity(trajectory: EvidenceDocument) -> EvidenceDocument:
    """Resolve the training/metrics identity from the metrics reporter."""
    # Validate strict ordering / no-duplicate invariant on the raw step list.
    step_rows = cast("list[JsonValue]", trajectory["steps"])
    observed: list[int] = []
    for raw in step_rows:
        if isinstance(raw, bool) or not isinstance(raw, int):
            _invalid("metrics step list must contain integers")
        observed.append(raw)
    if observed != list(range(len(observed))):
        _invalid("metrics trajectory is not strictly ordered with no duplicate steps")
    return {
        "parameter_count": _STORY_MODEL_PARAMETER_COUNT,
        "parameter_count_expected": expected_gpt_parameter_count(
            GPTConfig(
                vocab_size=_STORY_VOCAB_SIZE,
                block_size=_STORY_BLOCK_SIZE,
                n_layer=6,
                n_head=4,
                n_embd=208,
                dropout=0.1,
                bias=False,
            )
        ),
        "vocab_size": _STORY_VOCAB_SIZE,
        "final_step": trajectory["final_step"],
        "eval_points": trajectory["eval_points"],
        "tokenizer_sha256": trajectory["tokenizer_sha256"],
        "train_sha256": trajectory["train_sha256"],
        "val_sha256": trajectory["val_sha256"],
    }


def _case_bool(case: JsonValue, key: str) -> bool:
    if not isinstance(case, dict):
        _invalid("evaluation case entries must be objects")
    return cast("EvidenceDocument", case).get(key) is True


def _case_int(case: JsonValue, key: str) -> int:
    if not isinstance(case, dict):
        _invalid("evaluation case entries must be objects")
    value = cast("EvidenceDocument", case).get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(f"evaluation case field {key!r} must be an integer")
    return value


def _milestone(evaluation: EvidenceDocument) -> EvidenceDocument:
    """Extract a bounded milestone descriptor from one evaluation report."""
    identity = cast("EvidenceDocument", evaluation["identity"])
    validation = cast("EvidenceDocument", evaluation["validation"])
    return {
        "completed_step": identity["completed_step"],
        "val_loss": validation["mean_val_loss"],
        "val_perplexity": validation["val_perplexity"],
    }


def _bounded_summary(evaluation: EvidenceDocument, serving: EvidenceDocument) -> EvidenceDocument:
    """Fold evaluator and serving outputs into a bounded descriptor."""
    cases = cast("list[JsonValue]", evaluation["cases"])
    total = len(cases)
    eos_hits = sum(1 for case in cases if _case_bool(case, "eos_hit"))
    leaks = sum(_case_int(case, "special_token_leaks") for case in cases)
    invalid = sum(1 for case in cases if _case_bool(case, "invalid_decode"))
    max_loop = max(_case_int(case, "longest_loop") for case in cases)
    return {
        "case_count": total,
        "sampling": evaluation["sampling"],
        "eos_hits": eos_hits,
        "special_token_leaks": leaks,
        "invalid_decode": invalid,
        "max_immediate_loop": max_loop,
        "determinism": evaluation["determinism"],
        "diversity": evaluation["diversity"],
        "cached_uncached": evaluation["cached_uncached"],
        "serving_smoke": serving,
    }


def generate_story_forge_evidence(  # noqa: PLR0913, PLR0917
    metrics_reporter: Path,
    final_evaluation: Path,
    milestone_0500_evaluation: Path,
    milestone_1500_evaluation: Path,
    serving_smoke: Path,
    package_root: Path,
    source_commit: str,
) -> Path:
    """Assemble a hash-bound Story Forge evidence package and self-verify it."""
    if not source_commit:
        _invalid("source_commit must be non-empty")

    metrics = _read_json(metrics_reporter)
    evaluation = _read_json(final_evaluation)
    milestone_0500 = _read_json(milestone_0500_evaluation)
    milestone_1500 = _read_json(milestone_1500_evaluation)
    serving = _read_json(serving_smoke)

    # Canonical identity gate: the ignored data artifacts must match the pinned hashes.
    if metrics["tokenizer_sha256"] != CANONICAL_TOKENIZER_SHA256:
        _invalid("tokenizer identity is not the canonical Story Forge artifact")
    if metrics["train_sha256"] != CANONICAL_TRAIN_SHA256:
        _invalid("train identity is not the canonical Story Forge artifact")
    if metrics["val_sha256"] != CANONICAL_VAL_SHA256:
        _invalid("val identity is not the canonical Story Forge artifact")

    identity = _identity(metrics)

    evaluation_identity = cast("EvidenceDocument", evaluation["identity"])
    checkpoint_sha256 = cast("str", evaluation_identity["checkpoint_sha256"])
    checkpoint_bytes = cast("int", evaluation_identity["checkpoint_bytes"])
    checkpoint_basename = cast("str", metrics["checkpoint_basename"])
    checkpoint_note = cast("str", metrics["checkpoint_note"])

    summary: EvidenceDocument = {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "source_commit": source_commit,
        "model": {
            "vocab_size": identity["vocab_size"],
            "n_layer": 6,
            "n_head": 4,
            "n_embd": 208,
            "block_size": _STORY_BLOCK_SIZE,
            "bias": False,
            "parameter_count": identity["parameter_count"],
        },
        "data_identity": {
            "tokenizer_sha256": identity["tokenizer_sha256"],
            "train_sha256": identity["train_sha256"],
            "val_sha256": identity["val_sha256"],
        },
        "training": {
            "final_step": identity["final_step"],
            "eval_points": identity["eval_points"],
            "final_val_loss": metrics["final_val_loss"],
            "final_val_perplexity": metrics["final_val_perplexity"],
            "max_steps": metrics["max_steps"],
            "intentionally_resumed": metrics["intentionally_resumed"],
        },
        "checkpoint": {
            "basename": checkpoint_basename,
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
            "note": checkpoint_note,
        },
        "final_evaluation": _bounded_summary(evaluation, serving),
        "milestones": {
            "step-0500": _milestone(milestone_0500),
            "step-1500": _milestone(milestone_1500),
        },
        "claim_policy": {
            "verdict": "descriptive_only",
            "production_claim": False,
            "general_chat_claim": False,
            "semantic_understanding_claim": False,
            "universal_speedup_claim": False,
        },
    }
    _write_json(package_root / "summary.json", summary)

    # Bind the reporter documents verbatim for auditability.
    _copy_into(metrics_reporter, package_root / "metrics_reporter.json")
    _copy_into(final_evaluation, package_root / "evaluation_final.json")
    _copy_into(milestone_0500_evaluation, package_root / "evaluation_step-0500.json")
    _copy_into(milestone_1500_evaluation, package_root / "evaluation_step-1500.json")
    _copy_into(serving_smoke, package_root / "serving_smoke.json")

    readme = _readme(summary, evaluation)
    _ = (package_root / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    manifest_file = package_root / "artifact_manifest.json"
    artifacts: list[JsonValue] = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path != manifest_file
    ]
    _write_json(
        manifest_file,
        {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "source_commit": source_commit,
            "artifacts": artifacts,
        },
    )
    _ = verify_story_forge_evidence(package_root)
    return package_root


def _manifest_entries(entries: list[object]) -> dict[str, tuple[int, str]]:
    expected: dict[str, tuple[int, str]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            _invalid("manifest artifact entries must be objects")
        entry = cast("dict[str, object]", raw_entry)
        relative = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            _invalid("manifest artifact entry fields are invalid")
        if relative in expected:
            _invalid(f"duplicate manifest path {relative}")
        expected[relative] = (size, digest)
    return expected


def _verify_artifact_manifest(package_root: Path) -> tuple[EvidenceDocument, str]:
    manifest_file = package_root / "artifact_manifest.json"
    manifest = _read_json(manifest_file)
    if manifest.get("stage") != STAGE_NAME:
        _invalid(f"manifest stage must be {STAGE_NAME}")
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
        if path.is_file() and path != manifest_file
    }
    if actual != set(expected):
        _invalid("artifact membership differs from manifest")
    for relative, (size, digest) in expected.items():
        path = package_root / relative
        if path.stat().st_size != size or sha256_file(path) != digest:
            _invalid(f"artifact hash mismatch for {relative}")
    return manifest, source_commit


def _verify_summary(summary: EvidenceDocument, source_commit: str) -> None:
    if summary.get("source_commit") != source_commit:
        _invalid("summary source_commit differs from manifest")
    if summary.get("stage") != STAGE_NAME:
        _invalid(f"summary stage must be {STAGE_NAME}")
    model = cast("EvidenceDocument", summary.get("model"))
    if model.get("parameter_count") != _STORY_MODEL_PARAMETER_COUNT:
        _invalid("model parameter count differs from the reviewed 5M value")
    claim_policy = cast("EvidenceDocument", summary.get("claim_policy"))
    if claim_policy.get("verdict") != "descriptive_only":
        _invalid("claim policy verdict must be descriptive_only")
    for key in (
        "production_claim",
        "general_chat_claim",
        "semantic_understanding_claim",
        "universal_speedup_claim",
    ):
        if claim_policy.get(key) is not False:
            _invalid(f"claim policy {key} must be false")
    data_identity = cast("EvidenceDocument", summary.get("data_identity"))
    if data_identity.get("tokenizer_sha256") != CANONICAL_TOKENIZER_SHA256:
        _invalid("summary tokenizer identity is not canonical")
    if data_identity.get("train_sha256") != CANONICAL_TRAIN_SHA256:
        _invalid("summary train identity is not canonical")
    if data_identity.get("val_sha256") != CANONICAL_VAL_SHA256:
        _invalid("summary val identity is not canonical")


def verify_story_forge_evidence(package_root: Path) -> EvidenceDocument:
    """Verify exact membership, hashes, source identity, and bounded claims."""
    manifest, source_commit = _verify_artifact_manifest(package_root)
    _verify_summary(_read_json(package_root / "summary.json"), source_commit)
    return manifest


def _readme(summary: EvidenceDocument, evaluation: EvidenceDocument) -> str:
    sampling = cast("EvidenceDocument", evaluation["sampling"])
    training = cast("EvidenceDocument", summary["training"])
    final_eval = cast("EvidenceDocument", summary["final_evaluation"])
    model = cast("EvidenceDocument", summary["model"])
    source_commit = cast("str", summary["source_commit"])
    return (
        "\n".join(
            (
                "# Story Forge — model, evaluation, and local serving evidence",
                "",
                "This package records the 5M-parameter Story Forge training completion,",
                "bounded deterministic evaluation over the 16-case battery, milestone",
                "evaluations, and an isolated local serving smoke. It is a post-v1 research",
                "extension and does not alter historical Stage 7A-21 Evidence.",
                "",
                f"Source commit: {source_commit}.",
                f"Parameter count: {model['parameter_count']}.",
                (
                    f"Final step: {training['final_step']}; final validation loss "
                    f"{training['final_val_loss']:.4f} "
                    f"(perplexity {training['final_val_perplexity']:.2f})."
                ),
                f"Sampling: temperature {sampling['temperature']}, top_k {sampling['top_k']}.",
                (
                    f"Cases: {final_eval['case_count']}; EOS hits {final_eval['eos_hits']}; "
                    f"special-token leaks {final_eval['special_token_leaks']}; "
                    f"max immediate loop {final_eval['max_immediate_loop']}."
                ),
                "",
                "## Claim policy",
                "",
                "The verdict is descriptive_only. Objective metrics are bounded lexical and",
                "distributional proxies; they do not measure semantic understanding, story",
                "quality, or authorship. No production readiness, general-chat, or universal",
                "speedup claim is made. Checkpoint and canonical data remain external/ignored.",
            )
        )
        + "\n"
    )
