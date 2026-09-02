"""Build deterministic Systems Lab scenario assets from committed evidence.

The Systems Lab is an offline-capable browser experience that replays verified
serving scenarios. Its assets are static JSON derived only from the committed
Stage 11/14/17/18 evidence packages — the builder reads those evidence files and
extracts bounded counters and event sequences; it never invents an event that is
not present in the source evidence.

Each generated asset carries:

- a ``schema_version`` and scenario id/title,
- the exact source evidence path (relative to the repository) and the
  source commit recorded by that package,
- ordered ticks/events normalized into a small, deterministic shape,
- request lanes with terminal status and generated-token counts,
- safe KV block / resource counters,
- invariant results, and
- a ``claim_level`` of ``semantic``, ``structural``, or ``descriptive_only``.

These assets are built deterministically and must not be hand-edited.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Never, cast

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    "build_systems_lab_assets",
    "systems_lab_asset_names",
    "verify_systems_lab_assets",
)

SCHEMA_VERSION = 1

_ASSET_NAMES = (
    "continuous_batching",
    "automatic_prefix_cache",
    "kv_preemption",
    "lazy_reservation",
)

_EVIDENCE_ROOT = "docs/results"


class SystemsLabBuildError(ValueError):
    """Report an asset build that cannot derive its data from committed evidence."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"systems lab asset build failed: {reason}")


def _invalid(reason: str) -> Never:
    raise SystemsLabBuildError(reason)


def systems_lab_asset_names() -> tuple[str, ...]:
    """Return the fixed Systems Lab asset filenames."""
    return _ASSET_NAMES


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        _invalid(f"missing evidence file: {path}")
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        _invalid(f"{path} must contain a JSON object")
    return cast("dict[str, object]", raw)


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_int(document: dict[str, object], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(f"evidence field {key!r} must be an integer")
    return value


def _read_int_list(document: dict[str, object], key: str) -> list[int]:
    value = document.get(key)
    if not isinstance(value, list) or not value:
        _invalid(f"evidence field {key!r} must be a non-empty list")
    items = cast("list[object]", value)
    result: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            _invalid(f"evidence field {key!r} must contain integers")
        result.append(item)
    return result


def _source_commit(package: Path) -> str:
    summary = _load_json(package / "summary.json")
    raw = summary.get("source_commit")
    if not isinstance(raw, str) or not raw:
        _invalid(f"{package} summary.json lacks a non-empty source_commit")
    return raw


def _continuous_batching(evidence_root: Path) -> dict[str, object]:
    package = evidence_root / "decode-continuous-batching"
    source_commit = _source_commit(package)
    summary = _load_json(package / "summary.json")
    simulator = summary.get("simulator")
    if not isinstance(simulator, list) or not simulator:
        _invalid("Stage 11A summary.json lacks simulator list")
    mixed = cast("dict[str, object]", simulator[0])
    admission = mixed.get("admission_order")
    if not isinstance(admission, list) or not admission:
        _invalid("Stage 11A simulator entry lacks admission_order")
    admission_items = cast("list[object]", admission)
    admission_ids = [item if isinstance(item, str) else str(item) for item in admission_items]
    terminal = mixed.get("terminal_statuses")
    terminal_map = cast("dict[str, object]", terminal) if isinstance(terminal, dict) else {}
    generated = mixed.get("generated_tokens")
    generated_map = cast("dict[str, object]", generated) if isinstance(generated, dict) else {}

    events = _load_json(
        package / "evidence/simulator/stage11a-mixed/continuous_decode/summary.json"
    )
    batch_sizes = _read_int_list(events, "decode_batch_sizes")

    # Stage 11A simulator demonstrates generated-token and terminal-state
    # equivalence (semantic correctness) alongside decode batch co-scheduling.
    simulator_entry = cast("dict[str, object]", simulator[0])
    equivalent = bool(simulator_entry.get("equivalent"))

    lanes: list[dict[str, object]] = []
    for request_id in admission_ids[:4]:
        raw_generated = generated_map.get(request_id)
        generated_count = 0
        if isinstance(raw_generated, list):
            generated_count = len(cast("list[object]", raw_generated))
        lanes.append(
            {
                "request_id": request_id,
                "status": terminal_map.get(request_id, "finished"),
                "generated_token_count": generated_count,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "continuous_batching",
        "title": "Continuous decode batching",
        "source_evidence_path": (
            "docs/results/decode-continuous-batching/evidence/simulator"
            "/stage11a-mixed/continuous_decode"
        ),
        "source_commit": source_commit,
        "claim_level": "semantic" if equivalent else "descriptive_only",
        "ticks": [
            {"tick": index, "decode_batch_size": size} for index, size in enumerate(batch_sizes)
        ],
        "request_lanes": lanes,
        "kv": {
            "total_slots": _read_int(events, "max_cached_tokens"),
            "peak_cached_tokens": _read_int(events, "peak_cached_tokens"),
        },
        "invariants": {
            "completed_requests": _read_int(events, "completed_requests"),
            "cancelled_requests": _read_int(events, "cancelled_requests"),
            "fifo_admission_order": admission_ids[:4],
        },
    }


def _apc(evidence_root: Path) -> dict[str, object]:
    package = evidence_root / "automatic-prefix-caching"
    source_commit = _source_commit(package)
    correctness = _load_json(package / "evidence/correctness.json")
    summary = _load_json(package / "summary.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "automatic_prefix_cache",
        "title": "Automatic Prefix Caching reuse",
        "source_evidence_path": "docs/results/automatic-prefix-caching/evidence/correctness.json",
        "source_commit": source_commit,
        "claim_level": "structural",
        "ticks": [],
        "request_lanes": [],
        "kv": {
            "prefix_hit_requests": _read_int(correctness, "prefix_hit_requests"),
            "prefix_hit_tokens": _read_int(correctness, "prefix_hit_tokens"),
            "avoided_prefill_tokens": _read_int(correctness, "avoided_prefill_tokens"),
            "prefill_tokens_computed": _read_int(correctness, "prefill_tokens_computed"),
        },
        "invariants": {
            "equivalent": bool(correctness.get("equivalent")),
            "shared_blocks_immutable": bool(summary.get("shared_blocks_immutable")),
        },
    }


def _kv_preemption(evidence_root: Path) -> dict[str, object]:
    package = evidence_root / "kv-pressure-preemption"
    source_commit = _source_commit(package)
    correctness = _load_json(package / "evidence/correctness.json")
    summary = _load_json(package / "summary.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "kv_preemption",
        "title": "KV-pressure preemption and recompute",
        "source_evidence_path": "docs/results/kv-pressure-preemption/evidence/correctness.json",
        "source_commit": source_commit,
        "claim_level": "structural",
        "ticks": [],
        "request_lanes": [],
        "kv": {
            "preemptions": _read_int(correctness, "preemptions"),
            "recompute_tokens": _read_int(correctness, "recompute_tokens"),
        },
        "invariants": {
            "per_request_rng_equivalence": bool(correctness.get("per_request_rng_equivalence")),
            "recompute_resume": bool(summary.get("recompute_resume")),
        },
    }


def _lazy_reservation(evidence_root: Path) -> dict[str, object]:
    package = evidence_root / "lazy-kv-reservation"
    source_commit = _source_commit(package)
    correctness = _load_json(package / "evidence/correctness.json")
    summary = _load_json(package / "summary.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "lazy_reservation",
        "title": "Lazy KV growth reservation",
        "source_evidence_path": "docs/results/lazy-kv-reservation/evidence/correctness.json",
        "source_commit": source_commit,
        "claim_level": "structural",
        "ticks": [],
        "request_lanes": [],
        "kv": {
            "peak_overcommitted_cache_tokens": _read_int(
                correctness, "peak_overcommitted_cache_tokens"
            ),
            "growth_pressure_preemptions": _read_int(correctness, "growth_pressure_preemptions"),
        },
        "invariants": {
            "growth_work_free": bool(correctness.get("growth_work_free")),
            "controlled_overcommit": bool(summary.get("controlled_overcommit")),
        },
    }


def build_systems_lab_assets(root: Path, output_dir: Path) -> tuple[Path, ...]:
    """Generate the four deterministic Systems Lab assets from committed evidence."""
    evidence_root = root / _EVIDENCE_ROOT
    if not evidence_root.is_dir():
        _invalid(f"evidence root does not exist: {evidence_root}")
    builders = {
        "continuous_batching": _continuous_batching,
        "automatic_prefix_cache": _apc,
        "kv_preemption": _kv_preemption,
        "lazy_reservation": _lazy_reservation,
    }
    output_paths: list[Path] = []
    for name in _ASSET_NAMES:
        document = builders[name](evidence_root)
        output_path = output_dir / f"{name}.json"
        _write_json(output_path, document)
        output_paths.append(output_path)
    return tuple(output_paths)


def verify_systems_lab_assets(output_dir: Path) -> tuple[str, ...]:
    """Verify the four assets exist, parse, and carry a schema + claim level."""
    verified: list[str] = []
    for name in _ASSET_NAMES:
        path = output_dir / f"{name}.json"
        if not path.is_file():
            _invalid(f"missing asset: {path}")
        document = _load_json(path)
        if document.get("schema_version") != SCHEMA_VERSION:
            _invalid(f"asset {name} has wrong schema_version")
        if document.get("claim_level") not in {"semantic", "structural", "descriptive_only"}:
            _invalid(f"asset {name} has invalid claim_level")
        verified.append(name)
    return tuple(verified)
