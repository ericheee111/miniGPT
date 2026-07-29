"""Load exact, versioned policy for Benchmark v2 regression decisions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

import yaml
from typing_extensions import override

if TYPE_CHECKING:
    from pathlib import Path

PolicyValue: TypeAlias = (
    str | int | float | bool | list["PolicyValue"] | dict[str, "PolicyValue"] | None
)
PolicyMapping: TypeAlias = dict[str, PolicyValue]

_SCHEMA_VERSION = 1
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "minimum_successful_replicates",
        "max_cv_percent",
        "regression_threshold_percent",
        "require_equal_replicate_count",
    }
)


@dataclass(frozen=True, slots=True)
class ComparisonPolicy:
    """Bind authoritative comparison controls to the exact policy file bytes."""

    schema_version: int
    minimum_successful_replicates: int
    max_cv_percent: float
    regression_threshold_percent: float
    require_equal_replicate_count: bool
    sha256: str
    source_path: Path


@dataclass(slots=True)
class InvalidComparisonPolicyError(ValueError):
    """Report an unreadable, malformed, or unsupported comparison policy."""

    source: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Render the policy path and exact rejection reason."""
        return f"invalid Benchmark v2 comparison policy {self.source}: {self.reason}"


class _DuplicateKeySafeLoader(yaml.SafeLoader):
    """Reject duplicate YAML mapping keys before strict schema validation."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Construct one YAML mapping while rejecting duplicate and non-string keys."""
    if not isinstance(node, yaml.MappingNode):
        msg = "expected a YAML mapping node"
        raise yaml.YAMLError(msg)
    mapping: dict[str, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            msg = "YAML mapping keys must be strings"
            raise yaml.YAMLError(msg)
        if key in mapping:
            msg = f"duplicate YAML mapping key {key!r}"
            raise yaml.YAMLError(msg)
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_DuplicateKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(value: object, source: Path) -> PolicyMapping:
    """Require the policy document to be one string-keyed mapping."""
    if not isinstance(value, dict):
        raise InvalidComparisonPolicyError(source, "top-level YAML value must be a mapping")
    mapping = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        raise InvalidComparisonPolicyError(source, "YAML mapping keys must be strings")
    return cast("PolicyMapping", mapping)


def _require_exact_keys(document: PolicyMapping, source: Path) -> None:
    """Reject all omitted and unknown policy controls."""
    actual = set(document)
    missing = _POLICY_KEYS - actual
    unexpected = actual - _POLICY_KEYS
    if missing:
        raise InvalidComparisonPolicyError(source, f"top-level policy missing key {min(missing)!r}")
    if unexpected:
        raise InvalidComparisonPolicyError(
            source, f"top-level policy has unexpected key {min(unexpected)!r}"
        )


def _positive_integer(document: PolicyMapping, key: str, source: Path) -> int:
    """Read one strict positive integer policy control."""
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidComparisonPolicyError(source, f"{key} must be a positive integer")
    return value


def _positive_number(document: PolicyMapping, key: str, source: Path) -> float:
    """Read one finite positive numeric policy control."""
    value = document[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise InvalidComparisonPolicyError(source, f"{key} must be positive and finite")
    return float(value)


def load_comparison_policy(path: Path) -> ComparisonPolicy:
    """Load strict schema-v1 policy and hash the exact source bytes."""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise InvalidComparisonPolicyError(path, str(error)) from error
    try:
        raw_document = yaml.load(
            content.decode("utf-8"),
            Loader=_DuplicateKeySafeLoader,  # noqa: S506
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise InvalidComparisonPolicyError(path, str(error)) from error
    document = _mapping(raw_document, path)
    _require_exact_keys(document, path)
    schema_version = document["schema_version"]
    if isinstance(schema_version, bool) or schema_version != _SCHEMA_VERSION:
        raise InvalidComparisonPolicyError(
            path, f"schema_version must be integer {_SCHEMA_VERSION}"
        )
    require_equal = document["require_equal_replicate_count"]
    if not isinstance(require_equal, bool):
        raise InvalidComparisonPolicyError(path, "require_equal_replicate_count must be a boolean")
    return ComparisonPolicy(
        schema_version=_SCHEMA_VERSION,
        minimum_successful_replicates=_positive_integer(
            document, "minimum_successful_replicates", path
        ),
        max_cv_percent=_positive_number(document, "max_cv_percent", path),
        regression_threshold_percent=_positive_number(
            document, "regression_threshold_percent", path
        ),
        require_equal_replicate_count=require_equal,
        sha256=hashlib.sha256(content).hexdigest(),
        source_path=path.resolve(),
    )


def comparison_policy_document(policy: ComparisonPolicy) -> dict[str, PolicyValue]:
    """Return the authoritative semantic policy summary in stable schema order."""
    return {
        "schema_version": policy.schema_version,
        "minimum_successful_replicates": policy.minimum_successful_replicates,
        "max_cv_percent": policy.max_cv_percent,
        "regression_threshold_percent": policy.regression_threshold_percent,
        "require_equal_replicate_count": policy.require_equal_replicate_count,
    }
