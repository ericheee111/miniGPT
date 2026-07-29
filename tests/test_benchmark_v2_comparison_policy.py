"""Test strict, versioned Benchmark v2 comparison policy parsing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from minigpt.benchmark_v2_comparison_policy import (
    InvalidComparisonPolicyError,
    load_comparison_policy,
)

_VALID_POLICY = """\
schema_version: 1
minimum_successful_replicates: 5
max_cv_percent: 5.0
regression_threshold_percent: 7.5
require_equal_replicate_count: true
"""


def test_canonical_comparison_policy_uses_conservative_initial_controls() -> None:
    # Given: the repository's versioned comparison policy.
    policy_path = Path(__file__).parents[1] / "configs" / "benchmark_v2_comparison.yaml"

    # When: it is loaded through the public strict parser.
    policy = load_comparison_policy(policy_path)

    # Then: provisional eligibility and regression controls are explicit and independent.
    assert policy.minimum_successful_replicates == 5
    assert policy.max_cv_percent == 5.0
    assert policy.regression_threshold_percent == 7.5
    assert policy.require_equal_replicate_count is True


def test_load_comparison_policy_binds_strict_values_and_source_hash(tmp_path: Path) -> None:
    # Given: one exact strict comparison policy file.
    policy_path = tmp_path / "policy.yaml"
    content = _VALID_POLICY.encode()
    _ = policy_path.write_bytes(content)

    # When: the policy is parsed.
    policy = load_comparison_policy(policy_path)

    # Then: semantic values and the exact source bytes are bound together.
    assert policy.schema_version == 1
    assert policy.minimum_successful_replicates == 5
    assert policy.max_cv_percent == 5.0
    assert policy.regression_threshold_percent == 7.5
    assert policy.require_equal_replicate_count is True
    assert policy.sha256 == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        ("schema_version: 1", "missing key"),
        ("schema_version: 2", "schema_version"),
        ("minimum_successful_replicates: true", "minimum_successful_replicates"),
        ("minimum_successful_replicates: 0", "minimum_successful_replicates"),
        ("max_cv_percent: .nan", "max_cv_percent"),
        ("regression_threshold_percent: 0", "regression_threshold_percent"),
        ("require_equal_replicate_count: 1", "require_equal_replicate_count"),
        ("require_equal_replicate_count: true\nextra: 1", "unexpected key"),
        (
            "minimum_successful_replicates: 5",
            "duplicate YAML mapping key",
        ),
    ],
)
def test_load_comparison_policy_rejects_invalid_schema(
    tmp_path: Path, replacement: str, reason: str
) -> None:
    # Given: one policy whose selected line violates the strict schema.
    policy_path = tmp_path / "invalid.yaml"
    if reason == "duplicate YAML mapping key":
        content = _VALID_POLICY.replace(
            "minimum_successful_replicates: 5",
            "minimum_successful_replicates: 5\nminimum_successful_replicates: 5",
        )
    elif replacement == "schema_version: 1":
        content = _VALID_POLICY.replace("schema_version: 1\n", "")
    else:
        field = replacement.split(":", maxsplit=1)[0]
        original = next(line for line in _VALID_POLICY.splitlines() if line.startswith(f"{field}:"))
        content = _VALID_POLICY.replace(original, replacement)
    _ = policy_path.write_text(content, encoding="utf-8")

    # When/Then: malformed or semantically invalid policy cannot authorize a verdict.
    with pytest.raises(InvalidComparisonPolicyError, match=reason):
        _ = load_comparison_policy(policy_path)


def test_comparison_policy_hash_changes_when_file_changes(tmp_path: Path) -> None:
    # Given: two valid policies differing only in the authoritative regression threshold.
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    _ = first_path.write_text(_VALID_POLICY, encoding="utf-8")
    _ = second_path.write_text(
        _VALID_POLICY.replace(
            "regression_threshold_percent: 7.5", "regression_threshold_percent: 8"
        ),
        encoding="utf-8",
    )

    # When: both exact files are loaded.
    first = load_comparison_policy(first_path)
    second = load_comparison_policy(second_path)

    # Then: their policy identities differ.
    assert first.sha256 != second.sha256
