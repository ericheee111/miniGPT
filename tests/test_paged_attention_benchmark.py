from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from minigpt.paged_attention_benchmark import (
    PagedAttentionBenchmarkConfig,
    write_paged_attention_benchmark,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_paged_attention_benchmark_is_descriptive_and_logically_equivalent(tmp_path: Path) -> None:
    # Given: a bounded smoke matrix with both paged decode strategies.
    output = tmp_path / "benchmark.json"

    # When: the real CPU benchmark executes fresh engines.
    _ = write_paged_attention_benchmark(
        output,
        config=PagedAttentionBenchmarkConfig(
            warmups=0,
            repeats=2,
            cache_access_iterations=5,
        ),
    )
    document = cast(
        "dict[str, object]",
        json.loads(output.read_text(encoding="utf-8")),
    )

    # Then: timings are present, tokens matched internally, and no speedup is claimed.
    assert document["verdict"] == "descriptive_only"
    assert document["speedup_claim"] is False
    strategies = cast("dict[str, object]", document["strategies"])
    assert set(strategies) == {"materialized", "direct"}
    for raw in strategies.values():
        strategy = cast("dict[str, object]", raw)
        assert strategy["all_resources_released"] is True
        e2e = cast("dict[str, object]", strategy["e2e_seconds"])
        samples = cast("list[float]", e2e["samples"])
        assert len(samples) == 2
        assert all(sample > 0.0 for sample in samples)
    cache_access = cast("dict[str, object]", document["cache_access"])
    assert cast("float", cache_access["materialize_seconds"]) > 0.0
    assert cast("float", cache_access["request_view_seconds"]) > 0.0
