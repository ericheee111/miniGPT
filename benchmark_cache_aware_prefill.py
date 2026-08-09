"""Run the Stage 15 cache-aware batched paged prefill benchmark."""

from pathlib import Path

from minigpt.cache_aware_prefill_benchmark import write_cache_aware_prefill_benchmark


def main() -> int:
    """Write fresh-process raw samples under the Stage 15 evidence directory."""
    _ = write_cache_aware_prefill_benchmark(
        Path("docs/results/cache-aware-batched-prefill/evidence/benchmark.json")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
