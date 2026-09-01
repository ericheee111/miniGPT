# Post-v1 Story Forge data — implementation plan

Scope: the deterministic SimpleStories data layer only. No model/training/
serving/web changes. No push or merge. No new `tests/test_*.py` file.

## Deliverables

1. `src/minigpt/story_data.py` — constants, schema validation, label mapping,
   split/rank, two-pass streaming selection, quota redistribution, framing,
   encoding, metadata, atomic output.
2. `prepare_stories.py` — thin CLI wrapper.
3. `src/minigpt/cli.py` — lazy `prepare-stories` command.
4. `pyproject.toml` — `[story]` extra and `prepare_stories` py-module.
5. Tests appended to `tests/test_data.py` and `tests/test_cli.py`.

## Execution order

1. Pin the reviewer-verified upstream constants and SHA-256.
2. Implement label mapping and deterministic split/rank helpers as pure
   functions (no I/O), test them directly.
3. Implement Parquet schema validation and two-pass streaming selection using
   selected columns only, no pandas, no full materialization.
4. Implement quota redistribution and the actionable insufficient-capacity error.
5. Integrate BPE training, framing, `uint16` encoding, and metadata.
6. Implement the rollback-safe atomic directory swap.
7. Wire the CLI and packaging, then append fixture-based tests.

## Verification

- Focused `tests/test_data.py` and `tests/test_cli.py`.
- `ruff format` + `ruff check`, `basedpyright`, full `pytest`,
  `python -m minigpt verify --mode quick`, `git diff --check`.

## Acceptance

- Deterministic rerun produces byte-identical artifacts in two output dirs.
- Splits and selection are input-order independent and train/val disjoint.
- Metadata carries no absolute path or timestamp and its hashes validate.
- CLI help is lazy and reports the `[story]` install hint on missing backends.