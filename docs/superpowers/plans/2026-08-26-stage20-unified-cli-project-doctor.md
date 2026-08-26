# Stage 20 Implementation Plan

## Goal

Deliver an installable, lazily imported unified CLI and a deterministic project doctor that verifies the complete Stage 7A–19 evidence chain without changing existing model/runtime semantics.

## Work packages

1. **Packaging boundary**
   - add one source version module;
   - register `minigpt = minigpt.cli:main`;
   - include existing root command modules in wheel/sdist;
   - support `python -m minigpt`.
2. **Typed command dispatcher**
   - explicit immutable command registry;
   - stable help/version;
   - lazy optional imports;
   - preserve existing command `main(argv)` contracts.
3. **Evidence registry**
   - explicit Stage 7A–19 order;
   - modern verifier adapters;
   - legacy Stage 7A/8 exact local membership;
   - declared Stage 7A external checkpoint exception only.
4. **Project doctor**
   - version, docs, evidence, ancestry, canonical config checks;
   - stable JSON and structured failures;
   - quick/CI modes;
   - optional clean-worktree requirement.
5. **Runtime verification**
   - canonical Stage 18 simulation in temporary storage;
   - Stage 19 real runtime construction with tiny deterministic model;
   - installed module CLI subprocess.
6. **Tests and evidence**
   - dispatch/import/metadata tests;
   - tamper, path, ancestry, version, dirty-tree tests;
   - Stage 20 hash-bound evidence package;
   - fresh wheel/sdist install probe.
7. **CI and documentation**
   - full-history checkout;
   - doctor CI gate;
   - README, AGENTS, and project overview updates.

## Required gates

```text
python -m pip check
ruff format src tests
ruff format --check src tests
ruff check src tests
basedpyright
pytest
python -m minigpt verify --mode ci
```

A detached fresh checkout must build/install the distribution, run CLI help/version, pass the focused Stage 19–20 suite, and verify all registered evidence/source ancestry.

## Commit boundaries

1. source, tests, design, CI, and documentation;
2. generated Stage 20 evidence bound to commit 1;
3. review fixes, if any, followed by evidence regeneration when source contracts change.

Do not push, merge, or begin Stage 21 until Stage 20 passes independent review.
