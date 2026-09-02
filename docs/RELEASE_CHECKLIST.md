# miniGPT v1.0.0 Release Checklist

This checklist is executable policy. Do not create or move the `v1.0.0` tag until every required item is satisfied by the exact main commit.

> **Current record:** Sections 1–7 have been completed for the reviewed v1.0 code/evidence baseline. The full local suite recorded **730 passed / 1 documented platform skip / 0 failed**, and both the feature branch and `main` passed Windows/Python 3.14 and Ubuntu/Python 3.11 quality jobs. Section 8 remains intentionally open because the annotated tag has not been created.

## 1. Source and metadata

- [x] `src/minigpt/_version.py` contains `1.0.0`.
- [x] `pyproject.toml` reads dynamic version metadata from `minigpt._version.__version__`.
- [x] Generated `*.egg-info` is ignored and not tracked.
- [x] `minigpt --version`, `python -m minigpt --version`, wheel metadata, and distribution metadata agree.
- [x] `CHANGELOG.md` contains the v1.0.0 capability and limitation summary.
- [x] `README.md`, `AGENTS.md`, `docs/PROJECT_OVERVIEW.md`, and `docs/PROJECT_COMPLETION.md` agree on the Stage 21 closure boundary.

## 2. Local quality gates

```powershell
python -m pip install -e ".[dev,report]"
python -m pip check
ruff format src tests
ruff format --check src tests
ruff check src tests
basedpyright
pytest
```

- [x] No Ruff formatting changes remain.
- [x] Ruff lint is clean.
- [x] basedpyright reports 0 errors, 0 warnings, and 0 notes.
- [x] Full pytest has zero failures; only documented platform skips are allowed.
- [x] `git diff --check` is clean.

## 3. Evidence and ancestry

- [x] Stage 7A–20 project-doctor registry passes.
- [x] Stage 7A external checkpoint is declared only through `sources.checkpoint`; artifact paths are local, unique, exact members.
- [x] Stage 17–21 package verifiers pass exact membership and SHA-256 checks.
- [x] Historical squash exceptions bind the exact reviewed source SHA to the merged `main` SHA.
- [x] Every modern evidence `source_commit` is an ancestor of release HEAD or matches an explicit reviewed squash mapping.
- [x] Repository history is not shallow during ancestry verification.
- [x] Stage 21 capstone evidence is bound to the reviewed source commit.
- [x] The Stage 21 verifier runs with the repository root, confirms its source is an ancestor of release HEAD, and matches every committed `tests/test_*.py` exactly once.
- [x] No evidence package claims unsupported wall-clock improvement.

## 4. Release artifacts

```powershell
minigpt verify --mode release --require-clean
```

- [x] Exactly one wheel and one sdist build from a clean checkout.
- [x] Wheel contains package modules and supported root command modules.
- [x] Fresh wheel installation succeeds with inherited `PYTHONPATH` removed.
- [x] Fresh `minigpt.__file__` resolves inside the wheel venv and distribution metadata equals `1.0.0`.
- [x] Fresh `pip check` succeeds using the provisioned dependency environment.
- [x] Fresh module and console-script help/version succeed.
- [x] Fresh installed quick doctor succeeds against the release checkout.

## 5. Functional capstone

- [x] Checkpoint v2 exact-resume focused suite passes.
- [x] Stage 18 canonical lazy-KV simulation completes with expected correctness contracts.
- [x] Stage 19 real serving runtime builds with paged/lazy configuration.
- [x] HTTP/SSE lifecycle and Windows Uvicorn subprocess tests pass.
- [x] Runtime manifest remains deterministic, atomic, and free of absolute paths/secrets.

## 6. Fresh checkout review

- [x] Detached fresh checkout has no tracked or untracked pollution.
- [x] Editable install and release artifact build succeed in the fresh checkout.
- [x] Focused Stage 19–21 tests pass independently of the development worktree.
- [x] Full evidence registry and source ancestry pass from the fresh checkout.

## 7. GitHub delivery

- [x] Feature branch is pushed.
- [x] Feature-branch Windows/Python 3.14 quality job succeeds.
- [x] Feature-branch Ubuntu/Python 3.11 quality job succeeds.
- [x] Reviewed commits are fast-forwarded or merged to `main` without content changes.
- [x] Main Windows/Python 3.14 quality job succeeds.
- [x] Main Ubuntu/Python 3.11 quality job succeeds.
- [x] `origin/main` equals the reviewed local HEAD.

## 8. Tagging

After all items above pass:

```powershell
git tag -a v1.0.0 -m "miniGPT v1.0.0"
git push origin v1.0.0
```

## Post-v1 Story Forge 1.1 delivery checklist

This extension preserves the Stage 21 / v1.0.0 evidence package and uses a separate two-commit product-evidence flow.

- [x] `src/minigpt/_version.py` is the single authored current-version source and reports `1.1.0`.
- [x] Story Forge tokenizer/data/model evidence is hash-bound without committing the external checkpoint or prepared arrays.
- [x] Story, Prediction, and Systems Lab source paths have focused tests and explicit claim boundaries.
- [x] The Story Forge launcher validates checkpoint/tokenizer hashes, fixed control IDs, model configuration, and cached/uncached output before startup.
- [x] Port 8001 is validated before public cutover; port 8000 remains the rollback target.
- [ ] Commit the reviewed Story Forge source and record its exact source SHA.
- [ ] Generate and independently verify `docs/results/story-forge-product/` against that source SHA.
- [ ] Repeat Ruff, basedpyright, full pytest, Project Doctor, package validation, static builds, and model/API smoke in a detached fresh worktree.
- [ ] Push the feature branch and require green Windows/Python 3.14 and Linux/Python 3.11 CI.
- [ ] Fast-forward `main`, require green `main` CI, then repoint the Funnel and verify the public GitHub Pages experience.

The public switch is complete only after health, exact CORS, three-branch non-stream/SSE, Prediction Lab, cancellation cleanup, and desktop/mobile browser checks pass. A failed public validation must restore the recorded port-8000 Funnel target.

```text
Story source commit -> product evidence commit -> feature CI -> main CI -> public cutover
```

- [ ] Tag points to the exact CI-green main commit.
- [ ] Tag is annotated, not lightweight.
- [ ] No source or evidence changes occur between main CI success and tag creation.
