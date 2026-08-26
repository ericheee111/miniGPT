# miniGPT v1.0.0 Release Checklist

This checklist is executable policy. Do not create or move the `v1.0.0` tag until every required item is satisfied by the exact main commit.

## 1. Source and metadata

- [ ] `src/minigpt/_version.py` contains `1.0.0`.
- [ ] `pyproject.toml` reads dynamic version metadata from `minigpt._version.__version__`.
- [ ] Generated `*.egg-info` is ignored and not tracked.
- [ ] `minigpt --version`, `python -m minigpt --version`, wheel metadata, and distribution metadata agree.
- [ ] `CHANGELOG.md` contains the v1.0.0 capability and limitation summary.
- [ ] `README.md`, `AGENTS.md`, `docs/PROJECT_OVERVIEW.md`, and `docs/PROJECT_COMPLETION.md` agree on the Stage 21 closure boundary.

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

- [ ] No Ruff formatting changes remain.
- [ ] Ruff lint is clean.
- [ ] basedpyright reports 0 errors, 0 warnings, and 0 notes.
- [ ] Full pytest has zero failures; only documented platform skips are allowed.
- [ ] `git diff --check` is clean.

## 3. Evidence and ancestry

- [ ] Stage 7A–20 project-doctor registry passes.
- [ ] Stage 17–21 package verifiers pass exact membership and SHA-256 checks.
- [ ] Every modern evidence `source_commit` is an ancestor of release HEAD.
- [ ] Repository history is not shallow during ancestry verification.
- [ ] Stage 21 capstone evidence is bound to the reviewed source commit.
- [ ] No evidence package claims unsupported wall-clock improvement.

## 4. Release artifacts

```powershell
minigpt verify --mode release --require-clean
```

- [ ] Exactly one wheel and one sdist build from a clean checkout.
- [ ] Wheel contains package modules and supported root command modules.
- [ ] Fresh wheel installation succeeds.
- [ ] Fresh `pip check` succeeds using the provisioned dependency environment.
- [ ] Fresh module and console-script help/version succeed.
- [ ] Fresh installed quick doctor succeeds against the release checkout.

## 5. Functional capstone

- [ ] Checkpoint v2 exact-resume focused suite passes.
- [ ] Stage 18 canonical lazy-KV simulation completes with expected correctness contracts.
- [ ] Stage 19 real serving runtime builds with paged/lazy configuration.
- [ ] HTTP/SSE lifecycle and Windows Uvicorn subprocess tests pass.
- [ ] Runtime manifest remains deterministic, atomic, and free of absolute paths/secrets.

## 6. Fresh checkout review

- [ ] Detached fresh checkout has no tracked or untracked pollution.
- [ ] Editable install and release artifact build succeed in the fresh checkout.
- [ ] Focused Stage 19–21 tests pass independently of the development worktree.
- [ ] Full evidence registry and source ancestry pass from the fresh checkout.

## 7. GitHub delivery

- [ ] Feature branch is pushed.
- [ ] Feature-branch Windows/Python 3.14 quality job succeeds.
- [ ] Feature-branch Ubuntu/Python 3.11 quality job succeeds.
- [ ] Reviewed commits are fast-forwarded or merged to `main` without content changes.
- [ ] Main Windows/Python 3.14 quality job succeeds.
- [ ] Main Ubuntu/Python 3.11 quality job succeeds.
- [ ] `origin/main` equals the reviewed local HEAD.

## 8. Tagging

After all items above pass:

```powershell
git tag -a v1.0.0 -m "miniGPT v1.0.0"
git push origin v1.0.0
```

- [ ] Tag points to the exact CI-green main commit.
- [ ] Tag is annotated, not lightweight.
- [ ] No source or evidence changes occur between main CI success and tag creation.
