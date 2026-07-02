# AGENTS.md

Compact guide for OpenCode sessions. Read before editing.

## Status — most modules are empty stubs

Only `src/minigpt/data.py` is implemented. These files are **0-byte placeholders**; do not try to run, import from, or extend them as if they had logic:

- `train.py`, `generate.py` (root entrypoints — not built yet)
- `src/minigpt/model.py`, `src/minigpt/trainer.py`, `src/minigpt/metrics.py`

The real working entrypoint is `minigpt.data.prepare_tiny_shakespeare(data_dir=Path("data"))`: downloads Tiny Shakespeare, char-tokenizes, splits 90/10, writes artifacts under `data/raw/` and `data/processed/`.

## Python version — 3.14 only

`requires-python = ">=3.14,<3.15"`. The toolchain (`target-version = "py314"`, basedpyright `pythonVersion = "3.14"`) hard-rejects anything else. Do not assume 3.11/3.12 works.

## Install

```powershell
pip install -e ".[dev]"
```

`requirements.txt` is **empty** — all deps live in `pyproject.toml`. Do not run `pip install -r requirements.txt` expecting dependencies.

Distribution name is `minitrain-gpt`; importable package is `minigpt` (src layout under `src/minigpt/`). Tests do `import minigpt.data`, so the editable install is required for pytest to resolve the package.

## Verify (run all, in this order)

```powershell
ruff format src tests
ruff check src tests
basedpyright
pytest
```

- Typechecker is **`basedpyright`** (not `pyright`, not `mypy`). It runs on both `src` and `tests` — there is no per-file relaxation for tests.
- pytest: `testpaths = ["tests"]`, `--strict-config --strict-markers`. No markers are registered; adding `@pytest.mark.foo` without registering it fails collection.
- **No CI.** Verification is local and manual. Running all four gates is the only signal of "done."

## Strict toolchain — gotchas that bite

- basedpyright `typeCheckingMode = "all"` with `reportMissingParameterType`, `reportMissingReturnType`, `reportPrivateUsage`, `reportUnusedVariable` all set to `error`. Every function (including tests) needs full annotations. Accessing `_private` attributes from outside the class is an error even in tests.
- `reportUnnecessaryTypeIgnoreComment = "error"` — a stray `# type: ignore` is a hard error. Do not reach for `# type: ignore` or `# pyright: ignore` as a crutch; fix the type.
- ruff `select = ["ALL"]`, line-length 100, double quotes, Google docstring convention. Tests relax `ARG`, `D`, `PLR2004`, `S101`, `SLF001` via `[tool.ruff.lint.per-file-ignores]` — basedpyright does **not** inherit these relaxations.

## Test conventions

Tests use a BDD comment style — preserve it in new tests:

```python
def test_x(tmp_path: Path) -> None:
    # Given: ...
    # When: ...
    # Then: ...
```

Network-touching tests avoid real HTTP by using `file://` URLs via `Path.as_uri()` against `tmp_path` (see `test_download_text_fetches_source` and `test_prepare_tiny_shakespeare_writes_split_and_metadata`). Follow this pattern instead of mocking `urlopen`.

## Data artifacts (gitignored)

`data/raw/`, `data/processed/`, `checkpoints/`, `outputs/`, `runs/`, `*.pt`, `*.pth`, `*.prof`, `*.trace.json` are all gitignored. `configs/` exists but is empty. `.codegraph/` is gitignored but the local index is used for navigation.

## Commit messages

Git history is in Chinese (e.g. `实现 CharTokenizer 的 encode/decode 与对应单元测试`). Match this style for new commits unless the user requests otherwise.
