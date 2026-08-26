# Stage 21 Implementation Plan

## Goal

Close miniGPT v1.0 as an installable, auditable CPU reference project. Do not add new model or scheduler features.

## Work packages

1. **Single version source**
   - set `minigpt._version.__version__ = 1.0.0`;
   - switch setuptools to dynamic version metadata;
   - remove tracked generated egg-info;
   - test source/distribution/CLI agreement.
2. **Release artifact validator**
   - build one wheel and one sdist;
   - inspect required package/root command modules;
   - fresh-install wheel;
   - run pip check, module/console help/version, and quick doctor.
3. **Release doctor**
   - add `release` mode artifact validation;
   - keep output deterministic and path-independent;
   - retain full evidence/source ancestry checks through Stage 20.
4. **Closure documentation**
   - add changelog, completion statement, and release checklist;
   - update README, AGENTS, and technical overview;
   - state supported scope and deferred post-v1 research explicitly.
5. **Capstone evidence**
   - generate version/distribution evidence;
   - roll up Stage 7A–20 verifier results;
   - capture release doctor and runtime/simulator smoke;
   - capture exact-resume and release lifecycle tests;
   - publish hash-bound `docs/results/v1-release/`.
6. **Independent review and delivery**
   - detached fresh checkout;
   - build/install/doctor from release artifacts;
   - exhaustive pytest partitions;
   - Windows/Linux GitHub Actions;
   - push branch, fast-forward main, rerun CI;
   - create annotated `v1.0.0` tag only after main CI passes.

## Required gates

```text
python -m pip check
ruff format src tests
ruff format --check src tests
ruff check src tests
basedpyright
pytest
python -m minigpt verify --mode release --require-clean
```

Additionally verify every Stage 17–21 package and source ancestry, `git diff --check`, wheel/sdist contents, fresh installation, Stage 18 canonical simulation, Stage 19 real runtime, and checkpoint v2 exact resume.

## Commit boundaries

1. version, release validator, doctor, tests, docs, registry, CI;
2. Stage 21 source/capstone generator and lifecycle coverage;
3. generated v1-release evidence bound to the reviewed source commit;
4. review fixes only when required, followed by evidence regeneration.

The final tag must identify the exact main commit that passed required GitHub Actions.
