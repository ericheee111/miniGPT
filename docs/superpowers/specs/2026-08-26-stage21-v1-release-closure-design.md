# Stage 21 — v1.0 Release Closure + Capstone Evidence

## 1. Decision

Stage 21 is the final planned implementation stage for miniGPT v1.0. The project is considered complete when its existing training, inference, serving, packaging, and evidence contracts can be installed and verified from a clean source archive on the supported Windows/Linux matrix.

This stage deliberately does not add another model optimization or serving policy. Future ideas remain documented research directions rather than blockers for v1.0.

## 2. Completion definition

miniGPT v1.0 is complete when all of the following hold:

1. one source of truth reports version `1.0.0` in source, wheel metadata, console script, and module CLI;
2. wheel and sdist build from a clean checkout;
3. the wheel contains the package and all supported command modules;
4. a fresh environment can install the wheel, pass dependency validation, run help/version, and execute the quick project doctor;
5. the real HTTP runtime exposes the reviewed Stage 15–18 configuration through Stage 19;
6. the Stage 7A–20 evidence registry verifies exact hashes/contracts and source ancestry;
7. Stage 18 canonical simulation and Stage 19 runtime smoke pass from release code;
8. checkpoint v2 exact-resume tests remain green;
9. README, changelog, technical overview, completion statement, and release checklist agree;
10. Ruff, basedpyright, pytest, fresh checkout verification, and Windows/Linux CI are green;
11. the reviewed commit is merged to `main` and annotated as `v1.0.0` only after CI success.

## 3. Version contract

`src/minigpt/_version.py` is the sole authored version value. Setuptools reads it through dynamic metadata. `minigpt.__version__`, `minigpt --version`, `python -m minigpt --version`, distribution metadata, wheel filename, and release evidence must all agree.

Generated `*.egg-info` is build output and is not committed. A release verifier must detect source/metadata drift rather than relying on stale generated files.

## 4. Release doctor

`minigpt verify --mode release` extends Stage 20 CI mode with:

- wheel and sdist build using the repository build backend;
- exact wheel content inspection;
- fresh environment creation;
- wheel installation without importing the source tree as the installed package;
- `pip check` against the already provisioned test dependency environment;
- module and console-script help/version checks;
- quick doctor execution from the installed wheel against the reviewed repository.

Temporary paths, build timings, and host identities are excluded from stable reports. Artifact SHA-256 values may be recorded in capstone evidence.

## 5. Capstone evidence

The Stage 21 package under `docs/results/v1-release/` binds:

- version and distribution metadata;
- Stage 7A–20 registry roll-up;
- release-doctor report;
- wheel/sdist hashes and fresh-install result;
- Stage 18 canonical simulation and Stage 19 real runtime wiring;
- checkpoint v2 exact-resume focused test output;
- Stage 19–21 lifecycle tests;
- full quality-gate result summary;
- source commit and exact artifact membership.

The capstone verifier cross-checks internal documents rather than treating the summary as self-authenticating.

## 6. Documentation closure

- `CHANGELOG.md` records the v1.0 capability set and bounded claims.
- `docs/PROJECT_COMPLETION.md` states why the project is complete, what remains outside scope, and how maintenance proceeds.
- `docs/RELEASE_CHECKLIST.md` is an executable pre-tag checklist.
- `README.md`, `AGENTS.md`, and `docs/PROJECT_OVERVIEW.md` identify Stage 21 as the planned closure point.

## 7. Scope after v1.0

The repository enters maintenance/research-extension mode. Bug fixes, dependency compatibility, documentation corrections, and evidence hardening may ship as patch releases. New research features require a new explicit post-v1 design and are not implicitly part of “finishing” v1.0.

Deferred directions include BPE, partial-block COW, swap/offload, speculative decoding, optimized CPU/GPU kernels, quantization, distributed training, and adaptive/SLO scheduling.

## 8. Claim policy

The v1.0 claim is that miniGPT is a reproducible CPU reference lab with verified training, inference, serving, packaging, and evidence contracts. It is not a claim of production-scale throughput, GPU parity, or universal wall-clock performance improvement.
