# Stage 20 — Installable Unified CLI + Project Doctor

## 1. Motivation

Stage 19 makes the real HTTP runtime configurable, but the repository still exposes many root scripts and stage-specific evidence verifiers as separate entry points. A release candidate needs one installable command boundary and one auditable answer to the question: “Does this checkout still satisfy the project’s documented contracts?”

Stage 20 therefore adds two control-plane capabilities without changing model, training, inference, or serving semantics:

1. a unified, lazily imported `minigpt` CLI that forwards to existing command implementations;
2. a deterministic project doctor that verifies packaging, documentation, canonical configuration, evidence hashes/contracts, Git ancestry, and runtime wiring.

## 2. Scope

### 2.1 In scope

- `minigpt` console script and `python -m minigpt`;
- stable root help and version output;
- forwarding subcommands for data preparation, training, generation, simulation, serving, and verification;
- lazy imports so `--help` and `--version` do not require HTTP extras;
- an explicit Stage 7A–19 evidence registry;
- stage-specific verifier invocation and source-commit ancestry checks;
- special handling for the Stage 7A uncommitted checkpoint contract;
- deterministic `quick`, `ci`, and later `release` doctor modes;
- canonical Stage 18 simulation and Stage 19 runtime smoke in `ci` mode;
- stable, path-independent JSON output;
- CI integration with full Git history.

### 2.2 Out of scope

- replacing existing script parsers or changing their option semantics;
- downloading missing evidence or checkpoints;
- network services, telemetry upload, or remote attestation;
- performance claims;
- modifying training/inference numerical behavior;
- adding more serving features.

## 3. Unified CLI

The installed interface is:

```text
minigpt prepare-data ...
minigpt train ...
minigpt generate ...
minigpt simulate ...
minigpt serve ...
minigpt verify ...
```

`python -m minigpt` is equivalent. Each command is described by a typed immutable registry entry and imported only after dispatch. The existing module-level `main(argv)` remains the source of command-specific parsing and exit behavior.

`minigpt --version` imports only `minigpt._version` and `minigpt.cli`; it must not import FastAPI, HTTPX, or Uvicorn. If `serve` is selected without optional dependencies, the CLI returns an actionable installation error rather than failing during unrelated commands.

## 4. Evidence registry

The doctor uses an explicit ordered registry rather than filesystem discovery. This prevents an unregistered directory from silently becoming release evidence and makes stage aliases such as 7A, 11A, and 13B unambiguous.

Modern Stage 9–19 packages retain their own verifier functions. Stage 7A and Stage 8 use a legacy manifest adapter:

- every committed local artifact must be listed;
- every listed local artifact must exist and match size/hash;
- package membership must exactly match the local manifest;
- paths may not escape the package;
- Stage 7A alone may declare an external/uncommitted checkpoint through `sources.checkpoint`; it must be a repository-relative `checkpoints/**/*.pt` or `checkpoints/**/*.pth` path, while artifact-list entries remain package-local exact members.

No generic adapter may weaken a modern stage-specific claim policy.

## 5. Git ancestry

When a package exposes `source_commit`, the doctor requires it to be an ancestor of `HEAD`. Historical squash exceptions bind the exact reviewed source SHA and the exact merged `main` SHA; a different source is rejected even when it is otherwise an ancestor. A shallow repository is rejected because absence of ancestry information cannot be interpreted as success. CI therefore checks out with `fetch-depth: 0`.

Legacy packages without a source commit remain explicitly reported as a legacy contract; the doctor does not invent a source identity.

## 6. Doctor modes

### 6.1 `quick`

- installed/source version agreement;
- release-documentation presence;
- Stage 7A–19 evidence verification;
- source ancestry;
- Stage 18 canonical simulator config parsing;
- Stage 19 serving runtime config validation.

### 6.2 `ci`

Includes `quick`, then executes:

- the canonical Stage 18 simulation in a temporary output directory;
- an in-memory Stage 19 paged/lazy runtime construction using a tiny deterministic model;
- an installed `python -m minigpt --version` subprocess.

### 6.3 `release`

Reserved for Stage 21. It will extend `ci` with wheel/sdist build and fresh-install verification.

## 7. Stable output and failure behavior

The JSON schema contains:

```text
schema_version
mode
project_version
repository_root = "."
passed
checks[] = {name, status, detail}
```

It excludes absolute paths, timestamps, durations, random IDs, and host-specific environment details. Expected contract failures become structured failed checks and exit code 1; usage errors remain argparse exit code 2. Tracebacks are not the normal user interface.

## 8. Security and trust boundaries

- manifest paths are resolved beneath the package root before reading;
- parent traversal and undeclared absolute paths are rejected;
- evidence is never regenerated by the doctor;
- optional command imports happen only after command selection;
- Git commands use fixed argument forms and capture output;
- runtime smoke writes only to temporary directories.

## 9. Verification

Stage 20 must prove:

1. console-script metadata and `python -m minigpt` agree;
2. help/version are stable and do not import optional HTTP modules;
3. command arguments are forwarded unchanged;
4. missing optional serving dependencies produce an actionable error;
5. the registry is complete and ordered through Stage 19;
6. Stage 7A external checkpoint semantics do not weaken local artifact hashing;
7. tamper, unlisted artifact, shallow history, non-ancestor source, version drift, and dirty-tree requirements fail deterministically;
8. quick and CI doctor modes pass on the reviewed repository;
9. JSON is path-independent and reproducible;
10. Windows/Python 3.14 and Linux/Python 3.11 quality jobs pass.

## 10. Claim policy

Stage 20 is release engineering and verification infrastructure. Its evidence is structural and correctness-oriented. It makes no wall-clock performance improvement claim.
