# Post-v1 Story Forge data pipeline — design

Status: post-v1 research extension (not Stage 22). This document scopes the
deterministic SimpleStories *data* layer only; model, training, generation,
serving, and web integration are separate, later work streams.

## Scope and non-goals

In scope:

- Exact upstream identity binding for the reviewed SimpleStories revision.
- Deterministic label mapping, split, selection, quota redistribution, and
  ByteLevel BPE tokenization producing four reproducible artifacts.

Out of scope: model architecture, training, serving, web UI, public deployment,
and any wall-clock performance claim. The data layer makes no speed claim.

## Upstream identity (reviewer-verified)

| Field | Value |
|---|---|
| repo | `SimpleStories/SimpleStories` |
| revision | `e63b8adc3b1a1bdc7cac5b500d150b71346b0628` |
| filename | `processed.parquet` |
| size | 431,432,698 bytes |
| SHA-256 | `83ad95336a6b7a028be86a12c63facb3956097fe6177b577337510eeb5735938` |
| license | MIT |
| citation | Finke et al., "Parameterized Synthetic Text Generation with SimpleStories," arXiv:2504.09184 (2025) |

The official download enforces exact size and SHA-256 before scanning. A local
`--source-parquet` is marked `local_fixture` and has its identity measured but
not enforced against the official value.

## Actual schema (verified from the Parquet footer)

`processed.parquet` uses the Arrow schema (order fixed by the source):

```text
generation_id: string   story: string   topic: string   theme: string
style: string           feature: string grammar: string persona: string
initial_word_type: string initial_letter: string
word_count: int64  character_count: int64  num_paragraphs: int64
avg_word_length: float64  avg_sentence_length: float64
flesch_reading_ease: float64  flesch_kincaid_grade: float64
dale_chall_readability_score: float64
num_stories_in_completion: int64 expected_num_stories_in_completion: int64
model: string
```

The data layer declares `topic`, `theme`, and `style` as **scalar strings**.
This supersedes the earlier task-brief wording ("list topic"/"list theme"/
"boolean style struct"): inspection of the real Parquet footer metadata confirms
scalar strings. `style` carries a prose phrase such as `action-packed`, not a
boolean flag struct. Selected columns are `generation_id`, `story`, `topic`,
`theme`, `style`.

## Label mapping v2

Metadata phrases are normalized (lowercase, whitespace collapse, `-`/`_` folded
to space) and matched as whole phrases. Dimension sources:

- world ← `topic`
- tone ← `style` + `theme`
- theme ← `theme` + `topic`

The canonical label vocabularies and their alias tables (space/forest/robot/
mystery worlds; adventurous/mysterious/warm/funny tones; discovery/friendship/
logic/courage themes) are compiled into `src/minigpt/story_data.py`. A row is
eligible only when every dimension maps. Multi-label ties within a dimension are
broken by the smallest SHA-256 of `(generation_id, dimension, candidate)`, never
by Python set ordering.

### Mapping version 2

`MAPPING_VERSION` is `2`, written into `metadata.json` under `mapping_version`.
Version 2 expands the alias tables to cover the complete downstream
vocabularies of the pinned `processed.parquet`: all 47 `topic` phrases, all 61
`theme` phrases, and all 24 `style` phrases. Coverage is enforced by
`validate_mapping_tables()`, which also rejects any normalized alias shared by
two canonical labels and any alias that normalizes to an empty phrase. The
pinned phrase lists (`PINNED_TOPIC_PHRASES`, `PINNED_THEME_PHRASES`,
`PINNED_STYLE_PHRASES`) are exact, order-stable records of the upstream
vocabulary; they are coverage contracts, not classification claims.

Two tone reassignments were made for defensible product semantics:

- `tragic` and `melancholic` belong under `mysterious`, not `warm`.
- `modern` belongs under `adventurous`, not `funny`.

The four public world controls keep their canonical IDs (`space`, `forest`,
`robot`, `mystery`) and gain documented UI labels in `WORLD_UI_LABELS`:

| world | UI label |
|---|---|
| `space` | Space Expedition |
| `forest` | Enchanted Wilds |
| `robot` | Wonder Workshop |
| `mystery` | Curious Mystery |

The broad world mappings remain product-domain buckets (a "world" selects a
setting bucket, not a literal classifier). This is documented explicitly and
must not be read as a claim of semantic ground truth over the upstream tags.

## Deterministic split and selection

- Split: SHA-256 of `(seed, generation_id, "split")`; the bottom 5% of the
  digest space is validation, independent of input order.
- Selection rank: SHA-256 of `(seed, generation_id, "select")`; the smallest
  ranks survive per bucket via bounded max-heaps (`O(train + val)` peak memory).
- Quotas: sixteen `(world, tone)` cells. Equal per-cell desired quota with a
  lexicographic integer remainder; cap to availability; restore each world
  marginal from same-world spare cells; then lexicographic water-filling across
  all spare cells. Insufficient total eligible capacity raises an actionable
  error with counts.
- Duplicate `generation_id` within a split, and any train/val overlap, are
  rejected; selected records are sorted by rank before tokenizer training and
  encoding.

## Encoding and framing

`BPETokenizer` is trained only on selected training-story text in rank order.
Every story is caller-framed:

```text
<bos> <world_*> <tone_*> <theme_*> <story> <story-token-ids> <eos>
```

Outputs are flat `numpy.uint16` `train.npy`/`val.npy` compatible with the
existing `TokenBatcher`, plus `tokenizer.json` and `metadata.json`.

## Metadata

Canonical UTF-8/LF, sorted-key JSON with no timestamps, absolute paths,
hostname, PID, or secrets. It records the source, preparation parameters,
both-pass counts (verified equal), final bucket/world/tone/theme counts,
requested vs selected counts, sorted selected-ID SHA-256 for both splits, the
no-overlap flag, tokenizer settings/special-token identity, artifact hashes and
sizes, and an explicit deterministic + bounded non-performance claim.

## Atomic output

The four artifacts are built in a sibling staging directory, validated by
reloading the tokenizer and arrays, then atomically swapped into place with a
rollback-safe backup. Any failure preserves the prior output and removes
staging/backup residue.

## Packaging and CLI

- `[story]` optional extra: `huggingface-hub>=0.35,<1.0`, `pyarrow>=21,<22`,
  `tokenizers>=0.22,<0.23`.
- `minigpt prepare-stories` and `prepare_stories.py` are lazy at import/help
  time (no pyarrow/huggingface_hub import until preparation runs) and report
  `python -m pip install -e ".[story]"` when required.

## Calibration configs

Three Story Forge training configs ship with this extension (`vocab_size` is
resolved from the tokenizer to `4096` at load time):

| config | n_layer | n_head | n_embd | parameters (untied embedding/head) |
|---|---|---|---|---|
| `configs/story_forge_smoke.yaml` | 2 | 2 | 64 | 655,680 |
| `configs/story_forge_1m.yaml` | 3 | 4 | 96 | 1,168,032 |
| `configs/story_forge_5m.yaml` | 6 | 4 | 208 | 4,928,144 |

Counts are computed by the existing project helper
`minigpt.model.expected_gpt_parameter_count` with `block_size=512`, `bias=False`
(untied embedding and head). The 5M candidate targets 4.5M–5.5M and is divisible
by `n_head`.

## Canonical preparation results (descriptive local results)

The canonical two-run preparation of the pinned official source has completed.
These are descriptive local results reproduced on the development host only;
they are not model-quality or wall-clock performance claims and do not change
any historical Stage 7A–21 Evidence.

| run | wall time | peak RSS recorded |
|---|---|---|
| A | 1136.48s | 2,443,132,928 bytes |
| B | 1120.66s | 2,443,132,928 bytes |

Artifact identity (byte-identical across both runs):

| artifact | SHA-256 |
|---|---|
| `train.npy` | `d7d979d1c55c6eb3596b6b36cfda344cfdc8db0856291d7421a92e657288a42e` |
| `val.npy` | `e40251b9d47484306b35ab714dedc541807a920a9e4fc59d0564a073d69a4b77` |
| `tokenizer.json` | `7c897e0d51d135f6bb24bdbd18b0e40db88c0e62defcff9d23f3a204e092585c` |
| `metadata.json` | `6fc142c1a0c0d689e0a6f94b20c47a5ce7c7a4fcf9bfac08cedf35ce71e18819` |

Selection summary: 200,000 train stories, 5,000 validation stories, no
train/validation overlap, 4096 vocabulary, `mapping_version` 2, and equal world
marginals. These numbers describe what the deterministic pipeline produced, not
a claim that Story Forge training has achieved any downstream quality target.

## Story controls and framing

`src/minigpt/story.py` adds user-facing controls and context-aware framing on
top of the data layer. It does not change the training tokenization or the
existing `BPETokenizer` contract.

- `StoryControls(world, tone, theme)`: frozen, slotted, validates that each
  selector belongs to the current canonical vocabulary.
- `story_control_prefix_ids(tokenizer)`: returns the five-token
  `<bos> <world_space> <tone_adventurous> <theme_discovery> <story>` prefix
  for the default controls; training, serving, and evaluation share this
  single source of truth.
- `frame_story_prompt(tokenizer, controls, opening, *, max_context_tokens=None,
  reserved_generation_tokens=0)`: validates the opening (rejects empty text,
  `<...>` control syntax, NUL, and other control characters; caps at 10,000
  characters), encodes it through the BPE tokenizer, preserves the control
  prefix in full, and left-truncates the opening at a token boundary when the
  caller-supplied context budget is exceeded. The control prefix is never
  truncated; the function raises when the budget is too small to hold it.
- `STORY_EVALUATION_CASES`: 16 original cases covering every
  (world × tone) combination with a rotating theme index so each of the four
  themes appears exactly four times. Each case carries a stable `id`, original
  `opening`, and deterministic `seed`. No copyrighted IP or franchise
  references.

Character tokenizers are explicitly rejected by both
`story_control_prefix_ids` and `frame_story_prompt` because they lack the
required special tokens.

## Model training, evaluation, and evidence

The 5M candidate config (`configs/story_forge_5m.yaml`, 6 layers / 4 heads /
208 embedding / block 512 / vocab 4096) resolves to exactly 4,928,144 trainable
parameters (untied embedding and head, `bias=False`), as computed by
`minigpt.model.expected_gpt_parameter_count`. A single 3000-step CPU run was
trained in segments with intentional exact resume (`--run-until-step 3000
--resume ...`), producing 3000 strictly ordered metric steps (0..2999) with 60
validation points at every 50th step and checkpoints at steps 499, 1499, and
2999. Identity, tokenizer/data hashes, and resume behavior were verified.

### Bounded deterministic evaluation

`src/minigpt/story_evaluation.py` and the thin `evaluate_stories.py` entrypoint
load a Story Forge checkpoint/tokenizer through public APIs and evaluate all 16
`STORY_EVALUATION_CASES` with fixed per-case seeds. It reports bounded metrics
only:

- validation loss/perplexity on 20 fixed deterministic recorded batches
  (fresh val batcher seeded with `seed + 1`);
- EOS stop rate and generated length;
- special-token leakage outside the natural EOS stop;
- invalid decode;
- repeated 3/4-grams and the longest immediate tail loop;
- distinct 1/2/3 ratios;
- lexical proxy scores for world/tone/theme (alias-phrase keyword counts);
- same-seed determinism, different-seed diversity, and cached-vs-uncached
  exact token equality.

The lexical proxy scores are keyword-presence surrogates only. They do **not**
measure semantic understanding, story quality, or authorship, and no such claim
is made. The conservative public default is temperature 0.8 / top_k 20, chosen
from measured samples rather than guessed claims.

### Local serving smoke

An isolated localhost `serve.py` smoke (subprocess on a free non-8000 port plus
an in-process ASGI run) exercises health, models, non-stream and SSE completion,
three concurrent independent completions, disconnect cancellation, zero
active/queued return, and verifies no checkpoint/tokenizer path leaks. It does
not cut over the live demo backend.

### Evidence package

`generate_story_forge_evidence.py` (with `minigpt.story_forge_evidence`)
assembles the hash-bound package at `docs/results/story-forge-model/`, binding
source commit, canonical tokenizer/data hashes, parameter count, training
trajectory, evaluator outputs/samples, serving smoke, resource cleanup, and the
bounded claim policy. Checkpoint and canonical data remain external/ignored.
The package is deliberately outside the Stage 7A-20 doctor registry (mirroring
the Stage 21 capstone) so its own loop can be verified without self-
registration. The verdict is `descriptive_only`; no production readiness,
general-chat, semantic-understanding, or universal-speedup claim is made.
