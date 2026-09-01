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

## Label mapping v1

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