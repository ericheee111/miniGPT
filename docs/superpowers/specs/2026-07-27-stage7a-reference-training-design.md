# Stage 7A Reference Training Evidence Design

## 1. Scope and Objective

Stage 7A produces one trustworthy, reproducible CPU reference-training record for miniGPT. It does
not add model capabilities or optimize benchmark methodology. The result must answer:

- exactly which code, configuration, data, environment, and checkpoint produced the evidence;
- whether the complete configured experiment finished, including a real checkpoint resume;
- whether loss, learning rate, throughput, and generated samples came from unedited artifacts;
- which conclusions are numerical facts and which are limited qualitative observations.

Out of scope: Benchmark v2, KV cache, BPE, GPU/CUDA, LoRA, distributed training, profiler expansion,
and larger model architectures.

The reference experiment is distinct from:

- a **smoke test**, which proves that a short code path executes;
- a **calibration run**, which estimates cost and checks basic learning before fixing the experiment
  horizon;
- a **benchmark**, which controls warmup, repetition, machine state, and timed regions to compare
  performance;
- the **reference training**, which records one complete learning trajectory and its provenance.

## 2. Canonical Experiment

The canonical configuration is `configs/char_gpt_reference.yaml`, derived from
`configs/char_gpt.yaml`. Its teaching-scale architecture remains fixed:

```yaml
data:
  block_size: 128
  batch_size: 16
model:
  n_layer: 4
  n_head: 4
  n_embd: 128
  dropout: 0.1
  bias: false
```

The optimizer remains AdamW with the existing maximum/minimum learning rates, betas, weight decay,
and gradient clipping unless calibration reveals a correctness problem. Calibration may change
event intervals to obtain evidence quickly, but it does not define the reference result.

The final `max_steps`, `warmup_steps`, and `lr_decay_steps` are selected only after a calibration
run with the same model and batch dimensions. Selection uses:

- median training step time and tokens/s after the earliest startup steps;
- observed RSS;
- finite and decreasing train loss;
- at least two validation observations;
- successful scheduled generation and checkpoint loading;
- a bounded wall-clock target appropriate for this CPU.

The chosen values are written once into the canonical config before the formal run.
`max_steps` remains the full experiment definition, `lr_decay_steps` remains the fixed schedule
horizon, and the real interruption uses `--run-until-step` without changing either value.
The sample interval must divide `max_steps`, so the final completed step has an actual scheduled
sample without adding an exit-triggered sample.

## 3. Considered Provenance Approaches

### 3.1 Infer provenance from artifact timestamps

This is the smallest implementation, but it is rejected. File creation/change times do not have
portable semantics, cannot prove a resume occurred, and may be modified by copying.

### 3.2 Add environment and timestamps to checkpoint v2

This would make the checkpoint self-contained, but it is rejected for Stage 7A. Environment
evidence is observational rather than trajectory state, and changing the v2 schema would mix
training correctness with report provenance.

### 3.3 Optional run-provenance sidecar plus pure reporter

This is the selected design. `train.py` gains an opt-in provenance path. A focused module records
each actual CLI invocation around `run_training()` without changing the trainer or checkpoint
schema. A separate report module validates and transforms the recorded artifacts.

This design preserves normal training behavior, proves the real resume, and keeps report-specific
dependencies out of the base training installation.

## 4. Provenance Capture

When a provenance path is supplied, the training CLI records an ignored JSON sidecar. The first
segment refuses to start if Git is dirty. Resume segments require the same:

- experiment name derived from the canonical config stem;
- provenance schema version;
- repository-relative config path and its SHA-256;
- Git commit SHA, branch, and dirty state;
- experiment start UTC and last segment end UTC;
- operating system and architecture;
- Python, PyTorch, and NumPy versions;
- CPU name obtained from the Windows registry, Linux `/proc/cpuinfo`, or `platform.processor()`;
- physical/logical core counts;
- `torch.get_num_threads()` and CUDA availability;
- tokenizer/train/validation SHA-256;
- one or more run segments, each with start/end UTC, resume flag, input checkpoint SHA-256,
  exclusive run boundary, and final completed step;
- resolved config SHA-256 after the first checkpoint is written.

The final `environment.json` combines this captured run environment with report-validated
checkpoint format version, model parameter count, resolved config hash, and overall experiment
start/end times. These fields are computed rather than manually transcribed.

Every resume verifies the commit, source config hash, resolved config hash, and dataset
fingerprints against the existing sidecar before running. The sidecar is written atomically after a
successful segment. A failed segment does not claim a successful end or completed step.

The formal run uses two normal process invocations with the same full config. The second segment
must use the checkpoint produced by the first. The final sidecar therefore proves that a resume was
actually requested rather than inferred from continuous metrics.

## 5. Report Inputs and Validation Boundary

The root CLI `report_training.py` delegates to `minigpt.training_report`. Inputs are:

- canonical source config;
- `metrics.jsonl`;
- `samples.txt`;
- final checkpoint v2;
- run-provenance sidecar;
- output directory.

Before creating output, the reporter validates:

1. Git SHA and clean-at-run-start evidence exist.
2. The checkpoint is format v2 and `completed_step == max_steps - 1`.
3. Checkpoint resolved config matches the provenance resolved-config hash.
4. Current tokenizer/train/validation files match checkpoint fingerprints.
5. Metrics steps are exactly `0..max_steps-1`, with no duplicate or missing step.
6. All numerical fields are finite, except validation loss may be absent.
7. Validation values occur only on actual recorded steps; no missing values are synthesized.
8. Every recorded learning rate equals the configured schedule within a strict floating-point
   tolerance.
9. At least one recorded run segment used resume and the last segment reached the full horizon.
10. Samples parse into ordered `step=<absolute step>` records and include the final scheduled sample.

Any validation failure aborts before replacing a previously published report.

The checkpoint module exposes a read-only typed metadata API so the reporter does not duplicate
private checkpoint parsing. It returns version, completed step, resolved config, data fingerprints,
and model state information needed for validation. Training resume behavior is unchanged.

## 6. Stable Transformations

### 6.1 CSV

`metrics.csv` preserves one row per JSONL record and all source fields. It uses a fixed header,
UTF-8, `\n` line endings, and stable Python numeric serialization. The reporter never rounds source
values in CSV.

### 6.2 Figures

Matplotlib is provided only through a `report` optional dependency and the development environment.
The normal training dependency set does not include it. Figures use the non-interactive `Agg`
backend, fixed dimensions/DPI, global step on the x-axis, and titles containing the experiment name
and short Git SHA.

- `loss_curve.png`: train loss at every step and validation loss as scatter points only at observed
  steps. There is no interpolation or invented validation value.
- `learning_rate_curve.png`: the recorded learning rate sequence.
- `throughput_curve.png`: raw tokens/s per step plus a labeled median reference line. It is
  observational training telemetry, not Benchmark v2 evidence.

Tests verify parsed sequences and file existence, not PNG byte equality.

### 6.3 Samples

`generated_samples.txt` is an exact copy of every scheduled sample, not a curated subset. The
Markdown report presents:

- a deterministic untrained baseline reconstructed from the same resolved config, tokenizer,
  initialization seed, prompt, and isolated sample-generator seed;
- the first scheduled training sample;
- the sample at the median position in the ordered sample list;
- the final scheduled sample.

This position-based rule is fixed before the formal run. The report does not select text for
quality. Dropout is active during training but disabled by `model.eval()` during sampling; the
independent generator controls multinomial draws without perturbing training RNG.

## 7. Generated Artifact Set

The report writes to `docs/results/reference-training/`:

- `README.md`;
- `environment.json`;
- `resolved_config.yaml`;
- `metrics.csv`;
- `loss_curve.png`;
- `learning_rate_curve.png`;
- `throughput_curve.png`;
- `generated_samples.txt`;
- `artifact_manifest.json`.

The Markdown report contains:

- objective and distinction from smoke/benchmark evidence;
- exact commands, including the real interruption/resume;
- architecture and unique trainable parameter count;
- Tiny Shakespeare token counts and fingerprints;
- initial/final/best train loss and character-level perplexity;
- initial/final/best observed validation loss and perplexity;
- median tokens/s and observed RSS;
- total steps, schedule horizon, and resume status;
- baseline plus position-selected representative samples;
- Git SHA, source config/metrics/checkpoint hashes;
- limitations and reproduction instructions.

The output is generated in a temporary sibling directory and moved into place only after all files
are complete.

## 8. Artifact Manifest

`artifact_manifest.json` contains:

- manifest schema version;
- generator Git SHA/version;
- source config, metrics, samples, checkpoint, and provenance paths, SHA-256, and sizes;
- generated artifact paths, SHA-256, and sizes;
- tokenizer/train/validation fingerprints;
- resolved config SHA-256;
- checkpoint format version and final completed step.

A manifest cannot recursively contain its own final hash. It therefore lists every other generated
result and explicitly records `manifest_self_excluded: true`. The manifest's own SHA-256 is reported
in the Stage 7A handoff after generation.

All stored paths are repository-relative POSIX paths. No committed artifact contains a local drive
letter, user profile, or absolute machine path.

## 9. What Enters Git

Committed:

- canonical reference config;
- provenance/report implementation and tests;
- design and implementation plan;
- the compact report directory listed above;
- a concise link and evidence summary in the repository README.

Ignored and not committed:

- Tiny Shakespeare raw and processed data;
- calibration outputs;
- source `metrics.jsonl` and TensorBoard event files;
- training output directories;
- checkpoints;
- temporary provenance sidecar;
- profiler traces.

The final checkpoint is not committed because PyTorch checkpoints are comparatively large binary
blobs, create poor Git deltas, and include optimizer/RNG state not needed to review the report. Its
SHA-256, size, format version, completion step, config hash, and data fingerprints provide a durable
identity without storing the blob in Git.

## 10. Testing Strategy

Tests use tiny local datasets and synthetic metrics/checkpoints:

- strict JSONL parsing, finite-value rejection, continuous step validation;
- sparse validation-series extraction without interpolation;
- configured learning-rate sequence validation;
- deterministic first/middle/final sample selection;
- environment/provenance serialization and multi-segment resume recording;
- dirty Git and mismatched commit/config/data rejection;
- resolved-config and checkpoint-completion validation;
- stable CSV values;
- calculated summary statistics and character-level perplexity;
- all required artifacts exist and the manifest hashes match their bytes;
- report regeneration reproduces the same numerical conclusions;
- PNG tests inspect existence/nonzero size rather than binary equality.

The full existing Ruff, basedpyright, and pytest gates remain mandatory.

## 11. Calibration and Formal Run Procedure

1. Prepare Tiny Shakespeare and verify the generated metadata and SHA-256 files.
2. Run an ignored calibration configuration with the canonical model/batch dimensions.
3. Record loss trend, validation observations, median tokens/s, median step time, RSS, generation,
   and checkpoint-load success.
4. Select and commit the canonical `max_steps`, warmup, decay horizon, and event intervals.
5. Commit all report code/config changes and verify a clean Git tree.
6. Start the formal run with provenance capture and an exclusive boundary that does not redefine the
   experiment.
7. Resume in a new process using the same config and first segment checkpoint.
8. Require the final checkpoint to reach `max_steps - 1`.
9. Generate and validate the report.
10. Commit only the compact result directory and README link; retain the checkpoint locally only as
    an ignored artifact.

Calibration results are reported during development but never described as the reference result.

## 12. Acceptance and Interpretation

The reference result is accepted only if:

- losses and all telemetry are finite;
- train loss is materially below its initial value;
- validation behavior is reported honestly, whether improved or not;
- the configured LR sequence is exact;
- checkpoint v2 loads and reaches the complete horizon;
- metrics steps are continuous;
- provenance connects code SHA, clean state, config hash, and data fingerprints;
- final text shows more local character/word/format structure than the deterministic untrained
  baseline, or the report explicitly states that it does not;
- rerunning the reporter reproduces the same numerical conclusions.

Loss reduction alone does not prove generalization. Character-level perplexity is `exp(loss)` under
cross-entropy, but both loss and perplexity remain dataset/split-specific. Throughput from the
reference run is descriptive and must not be presented as a controlled benchmark.
