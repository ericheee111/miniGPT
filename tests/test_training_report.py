from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from minigpt.batching import TokenBatcher
from minigpt.checkpoint import (
    CheckpointResources,
    LegacyCheckpointResumeError,
    StateValue,
    compute_dataset_fingerprints,
    save_checkpoint,
)
from minigpt.data import CharTokenizer
from minigpt.model import GPT
from minigpt.optimization import learning_rate_at_step
from minigpt.run_provenance import (
    RunInvocation,
    begin_run_segment,
    complete_run_segment,
)
from minigpt.settings import (
    DataSettings,
    ExperimentConfig,
    ModelSettings,
    OptimizerSettings,
    RuntimeSettings,
    TrainingSettings,
)
from minigpt.training_report import (
    ReportInputs,
    ReportOutputExistsError,
    TrainingEvidenceError,
    build_training_series,
    character_perplexity,
    figure_title,
    generate_training_report,
    load_generated_samples,
    load_metric_records,
    select_representative_samples,
    summarize_training,
    throughput_reference_line,
    validate_reference_sources,
)

if TYPE_CHECKING:
    from collections.abc import Callable

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None

PROJECT_ROOT = Path(__file__).parents[1]
REFERENCE_RESULT_DIRECTORY = Path("docs/results/reference-training")
REFERENCE_TEXT_ARTIFACT_EOL = {
    "README.md": "crlf",
    "artifact_manifest.json": "crlf",
    "environment.json": "crlf",
    "generated_samples.txt": "crlf",
    "metrics.csv": "lf",
    "resolved_config.yaml": "crlf",
}


def metric_record(
    step: int,
    *,
    train_loss: float = 4.0,
    val_loss: float | None = None,
    learning_rate: float = 1e-3,
) -> dict[str, int | float | None]:
    return {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": learning_rate,
        "step_time_ms": 20.0 + step,
        "tokens_per_sec": 100.0 + step,
        "data_time_ms": 1.0,
        "forward_backward_time_ms": 15.0,
        "optimizer_time_ms": 4.0,
        "cpu_memory_mb": 250.0 + step,
    }


def write_metrics(path: Path, records: list[dict[str, int | float | None]]) -> None:
    _ = path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", "-C", str(repository), *arguments],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def reference_config(repository: Path) -> ExperimentConfig:
    return ExperimentConfig(
        runtime=RuntimeSettings(seed=7, num_threads=1, device="cpu"),
        data=DataSettings(
            directory=repository / "data" / "processed",
            block_size=4,
            batch_size=2,
        ),
        model=ModelSettings(
            vocab_size=None,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        ),
        optimizer=OptimizerSettings(
            optimizer_type="adamw",
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            grad_clip=1.0,
        ),
        training=TrainingSettings(
            max_steps=4,
            warmup_steps=0,
            lr_decay_steps=4,
            eval_interval=2,
            eval_batches=1,
            log_interval=1,
            checkpoint_interval=2,
            sample_interval=2,
            sample_tokens=4,
            sample_prompt="a",
            output_dir=repository / "outputs" / "reference",
            checkpoint_dir=repository / "checkpoints" / "reference",
            tensorboard_dir=repository / "outputs" / "reference" / "tensorboard",
        ),
    )


def save_fixture_checkpoint(
    repository: Path,
    config: ExperimentConfig,
    *,
    completed_step: int,
) -> Path:
    model = GPT(config.model.to_gpt_config(config.data.block_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate)
    generator = torch.Generator(device="cpu")
    _ = generator.manual_seed(config.runtime.seed + 2)
    resources = CheckpointResources(
        model=model,
        optimizer=optimizer,
        train_batcher=TokenBatcher(
            np.arange(32),
            batch_size=config.data.batch_size,
            block_size=config.data.block_size,
            seed=1,
        ),
        val_batcher=TokenBatcher(
            np.arange(32),
            batch_size=config.data.batch_size,
            block_size=config.data.block_size,
            seed=2,
        ),
        sample_generator=generator,
        dataset_fingerprints=compute_dataset_fingerprints(config.data),
    )
    checkpoint_path = repository / "checkpoints" / "reference" / "latest.pt"
    save_checkpoint(
        checkpoint_path,
        resources=resources,
        step=completed_step,
        config=config,
    )
    return checkpoint_path


@dataclass(frozen=True, slots=True)
class EvidenceFixture:
    repository: Path
    inputs: ReportInputs


def prepare_evidence_fixture(tmp_path: Path) -> EvidenceFixture:
    repository = tmp_path / "repository"
    repository.mkdir()
    _ = (repository / ".gitignore").write_text(
        "data/\noutputs/\ncheckpoints/\n",
        encoding="utf-8",
    )
    config = reference_config(repository)
    data_dir = config.data.directory
    data_dir.mkdir(parents=True)
    tokenizer = CharTokenizer.from_text("abcd")
    tokenizer.save(config.data.tokenizer_path)
    save_tokens = cast(
        "Callable[[Path, npt.NDArray[np.uint16]], None]",
        np.save,
    )
    save_tokens(
        config.data.train_path,
        np.arange(64, dtype=np.uint16) % tokenizer.vocab_size,
    )
    save_tokens(
        config.data.val_path,
        np.arange(24, dtype=np.uint16) % tokenizer.vocab_size,
    )
    config_path = repository / "reference.yaml"
    _ = config_path.write_text(config.to_yaml(), encoding="utf-8")
    _ = run_git(repository, "init")
    _ = run_git(repository, "config", "user.email", "tests@example.com")
    _ = run_git(repository, "config", "user.name", "miniGPT tests")
    _ = run_git(repository, "add", ".gitignore", "reference.yaml")
    _ = run_git(repository, "commit", "-m", "reference fixture")

    resolved = config.resolve_vocab_size(tokenizer.vocab_size)
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True)
    provenance_path = output_dir / "run_provenance.json"
    checkpoint_path = save_fixture_checkpoint(repository, resolved, completed_step=1)
    first = begin_run_segment(
        provenance_path,
        config_path=config_path,
        config=config,
        invocation=RunInvocation(
            argv=("--config", "reference.yaml", "--run-until-step", "2"),
            run_until_step=2,
            resume_path=None,
        ),
    )
    complete_run_segment(
        provenance_path,
        segment=first,
        checkpoint_path=checkpoint_path,
        final_step=1,
    )
    second = begin_run_segment(
        provenance_path,
        config_path=config_path,
        config=config,
        invocation=RunInvocation(
            argv=("--config", "reference.yaml", "--resume", str(checkpoint_path)),
            run_until_step=None,
            resume_path=checkpoint_path,
        ),
    )
    checkpoint_path = save_fixture_checkpoint(repository, resolved, completed_step=3)
    complete_run_segment(
        provenance_path,
        segment=second,
        checkpoint_path=checkpoint_path,
        final_step=3,
    )

    metrics_path = output_dir / "metrics.jsonl"
    records = [
        metric_record(
            step,
            train_loss=4.0 - step * 0.25,
            val_loss=3.5 - step * 0.1 if step in {1, 3} else None,
            learning_rate=learning_rate_at_step(
                step,
                max_learning_rate=config.optimizer.learning_rate,
                min_learning_rate=config.optimizer.min_learning_rate,
                warmup_steps=config.training.warmup_steps,
                lr_decay_steps=config.training.lr_decay_steps,
            ),
        )
        for step in range(config.training.max_steps)
    ]
    write_metrics(metrics_path, records)
    samples_path = output_dir / "samples.txt"
    _ = samples_path.write_text(
        "step=1\nfirst sample\n\nstep=3\nfinal sample\n\n",
        encoding="utf-8",
    )
    return EvidenceFixture(
        repository=repository,
        inputs=ReportInputs(
            repository_root=repository,
            config_path=config_path,
            metrics_path=metrics_path,
            samples_path=samples_path,
            checkpoint_path=checkpoint_path,
            provenance_path=provenance_path,
        ),
    )


def test_metric_parser_preserves_sparse_validation_points(tmp_path: Path) -> None:
    # Given: contiguous step records with validation measured at one scheduled step.
    path = tmp_path / "metrics.jsonl"
    write_metrics(
        path,
        [
            metric_record(0, train_loss=4.2),
            metric_record(1, train_loss=3.8, val_loss=3.5),
            metric_record(2, train_loss=3.2),
        ],
    )

    # When: the metrics JSONL is parsed.
    records = load_metric_records(path)

    # Then: every step is retained and missing validation values remain absent.
    assert [item.step for item in records] == [0, 1, 2]
    assert [(item.step, item.val_loss) for item in records if item.val_loss is not None] == [
        (1, 3.5)
    ]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("not-json\n", "JSON"),
        (json.dumps({"step": 0}) + "\n", "fields"),
        (
            json.dumps({**metric_record(0), "unexpected": 1}) + "\n",
            "fields",
        ),
        (
            json.dumps({**metric_record(0), "step": True}) + "\n",
            "step",
        ),
        (
            json.dumps({**metric_record(0), "train_loss": math.nan}) + "\n",
            "finite",
        ),
        (
            json.dumps({**metric_record(0), "tokens_per_sec": math.inf}) + "\n",
            "finite",
        ),
        (
            "".join(
                (
                    json.dumps(metric_record(0)) + "\n",
                    json.dumps(metric_record(0)) + "\n",
                )
            ),
            "contiguous",
        ),
        (
            "".join(
                (
                    json.dumps(metric_record(0)) + "\n",
                    json.dumps(metric_record(2)) + "\n",
                )
            ),
            "contiguous",
        ),
        (json.dumps(metric_record(1)) + "\n", "contiguous"),
    ],
)
def test_metric_parser_rejects_invalid_evidence(
    tmp_path: Path,
    content: str,
    reason: str,
) -> None:
    # Given: malformed, ambiguous, or non-contiguous training metrics.
    path = tmp_path / "metrics.jsonl"
    _ = path.write_text(content, encoding="utf-8")

    # When/Then: the evidence boundary rejects the exact source file.
    with pytest.raises(TrainingEvidenceError, match=reason):
        _ = load_metric_records(path)


def test_sample_parser_and_selection_use_fixed_positions(tmp_path: Path) -> None:
    # Given: trainer-formatted samples at three scheduled global steps.
    path = tmp_path / "samples.txt"
    _ = path.write_text(
        "step=1\nfirst text\n\nstep=3\nmiddle text\nwith another line\n\nstep=5\nfinal text\n\n",
        encoding="utf-8",
    )

    # When: the samples are parsed and selected mechanically.
    samples = load_generated_samples(path)
    representative = select_representative_samples(samples)

    # Then: first, middle, and final positions are used without editorial selection.
    assert [(sample.step, sample.text) for sample in samples] == [
        (1, "first text"),
        (3, "middle text\nwith another line"),
        (5, "final text"),
    ]
    assert representative == (samples[0], samples[1], samples[2])


def test_sample_parser_rejects_missing_or_non_increasing_headers(tmp_path: Path) -> None:
    # Given: sample files that cannot prove a chronological scheduled sequence.
    no_header = tmp_path / "no-header.txt"
    repeated = tmp_path / "repeated.txt"
    _ = no_header.write_text("generated text only\n", encoding="utf-8")
    _ = repeated.write_text("step=1\nfirst\n\nstep=1\nsecond\n\n", encoding="utf-8")

    # When/Then: both ambiguous formats are rejected.
    with pytest.raises(TrainingEvidenceError, match="sample"):
        _ = load_generated_samples(no_header)
    with pytest.raises(TrainingEvidenceError, match="increasing"):
        _ = load_generated_samples(repeated)


def test_summary_and_character_perplexity_are_derived_from_raw_metrics(tmp_path: Path) -> None:
    # Given: literal metrics with two validation observations and non-monotonic train loss.
    path = tmp_path / "metrics.jsonl"
    write_metrics(
        path,
        [
            {
                **metric_record(0, train_loss=4.0, val_loss=3.8),
                "tokens_per_sec": 10.0,
                "cpu_memory_mb": 200.0,
            },
            {
                **metric_record(1, train_loss=2.5),
                "tokens_per_sec": 30.0,
                "cpu_memory_mb": 240.0,
            },
            {
                **metric_record(2, train_loss=3.0, val_loss=2.9),
                "tokens_per_sec": 20.0,
                "cpu_memory_mb": 220.0,
            },
        ],
    )

    # When: summary statistics and character-level perplexity are calculated.
    summary = summarize_training(load_metric_records(path))

    # Then: all conclusions are mechanical reductions over the raw records.
    assert summary.total_steps == 3
    assert summary.initial_train_loss == 4.0
    assert summary.final_train_loss == 3.0
    assert summary.best_train_loss == 2.5
    assert summary.initial_val_loss == 3.8
    assert summary.final_val_loss == 2.9
    assert summary.best_val_loss == 2.9
    assert summary.median_tokens_per_sec == 20.0
    assert summary.peak_cpu_memory_mb == 240.0
    assert character_perplexity(math.log(5.0)) == pytest.approx(5.0)


def test_cross_artifact_validation_accepts_complete_resumed_evidence(tmp_path: Path) -> None:
    # Given: one clean experiment with a real v2 checkpoint and two completed process segments.
    fixture = prepare_evidence_fixture(tmp_path)

    # When: every source artifact is validated as one experiment.
    evidence = validate_reference_sources(fixture.inputs)

    # Then: the resolved identity and raw data counts are exposed for report generation.
    assert evidence.metadata.format_version == 2
    assert evidence.metadata.completed_step == 3
    assert evidence.config.model.vocab_size == 4
    assert evidence.train_token_count == 64
    assert evidence.val_token_count == 24
    assert evidence.summary.total_steps == 4
    assert len(evidence.provenance.segments) == 2
    assert evidence.provenance.segments[1].resume_checkpoint_sha256 is not None


def test_cross_artifact_validation_rejects_wrong_learning_rate(tmp_path: Path) -> None:
    # Given: otherwise valid evidence with one hand-edited learning rate.
    fixture = prepare_evidence_fixture(tmp_path)
    records = [
        metric_record(
            step,
            train_loss=4.0 - step * 0.25,
            val_loss=3.5 - step * 0.1 if step in {1, 3} else None,
            learning_rate=0.123
            if step == 2
            else learning_rate_at_step(
                step,
                max_learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_steps=0,
                lr_decay_steps=4,
            ),
        )
        for step in range(4)
    ]
    write_metrics(fixture.inputs.metrics_path, records)

    # When/Then: the schedule inconsistency is rejected.
    with pytest.raises(TrainingEvidenceError, match="learning rate"):
        _ = validate_reference_sources(fixture.inputs)


@pytest.mark.parametrize(
    ("step", "val_loss"),
    [
        (1, None),
        (0, 3.7),
    ],
)
def test_cross_artifact_validation_enforces_validation_schedule(
    tmp_path: Path,
    step: int,
    val_loss: float | None,
) -> None:
    # Given: a scheduled validation is missing or an unscheduled value is fabricated.
    fixture = prepare_evidence_fixture(tmp_path)
    records = [
        {
            **metric_record(
                current_step,
                train_loss=4.0 - current_step * 0.25,
                val_loss=3.5 - current_step * 0.1 if current_step in {1, 3} else None,
                learning_rate=learning_rate_at_step(
                    current_step,
                    max_learning_rate=1e-3,
                    min_learning_rate=1e-4,
                    warmup_steps=0,
                    lr_decay_steps=4,
                ),
            ),
            "val_loss": val_loss
            if current_step == step
            else (3.5 - current_step * 0.1 if current_step in {1, 3} else None),
        }
        for current_step in range(4)
    ]
    write_metrics(fixture.inputs.metrics_path, records)

    # When/Then: validation evidence must occur exactly on configured absolute steps.
    with pytest.raises(TrainingEvidenceError, match="validation schedule"):
        _ = validate_reference_sources(fixture.inputs)


def test_cross_artifact_validation_rejects_metrics_count_mismatch(tmp_path: Path) -> None:
    # Given: otherwise valid metrics with the final global step removed.
    fixture = prepare_evidence_fixture(tmp_path)
    lines = fixture.inputs.metrics_path.read_text(encoding="utf-8").splitlines()
    _ = fixture.inputs.metrics_path.write_text(
        "\n".join(lines[:-1]) + "\n",
        encoding="utf-8",
    )

    # When/Then: the report cannot silently accept a shortened experiment.
    with pytest.raises(TrainingEvidenceError, match="metric count"):
        _ = validate_reference_sources(fixture.inputs)


def test_cross_artifact_validation_requires_material_train_loss_reduction(
    tmp_path: Path,
) -> None:
    # Given: complete evidence whose final train loss is not materially below its initial value.
    fixture = prepare_evidence_fixture(tmp_path)
    records = [
        metric_record(
            step,
            train_loss=4.0 - step * 0.05,
            val_loss=3.5 - step * 0.1 if step in {1, 3} else None,
            learning_rate=learning_rate_at_step(
                step,
                max_learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_steps=0,
                lr_decay_steps=4,
            ),
        )
        for step in range(4)
    ]
    write_metrics(fixture.inputs.metrics_path, records)

    # When/Then: a weak or failed optimization run cannot be published as reference evidence.
    with pytest.raises(TrainingEvidenceError, match="materially lower"):
        _ = validate_reference_sources(fixture.inputs)


def test_cross_artifact_validation_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    # Given: valid run evidence whose final checkpoint was replaced by an earlier step.
    fixture = prepare_evidence_fixture(tmp_path)
    config = reference_config(fixture.repository).resolve_vocab_size(4)
    _ = save_fixture_checkpoint(fixture.repository, config, completed_step=2)

    # When/Then: report generation cannot claim a complete experiment.
    with pytest.raises(TrainingEvidenceError, match="completed step"):
        _ = validate_reference_sources(fixture.inputs)


def test_cross_artifact_validation_rejects_v1_checkpoint(tmp_path: Path) -> None:
    # Given: a structurally valid legacy checkpoint substituted for the v2 source.
    fixture = prepare_evidence_fixture(tmp_path)
    payload = cast(
        "dict[str, StateValue]",
        torch.load(fixture.inputs.checkpoint_path, map_location="cpu", weights_only=True),
    )
    payload["format_version"] = 1
    payload["step"] = 3
    _ = payload.pop("completed_step")
    _ = payload.pop("sample_generator_random_state")
    _ = payload.pop("dataset_fingerprints")
    torch.save(payload, fixture.inputs.checkpoint_path)

    # When/Then: reference reporting preserves the Stage 6 inference-only v1 policy.
    with pytest.raises(LegacyCheckpointResumeError):
        _ = validate_reference_sources(fixture.inputs)


def test_cross_artifact_validation_rejects_changed_data(tmp_path: Path) -> None:
    # Given: valid evidence followed by an ignored training-data mutation.
    fixture = prepare_evidence_fixture(tmp_path)
    save_tokens = cast(
        "Callable[[Path, npt.NDArray[np.uint16]], None]",
        np.save,
    )
    save_tokens(
        fixture.repository / "data" / "processed" / "train.npy",
        np.arange(65, dtype=np.uint16) % 4,
    )

    # When/Then: the persisted dataset fingerprint catches the mutation.
    with pytest.raises(TrainingEvidenceError, match="fingerprint"):
        _ = validate_reference_sources(fixture.inputs)


def test_cross_artifact_validation_rejects_resolved_config_hash_mismatch(
    tmp_path: Path,
) -> None:
    # Given: a provenance document whose resolved experiment hash was edited.
    fixture = prepare_evidence_fixture(tmp_path)
    provenance_document = cast(
        "JsonValue",
        json.loads(fixture.inputs.provenance_path.read_text(encoding="utf-8")),
    )
    assert isinstance(provenance_document, dict)
    provenance_document["resolved_config_sha256"] = "0" * 64
    _ = fixture.inputs.provenance_path.write_text(
        json.dumps(provenance_document),
        encoding="utf-8",
    )

    # When/Then: a config from another experiment cannot be reported.
    with pytest.raises(TrainingEvidenceError, match="resolved config hash"):
        _ = validate_reference_sources(fixture.inputs)


def test_cross_artifact_validation_requires_completed_resume(tmp_path: Path) -> None:
    # Given: a provenance document edited to hide the first process segment.
    fixture = prepare_evidence_fixture(tmp_path)
    provenance_document = cast(
        "JsonValue",
        json.loads(fixture.inputs.provenance_path.read_text(encoding="utf-8")),
    )
    assert isinstance(provenance_document, dict)
    segments = provenance_document["segments"]
    assert isinstance(segments, list)
    provenance_document["segments"] = segments[-1:]
    _ = fixture.inputs.provenance_path.write_text(
        json.dumps(provenance_document),
        encoding="utf-8",
    )

    # When/Then: a single process cannot masquerade as resume evidence.
    with pytest.raises(TrainingEvidenceError, match="resume"):
        _ = validate_reference_sources(fixture.inputs)


def test_cross_artifact_validation_rejects_broken_resume_hash_chain(tmp_path: Path) -> None:
    # Given: a sidecar whose second segment names a checkpoint other than the first output.
    fixture = prepare_evidence_fixture(tmp_path)
    provenance_document = cast(
        "JsonValue",
        json.loads(fixture.inputs.provenance_path.read_text(encoding="utf-8")),
    )
    assert isinstance(provenance_document, dict)
    raw_segments = provenance_document["segments"]
    assert isinstance(raw_segments, list)
    second = raw_segments[1]
    assert isinstance(second, dict)
    second["resume_checkpoint_sha256"] = "0" * 64
    _ = fixture.inputs.provenance_path.write_text(
        json.dumps(provenance_document),
        encoding="utf-8",
    )

    # When/Then: non-null text alone cannot prove that the process consumed the prior checkpoint.
    with pytest.raises(TrainingEvidenceError, match="resume checkpoint SHA-256"):
        _ = validate_reference_sources(fixture.inputs)


def test_cross_artifact_validation_requires_final_scheduled_sample(tmp_path: Path) -> None:
    # Given: valid evidence whose final scheduled sample is absent.
    fixture = prepare_evidence_fixture(tmp_path)
    _ = fixture.inputs.samples_path.write_text(
        "step=1\nfirst sample\n\n",
        encoding="utf-8",
    )

    # When/Then: the sample schedule is enforced without fabricating a point.
    with pytest.raises(TrainingEvidenceError, match="sample"):
        _ = validate_reference_sources(fixture.inputs)


def test_training_series_preserve_observed_steps_and_values(tmp_path: Path) -> None:
    # Given: metrics with validation observed only at scheduled steps.
    fixture = prepare_evidence_fixture(tmp_path)
    evidence = validate_reference_sources(fixture.inputs)

    # When: pure plotting series and a title are derived.
    series = build_training_series(evidence.metrics)
    title = figure_title(
        "Loss",
        evidence.provenance.experiment_name,
        evidence.provenance.git.commit_sha,
    )

    # Then: no value is smoothed or interpolated and provenance appears in the title.
    assert series.train_loss == (
        (0, 4.0),
        (1, 3.75),
        (2, 3.5),
        (3, 3.25),
    )
    assert series.validation_loss == ((1, 3.4), (3, 3.2))
    assert series.learning_rate == tuple(
        (record.step, record.learning_rate) for record in evidence.metrics
    )
    assert series.tokens_per_sec == (
        (0, 100.0),
        (1, 101.0),
        (2, 102.0),
        (3, 103.0),
    )
    assert throughput_reference_line(series) == ("median tokens/s: 101.50", 101.5)
    assert title == f"Loss — reference @ {evidence.provenance.git.commit_sha[:8]}"


def assert_file_identity(repository: Path, raw_identity: JsonValue) -> None:
    assert isinstance(raw_identity, dict)
    assert set(raw_identity) == {"path", "sha256", "size_bytes"}
    relative_path = raw_identity["path"]
    digest = raw_identity["sha256"]
    size_bytes = raw_identity["size_bytes"]
    assert isinstance(relative_path, str)
    assert isinstance(digest, str)
    assert isinstance(size_bytes, int)
    artifact_path = repository / relative_path
    assert digest == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert size_bytes == artifact_path.stat().st_size


def assert_manifest_contract(
    fixture: EvidenceFixture,
    manifest_path: Path,
    environment: dict[str, JsonValue],
) -> None:
    manifest = cast(
        "JsonValue",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    assert isinstance(manifest, dict)
    manifest_artifacts = manifest["artifacts"]
    assert isinstance(manifest_artifacts, list)
    assert len(manifest_artifacts) == 8
    for raw_item in manifest_artifacts:
        assert_file_identity(fixture.repository, raw_item)
    generator = manifest["generator"]
    assert isinstance(generator, dict)
    assert generator == {
        "name": "minigpt.training_report",
        "version": 1,
        "git_sha": environment["git_commit_sha"],
    }
    assert manifest["checkpoint_format_version"] == 2
    assert manifest["final_completed_step"] == 3
    sources = manifest["sources"]
    assert isinstance(sources, dict)
    assert set(sources) == {
        "checkpoint",
        "config",
        "metrics",
        "provenance",
        "samples",
        "tokenizer",
        "train_data",
        "validation_data",
    }
    for source_name, raw_source in sources.items():
        assert isinstance(source_name, str)
        assert_file_identity(fixture.repository, raw_source)
    assert manifest["resolved_config_sha256"] == environment["resolved_config_sha256"]


def test_report_generator_writes_complete_verified_evidence_package(tmp_path: Path) -> None:
    # Given: complete validated source evidence and a new ignored destination.
    fixture = prepare_evidence_fixture(tmp_path)
    output_dir = fixture.repository / "outputs" / "generated-report"

    # When: the report is generated without hand-entered conclusions.
    artifacts = generate_training_report(fixture.inputs, output_dir)

    # Then: all required compact artifacts exist.
    artifact_paths = (
        artifacts.readme,
        artifacts.environment,
        artifacts.resolved_config,
        artifacts.metrics_csv,
        artifacts.loss_curve,
        artifacts.learning_rate_curve,
        artifacts.throughput_curve,
        artifacts.generated_samples,
        artifacts.manifest,
    )
    assert all(path.is_file() for path in artifact_paths)
    assert artifacts.generated_samples.read_bytes() == fixture.inputs.samples_path.read_bytes()

    with artifacts.metrics_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == [0, 1, 2, 3]
    assert [float(row["train_loss"]) for row in rows] == [4.0, 3.75, 3.5, 3.25]
    assert rows[0]["val_loss"] == ""
    assert float(rows[1]["val_loss"]) == 3.4

    environment = cast(
        "JsonValue",
        json.loads(artifacts.environment.read_text(encoding="utf-8")),
    )
    assert isinstance(environment, dict)
    expected_environment_keys = {
        "experiment_name",
        "git_commit_sha",
        "git_dirty",
        "run_started_at_utc",
        "run_ended_at_utc",
        "operating_system",
        "python_version",
        "pytorch_version",
        "numpy_version",
        "cpu_name",
        "physical_cores",
        "logical_cores",
        "torch_num_threads",
        "cuda_available",
        "dataset_fingerprints",
        "checkpoint_format_version",
        "model_parameter_count",
        "resolved_config_sha256",
    }
    assert expected_environment_keys <= set(environment)
    assert environment["git_dirty"] is False
    assert environment["checkpoint_format_version"] == 2

    assert_manifest_contract(fixture, artifacts.manifest, environment)

    readme = artifacts.readme.read_text(encoding="utf-8")
    assert "first / middle / final scheduled samples" in readme
    assert "checkpoint resume: yes" in readme
    assert "Deterministic local-structure observation" in readme
    assert "excludes validation, sampling, checkpoint, logging" in readme
    assert fixture.inputs.checkpoint_path.name in readme
    assert artifacts.loss_curve.stat().st_size > 0
    assert artifacts.learning_rate_curve.stat().st_size > 0
    assert artifacts.throughput_curve.stat().st_size > 0


def test_report_generator_rejects_existing_destination(tmp_path: Path) -> None:
    # Given: valid evidence and a destination that already contains unrelated bytes.
    fixture = prepare_evidence_fixture(tmp_path)
    output_dir = fixture.repository / "outputs" / "existing-report"
    output_dir.mkdir(parents=True)
    marker = output_dir / "keep.txt"
    _ = marker.write_text("do not mix", encoding="utf-8")

    # When/Then: report generation refuses to merge or overwrite evidence.
    with pytest.raises(ReportOutputExistsError):
        _ = generate_training_report(fixture.inputs, output_dir)
    assert marker.read_text(encoding="utf-8") == "do not mix"


def test_report_regeneration_is_stable_for_text_artifacts(tmp_path: Path) -> None:
    # Given: one immutable source set and two fresh ignored destinations.
    fixture = prepare_evidence_fixture(tmp_path)
    first = generate_training_report(
        fixture.inputs,
        fixture.repository / "outputs" / "report-a",
    )
    second = generate_training_report(
        fixture.inputs,
        fixture.repository / "outputs" / "report-b",
    )

    # When: deterministic text artifacts are compared.
    first_text_paths = (
        first.readme,
        first.environment,
        first.resolved_config,
        first.metrics_csv,
        first.generated_samples,
    )
    second_text_paths = (
        second.readme,
        second.environment,
        second.resolved_config,
        second.metrics_csv,
        second.generated_samples,
    )

    # Then: repeated generation preserves exact report conclusions and source text.
    assert [path.read_bytes() for path in first_text_paths] == [
        path.read_bytes() for path in second_text_paths
    ]


def test_report_cli_generates_package_from_explicit_sources(tmp_path: Path) -> None:
    # Given: valid evidence and the repository's typed report CLI.
    fixture = prepare_evidence_fixture(tmp_path)
    output_dir = fixture.repository / "outputs" / "cli-report"
    cli_path = PROJECT_ROOT / "report_training.py"

    # When: all sources are passed explicitly in a separate process.
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(cli_path),
            "--config",
            str(fixture.inputs.config_path),
            "--metrics",
            str(fixture.inputs.metrics_path),
            "--samples",
            str(fixture.inputs.samples_path),
            "--checkpoint",
            str(fixture.inputs.checkpoint_path),
            "--provenance",
            str(fixture.inputs.provenance_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then: the CLI reports the durable Markdown and manifest paths.
    assert completed.returncode == 0, completed.stderr
    assert str(output_dir / "README.md") in completed.stdout
    assert str(output_dir / "artifact_manifest.json") in completed.stdout


def test_committed_report_text_artifacts_preserve_generated_line_endings() -> None:
    # Given: generated artifacts whose hashes cover their original Windows newline bytes.
    artifact_paths = [
        (REFERENCE_RESULT_DIRECTORY / name).as_posix() for name in REFERENCE_TEXT_ARTIFACT_EOL
    ]

    # When: Git's checkout attributes are resolved for every committed text artifact.
    command = [
        "git",
        "-C",
        str(PROJECT_ROOT),
        "check-attr",
        "eol",
        "--",
        *artifact_paths,
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    # Then: core.autocrlf cannot rewrite evidence bytes and invalidate the manifest.
    lines = completed.stdout.splitlines()
    assert len(lines) == len(artifact_paths)
    expected_eol = [
        f": eol: {REFERENCE_TEXT_ARTIFACT_EOL[name]}" for name in REFERENCE_TEXT_ARTIFACT_EOL
    ]
    assert all(line.endswith(expected) for line, expected in zip(lines, expected_eol, strict=True))


def test_committed_reference_manifest_verifies_reviewable_evidence() -> None:
    # Given: the compact evidence package committed without its large raw sources.
    manifest_path = PROJECT_ROOT / REFERENCE_RESULT_DIRECTORY / "artifact_manifest.json"
    manifest = cast(
        "JsonValue",
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    assert isinstance(manifest, dict)
    artifacts = manifest["artifacts"]
    sources = manifest["sources"]
    generator = manifest["generator"]
    assert isinstance(artifacts, list)
    assert isinstance(sources, dict)
    assert isinstance(generator, dict)

    # When/Then: every committed artifact matches its declared identity and raw inputs remain bound
    # by durable repository-relative paths, sizes, and SHA-256 values.
    assert len(artifacts) == 8
    for raw_artifact in artifacts:
        assert_file_identity(PROJECT_ROOT, raw_artifact)
    assert set(sources) == {
        "checkpoint",
        "config",
        "metrics",
        "provenance",
        "samples",
        "tokenizer",
        "train_data",
        "validation_data",
    }
    for raw_source in sources.values():
        assert isinstance(raw_source, dict)
        assert set(raw_source) == {"path", "sha256", "size_bytes"}
        assert isinstance(raw_source["path"], str)
        assert isinstance(raw_source["sha256"], str)
        assert len(raw_source["sha256"]) == 64
        assert isinstance(raw_source["size_bytes"], int)
        assert raw_source["size_bytes"] > 0
    assert generator == {
        "name": "minigpt.training_report",
        "version": 1,
        "git_sha": "ed7da8722d6fed89b1b05ed99c8fc7ca4e6231d6",
    }
    assert manifest["checkpoint_format_version"] == 2
    assert manifest["final_completed_step"] == 2799
