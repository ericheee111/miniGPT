from __future__ import annotations

import subprocess
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
import torch

from minigpt.batching import TokenBatcher
from minigpt.checkpoint import (
    CheckpointResources,
    compute_dataset_fingerprints,
    save_checkpoint,
)
from minigpt.model import GPT
from minigpt.run_provenance import (
    DirtyReferenceRunError,
    IncompatibleRunProvenanceError,
    RunInvocation,
    RunSegment,
    begin_run_segment,
    complete_run_segment,
    fail_run_segment,
    load_run_provenance,
)
from minigpt.settings import (
    DataSettings,
    ExperimentConfig,
    ModelSettings,
    OptimizerSettings,
    RuntimeSettings,
    TrainingSettings,
)


class TrainingCommand(Protocol):
    """Expose the typed command entrypoint exercised by provenance tests."""

    def main(self, argv: tuple[str, ...] | None = None) -> int:
        """Run the training command with explicit command-line arguments."""
        ...


def load_training_command() -> TrainingCommand:
    """Load the root training command without relying on the editable package path."""
    path = Path(__file__).parent.parent / "train.py"
    specification = spec_from_file_location("test_run_provenance_train", path)
    assert specification is not None
    assert specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return cast("TrainingCommand", cast("object", module))


def run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", "-C", str(repository), *arguments],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def reference_config(repository: Path, *, seed: int = 7) -> ExperimentConfig:
    return ExperimentConfig(
        runtime=RuntimeSettings(seed=seed, num_threads=1, device="cpu"),
        data=DataSettings(directory=repository / "data", block_size=4, batch_size=2),
        model=ModelSettings(
            vocab_size=11,
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
            max_steps=2,
            warmup_steps=0,
            lr_decay_steps=2,
            eval_interval=1,
            eval_batches=1,
            log_interval=1,
            checkpoint_interval=1,
            sample_interval=1,
            sample_tokens=2,
            sample_prompt="a",
            output_dir=repository / "outputs" / "reference",
            checkpoint_dir=repository / "checkpoints" / "reference",
            tensorboard_dir=repository / "outputs" / "reference" / "tensorboard",
        ),
    )


def prepare_repository(tmp_path: Path) -> tuple[Path, Path, ExperimentConfig]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _ = (repository / ".gitignore").write_text(
        "data/\noutputs/\ncheckpoints/\n",
        encoding="utf-8",
    )
    data_dir = repository / "data"
    data_dir.mkdir()
    _ = (data_dir / "tokenizer.json").write_bytes(b"tokenizer")
    _ = (data_dir / "train.npy").write_bytes(b"train")
    _ = (data_dir / "val.npy").write_bytes(b"validation")
    config = reference_config(repository)
    config_path = repository / "reference.yaml"
    _ = config_path.write_text(config.to_yaml(), encoding="utf-8")
    alternate = reference_config(repository, seed=8)
    _ = (repository / "alternate.yaml").write_text(alternate.to_yaml(), encoding="utf-8")
    _ = run_git(repository, "init")
    _ = run_git(repository, "config", "user.email", "tests@example.com")
    _ = run_git(repository, "config", "user.name", "miniGPT tests")
    _ = run_git(repository, "add", ".gitignore", "reference.yaml", "alternate.yaml")
    _ = run_git(repository, "commit", "-m", "test fixture")
    return repository, config_path, config


def save_v2_checkpoint(
    repository: Path,
    config: ExperimentConfig,
    *,
    completed_step: int,
) -> Path:
    model = GPT(config.model.to_gpt_config(config.data.block_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sample_generator = torch.Generator(device="cpu")
    _ = sample_generator.manual_seed(config.runtime.seed + 2)
    resources = CheckpointResources(
        model=model,
        optimizer=optimizer,
        train_batcher=TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=1),
        val_batcher=TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=2),
        sample_generator=sample_generator,
        dataset_fingerprints=compute_dataset_fingerprints(config.data),
    )
    path = repository / "checkpoints" / "reference" / "latest.pt"
    save_checkpoint(path, resources=resources, step=completed_step, config=config)
    return path


def test_provenance_records_clean_environment_and_completed_resume(tmp_path: Path) -> None:
    # Given: a clean repository, fixed data, and a v2 checkpoint.
    repository, config_path, config = prepare_repository(tmp_path)
    provenance_path = repository / "outputs" / "reference" / "run_provenance.json"
    checkpoint_path = save_v2_checkpoint(repository, config, completed_step=0)

    # When: one bounded segment completes and a second segment resumes from its checkpoint.
    first = begin_run_segment(
        provenance_path,
        config_path=config_path,
        config=config,
        invocation=RunInvocation(
            argv=("--config", "reference.yaml", "--run-until-step", "1"),
            run_until_step=1,
            resume_path=None,
        ),
    )
    complete_run_segment(
        provenance_path,
        segment=first,
        checkpoint_path=checkpoint_path,
        final_step=0,
    )
    second = begin_run_segment(
        provenance_path,
        config_path=config_path,
        config=config,
        invocation=RunInvocation(
            argv=("--config", "reference.yaml", "--resume", "checkpoints/reference/latest.pt"),
            run_until_step=None,
            resume_path=checkpoint_path,
        ),
    )
    checkpoint_path = save_v2_checkpoint(repository, config, completed_step=1)
    complete_run_segment(
        provenance_path,
        segment=second,
        checkpoint_path=checkpoint_path,
        final_step=1,
    )

    # Then: the journal proves clean identity, concrete environment, and a real resume.
    provenance = load_run_provenance(provenance_path)
    assert provenance.schema_version == 1
    assert provenance.experiment_name == "reference"
    assert provenance.config_path == "reference.yaml"
    assert len(provenance.git.commit_sha) == 40
    assert provenance.git.dirty is False
    assert provenance.environment.python_version
    assert provenance.environment.pytorch_version
    assert provenance.environment.numpy_version
    assert provenance.environment.cpu_name
    assert provenance.environment.physical_cores is not None
    assert provenance.environment.logical_cores is not None
    assert provenance.environment.torch_num_threads > 0
    assert provenance.environment.cuda_available is False
    assert provenance.started_at_utc.endswith("Z")
    assert provenance.ended_at_utc is not None
    assert provenance.ended_at_utc.endswith("Z")
    assert provenance.resolved_config_sha256 is not None
    assert len(provenance.segments) == 2
    assert provenance.segments[0].status == "completed"
    assert provenance.segments[0].final_completed_step == 0
    assert provenance.segments[0].resume_checkpoint_sha256 is None
    assert provenance.segments[1].status == "completed"
    assert provenance.segments[1].final_completed_step == 1
    assert provenance.segments[1].resume_checkpoint_sha256 is not None
    assert provenance.segments[1].checkpoint_sha256 is not None


def test_provenance_rejects_dirty_repository_before_recording(tmp_path: Path) -> None:
    # Given: a tracked canonical config modified after its commit.
    repository, config_path, config = prepare_repository(tmp_path)
    _ = config_path.write_text(f"{config.to_yaml()}\n", encoding="utf-8")

    # When/Then: no reference segment starts from an uncommitted tree.
    with pytest.raises(DirtyReferenceRunError):
        _ = begin_run_segment(
            repository / "outputs" / "run.json",
            config_path=config_path,
            config=config,
            invocation=RunInvocation(
                argv=("--config", "reference.yaml"),
                run_until_step=None,
                resume_path=None,
            ),
        )


def test_resume_rejects_changed_commit_config_or_data_identity(tmp_path: Path) -> None:
    # Given: one completed segment bound to a commit, config, and ignored dataset bytes.
    repository, config_path, config = prepare_repository(tmp_path)
    provenance_path = repository / "outputs" / "reference" / "run.json"
    checkpoint_path = save_v2_checkpoint(repository, config, completed_step=0)
    segment = begin_run_segment(
        provenance_path,
        config_path=config_path,
        config=config,
        invocation=RunInvocation(
            argv=("--config", "reference.yaml", "--run-until-step", "1"),
            run_until_step=1,
            resume_path=None,
        ),
    )
    complete_run_segment(
        provenance_path,
        segment=segment,
        checkpoint_path=checkpoint_path,
        final_step=0,
    )

    # When/Then: a different committed config is rejected.
    alternate_path = repository / "alternate.yaml"
    with pytest.raises(IncompatibleRunProvenanceError, match="config"):
        _ = begin_run_segment(
            provenance_path,
            config_path=alternate_path,
            config=reference_config(repository, seed=8),
            invocation=RunInvocation(
                argv=("--config", "alternate.yaml", "--resume", str(checkpoint_path)),
                run_until_step=None,
                resume_path=checkpoint_path,
            ),
        )

    # And: changed ignored data is rejected without relying on Git dirty state.
    _ = (repository / "data" / "train.npy").write_bytes(b"changed-train")
    with pytest.raises(IncompatibleRunProvenanceError, match="train"):
        _ = begin_run_segment(
            provenance_path,
            config_path=config_path,
            config=config,
            invocation=RunInvocation(
                argv=("--config", "reference.yaml", "--resume", str(checkpoint_path)),
                run_until_step=None,
                resume_path=checkpoint_path,
            ),
        )

    # And: a different clean commit is rejected.
    _ = (repository / "data" / "train.npy").write_bytes(b"train")
    _ = run_git(repository, "commit", "--allow-empty", "-m", "different commit")
    with pytest.raises(IncompatibleRunProvenanceError, match="commit"):
        _ = begin_run_segment(
            provenance_path,
            config_path=config_path,
            config=config,
            invocation=RunInvocation(
                argv=("--config", "reference.yaml", "--resume", str(checkpoint_path)),
                run_until_step=None,
                resume_path=checkpoint_path,
            ),
        )


def test_resume_rejects_same_commit_on_different_branch(tmp_path: Path) -> None:
    # Given: one completed segment and the same commit checked out under another branch name.
    repository, config_path, config = prepare_repository(tmp_path)
    provenance_path = repository / "outputs" / "reference" / "run.json"
    checkpoint_path = save_v2_checkpoint(repository, config, completed_step=0)
    segment = begin_run_segment(
        provenance_path,
        config_path=config_path,
        config=config,
        invocation=RunInvocation(
            argv=("--config", "reference.yaml", "--run-until-step", "1"),
            run_until_step=1,
            resume_path=None,
        ),
    )
    complete_run_segment(
        provenance_path,
        segment=segment,
        checkpoint_path=checkpoint_path,
        final_step=0,
    )
    _ = run_git(repository, "checkout", "-b", "alternate-branch")

    # When/Then: the run journal preserves the recorded branch identity.
    with pytest.raises(IncompatibleRunProvenanceError, match="branch"):
        _ = begin_run_segment(
            provenance_path,
            config_path=config_path,
            config=config,
            invocation=RunInvocation(
                argv=("--config", "reference.yaml", "--resume", str(checkpoint_path)),
                run_until_step=None,
                resume_path=checkpoint_path,
            ),
        )


def test_failed_segment_records_end_without_claiming_completion(tmp_path: Path) -> None:
    # Given: a clean first segment recorded as running.
    repository, config_path, config = prepare_repository(tmp_path)
    provenance_path = repository / "outputs" / "reference" / "run.json"
    segment = begin_run_segment(
        provenance_path,
        config_path=config_path,
        config=config,
        invocation=RunInvocation(
            argv=("--config", "reference.yaml"),
            run_until_step=None,
            resume_path=None,
        ),
    )

    # When: the caller records training failure.
    fail_run_segment(provenance_path, segment=segment)

    # Then: the segment has an end time but no successful checkpoint claim.
    failed = load_run_provenance(provenance_path).segments[-1]
    assert failed.status == "failed"
    assert failed.ended_at_utc is not None
    assert failed.final_completed_step is None
    assert failed.checkpoint_sha256 is None


def test_train_main_finalizes_provenance_after_ordinary_training_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a reference invocation whose trainer raises an ordinary exception.
    repository, config_path, _ = prepare_repository(tmp_path)
    provenance_path = repository / "outputs" / "reference" / "run.json"
    expected_error = TypeError("boom")
    training_command = load_training_command()

    def raise_type_error(*_args: object, **_kwargs: object) -> None:
        raise expected_error

    monkeypatch.setattr(training_command, "run_training", raise_type_error)

    # When: the training command exits through that failure.
    with pytest.raises(TypeError) as caught:
        _ = training_command.main(
            ("--config", str(config_path), "--provenance", str(provenance_path))
        )

    # Then: the original exception and traceback propagate after a durable failed terminal segment.
    failed = load_run_provenance(provenance_path).segments[-1]
    assert caught.value is expected_error
    assert caught.traceback[-1].name == "raise_type_error"
    assert failed.status == "failed"
    assert failed.ended_at_utc is not None


def test_train_main_finalizes_provenance_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a reference invocation whose trainer receives a keyboard interrupt.
    repository, config_path, _ = prepare_repository(tmp_path)
    provenance_path = repository / "outputs" / "reference" / "run.json"
    expected_interrupt = KeyboardInterrupt()
    training_command = load_training_command()

    def raise_keyboard_interrupt(*_args: object, **_kwargs: object) -> None:
        raise expected_interrupt

    monkeypatch.setattr(training_command, "run_training", raise_keyboard_interrupt)

    # When: the training command exits through that interrupt.
    with pytest.raises(KeyboardInterrupt) as caught:
        _ = training_command.main(
            ("--config", str(config_path), "--provenance", str(provenance_path))
        )

    # Then: the interrupt and traceback propagate after a durable failed terminal segment.
    failed = load_run_provenance(provenance_path).segments[-1]
    assert caught.value is expected_interrupt
    assert caught.traceback[-1].name == "raise_keyboard_interrupt"
    assert failed.status == "failed"
    assert failed.ended_at_utc is not None


def test_train_main_preserves_training_failure_when_provenance_finalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: training and its provenance finalization both raise ordinary exceptions.
    repository, config_path, _ = prepare_repository(tmp_path)
    provenance_path = repository / "outputs" / "reference" / "run.json"
    expected_error = TypeError("boom")
    finalization_message = "provenance write failed"
    training_command = load_training_command()

    def raise_type_error(*_args: object, **_kwargs: object) -> None:
        raise expected_error

    def raise_provenance_error(_path: Path, *, segment: RunSegment) -> None:
        _ = segment
        raise OSError(finalization_message)

    monkeypatch.setattr(training_command, "run_training", raise_type_error)
    monkeypatch.setattr(training_command, "fail_run_segment", raise_provenance_error)

    # When: the training command attempts to finalize its failed provenance segment.
    with pytest.raises(TypeError) as caught:
        _ = training_command.main(
            ("--config", str(config_path), "--provenance", str(provenance_path))
        )

    # Then: best-effort provenance cannot replace the original training failure.
    assert caught.value is expected_error
    assert caught.traceback[-1].name == "raise_type_error"
