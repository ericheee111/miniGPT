from __future__ import annotations

import random
from dataclasses import replace
from hashlib import sha256
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pytest
import torch

from minigpt import checkpoint
from minigpt.checkpoint import CheckpointResources
from minigpt.data import TokenBatcher
from minigpt.model import GPT, GPTConfig
from minigpt.settings import (
    DataSettings,
    ExperimentConfig,
    ModelSettings,
    OptimizerSettings,
    RuntimeSettings,
    TrainingSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

_IMMUTABLE_CONFIG_FIELDS: Final = (
    "runtime.seed",
    "runtime.num_threads",
    "data.block_size",
    "data.batch_size",
    "model.vocab_size",
    "model.n_layer",
    "model.n_head",
    "model.n_embd",
    "model.dropout",
    "model.bias",
    "optimizer.learning_rate",
    "optimizer.min_learning_rate",
    "optimizer.weight_decay",
    "optimizer.beta1",
    "optimizer.beta2",
    "optimizer.grad_clip",
    "training.max_steps",
    "training.warmup_steps",
    "training.lr_decay_steps",
    "training.eval_interval",
    "training.eval_batches",
)


def experiment_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        runtime=RuntimeSettings(seed=7, num_threads=1, device="cpu"),
        data=DataSettings(directory=tmp_path, block_size=4, batch_size=2),
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
            max_steps=4,
            warmup_steps=1,
            lr_decay_steps=4,
            eval_interval=2,
            eval_batches=1,
            log_interval=1,
            checkpoint_interval=2,
            sample_interval=2,
            sample_tokens=4,
            sample_prompt="a",
            output_dir=tmp_path / "outputs",
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=tmp_path / "tensorboard",
        ),
    )


def write_dataset_artifacts(directory: Path) -> None:
    _ = (directory / "tokenizer.json").write_bytes(b"tokenizer-v1")
    _ = (directory / "train.npy").write_bytes(b"train-v1")
    _ = (directory / "val.npy").write_bytes(b"validation-v1")


def build_checkpoint_resources(
    config: ExperimentConfig,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_batcher: TokenBatcher,
    val_batcher: TokenBatcher,
) -> CheckpointResources:
    sample_generator = torch.Generator(device="cpu")
    _ = sample_generator.manual_seed(config.runtime.seed + 2)
    return CheckpointResources(
        model=model,
        optimizer=optimizer,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
        sample_generator=sample_generator,
        dataset_fingerprints=checkpoint.compute_dataset_fingerprints(config.data),
    )


def write_v1_checkpoint(
    path: Path,
    *,
    config: ExperimentConfig,
    resources: CheckpointResources,
) -> None:
    legacy_yaml = (
        config.to_yaml()
        .replace("  type: adamw\n", "")
        .replace(
            f"  lr_decay_steps: {config.training.lr_decay_steps}\n",
            "",
        )
    )
    legacy_payload: dict[str, checkpoint.StateValue] = {
        "format_version": 1,
        "step": 0,
        "config_yaml": legacy_yaml,
        "model_state": cast(
            "dict[str | int, checkpoint.StateValue]",
            resources.model.state_dict(),
        ),
        "optimizer_state": cast(
            "dict[str | int, checkpoint.StateValue]",
            resources.optimizer.state_dict(),
        ),
        "python_random_state": cast("checkpoint.PythonRandomState", random.getstate()),
        "numpy_random_state_json": "{}",
        "torch_random_state": torch.get_rng_state(),
        "train_batcher_random_state": resources.train_batcher.capture_random_state(),
        "val_batcher_random_state": resources.val_batcher.capture_random_state(),
    }
    torch.save(legacy_payload, path)


def _change_runtime_field(config: ExperimentConfig, field: str) -> ExperimentConfig:
    settings = config.runtime
    match field:
        case "runtime.seed":
            settings = replace(settings, seed=8)
        case "runtime.num_threads":
            settings = replace(settings, num_threads=2)
        case _:
            raise AssertionError(field)
    return replace(config, runtime=settings)


def _change_data_field(config: ExperimentConfig, field: str) -> ExperimentConfig:
    settings = config.data
    match field:
        case "data.block_size":
            settings = replace(settings, block_size=5)
        case "data.batch_size":
            settings = replace(settings, batch_size=3)
        case _:
            raise AssertionError(field)
    return replace(config, data=settings)


def _change_model_field(config: ExperimentConfig, field: str) -> ExperimentConfig:
    settings = config.model
    match field:
        case "model.vocab_size":
            settings = replace(settings, vocab_size=12)
        case "model.n_layer":
            settings = replace(settings, n_layer=2)
        case "model.n_head":
            settings = replace(settings, n_head=2)
        case "model.n_embd":
            settings = replace(settings, n_embd=16)
        case "model.dropout":
            settings = replace(settings, dropout=0.1)
        case "model.bias":
            settings = replace(settings, bias=True)
        case _:
            raise AssertionError(field)
    return replace(config, model=settings)


def _change_optimizer_field(config: ExperimentConfig, field: str) -> ExperimentConfig:
    settings = config.optimizer
    match field:
        case "optimizer.learning_rate":
            settings = replace(settings, learning_rate=2e-3)
        case "optimizer.min_learning_rate":
            settings = replace(settings, min_learning_rate=2e-4)
        case "optimizer.weight_decay":
            settings = replace(settings, weight_decay=0.02)
        case "optimizer.beta1":
            settings = replace(settings, beta1=0.8)
        case "optimizer.beta2":
            settings = replace(settings, beta2=0.9)
        case "optimizer.grad_clip":
            settings = replace(settings, grad_clip=0.5)
        case _:
            raise AssertionError(field)
    return replace(config, optimizer=settings)


def _change_training_field(config: ExperimentConfig, field: str) -> ExperimentConfig:
    settings = config.training
    match field:
        case "training.max_steps":
            settings = replace(settings, max_steps=5)
        case "training.warmup_steps":
            settings = replace(settings, warmup_steps=0)
        case "training.lr_decay_steps":
            settings = replace(settings, lr_decay_steps=3)
        case "training.eval_interval":
            settings = replace(settings, eval_interval=3)
        case "training.eval_batches":
            settings = replace(settings, eval_batches=2)
        case _:
            raise AssertionError(field)
    return replace(config, training=settings)


def change_immutable_field(config: ExperimentConfig, field: str) -> ExperimentConfig:
    if field.startswith("runtime."):
        return _change_runtime_field(config, field)
    if field.startswith("data."):
        return _change_data_field(config, field)
    if field.startswith("model."):
        return _change_model_field(config, field)
    if field.startswith("optimizer."):
        return _change_optimizer_field(config, field)
    return _change_training_field(config, field)


def save_test_checkpoint(
    tmp_path: Path,
) -> tuple[ExperimentConfig, CheckpointResources, Path]:
    write_dataset_artifacts(tmp_path)
    config = experiment_config(tmp_path)
    model = GPT(config.model.to_gpt_config(config.data.block_size))
    resources = build_checkpoint_resources(
        config,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        train_batcher=TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=1),
        val_batcher=TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=2),
    )
    checkpoint_path = tmp_path / "latest.pt"
    checkpoint.save_checkpoint(
        checkpoint_path,
        resources=resources,
        step=1,
        config=config,
    )
    return config, resources, checkpoint_path


def test_compute_dataset_fingerprints_hashes_each_persisted_artifact(tmp_path: Path) -> None:
    # Given: three persisted dataset artifacts with independently known bytes.
    tokenizer_bytes = b'{"version": 1, "vocabulary": ["a"]}\n'
    train_bytes = b"train-array-bytes"
    val_bytes = b"validation-array-bytes"
    _ = (tmp_path / "tokenizer.json").write_bytes(tokenizer_bytes)
    _ = (tmp_path / "train.npy").write_bytes(train_bytes)
    _ = (tmp_path / "val.npy").write_bytes(val_bytes)

    # When: the dataset identity is computed.
    fingerprints = checkpoint.compute_dataset_fingerprints(experiment_config(tmp_path).data)

    # Then: every named fingerprint is the SHA-256 of its exact file bytes.
    assert fingerprints.tokenizer_sha256 == sha256(tokenizer_bytes).hexdigest()
    assert fingerprints.train_sha256 == sha256(train_bytes).hexdigest()
    assert fingerprints.val_sha256 == sha256(val_bytes).hexdigest()


def test_checkpoint_restores_model_optimizer_and_step(tmp_path: Path) -> None:
    # Given: a model and optimizer saved after one update.
    write_dataset_artifacts(tmp_path)
    config = experiment_config(tmp_path)
    gpt = GPT(GPTConfig(vocab_size=11, block_size=4, n_layer=1, n_head=1, n_embd=8))
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    train_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=1)
    val_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=2)
    resources = build_checkpoint_resources(
        config,
        model=gpt,
        optimizer=optimizer,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
    )
    checkpoint_path = tmp_path / "latest.pt"
    saved_parameter = next(gpt.parameters()).detach().clone()
    checkpoint.save_checkpoint(
        checkpoint_path,
        resources=resources,
        step=3,
        config=config,
    )
    with torch.no_grad():
        _ = next(gpt.parameters()).add_(10.0)

    # When: the checkpoint is loaded into the existing objects.
    resume_state = checkpoint.load_checkpoint(
        checkpoint_path,
        resources=resources,
        config=config,
    )

    # Then: parameters and the next training step are restored.
    assert torch.equal(next(gpt.parameters()), saved_parameter)
    assert resume_state.next_step == 4


def test_checkpoint_restores_random_and_batcher_states(tmp_path: Path) -> None:
    # Given: global RNGs and batchers captured in a checkpoint.
    write_dataset_artifacts(tmp_path)
    config = experiment_config(tmp_path)
    random.seed(10)
    np.random.seed(11)  # noqa: NPY002
    _ = torch.default_generator.manual_seed(12)
    gpt = GPT(GPTConfig(vocab_size=11, block_size=4, n_layer=1, n_head=1, n_embd=8))
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    train_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=13)
    val_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=14)
    resources = build_checkpoint_resources(
        config,
        model=gpt,
        optimizer=optimizer,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
    )
    checkpoint_path = tmp_path / "latest.pt"
    checkpoint.save_checkpoint(
        checkpoint_path,
        resources=resources,
        step=0,
        config=config,
    )
    expected_python = random.random()  # noqa: S311
    expected_numpy = float(np.random.random())  # noqa: NPY002
    expected_torch = torch.rand(3)
    expected_batch = train_batcher.next_batch()
    sample_probabilities = torch.tensor([[0.2, 0.3, 0.5]])
    expected_sample = torch.multinomial(
        sample_probabilities,
        num_samples=1,
        generator=resources.sample_generator,
    )

    # When: all random states are restored.
    _ = checkpoint.load_checkpoint(
        checkpoint_path,
        resources=resources,
        config=config,
    )

    # Then: subsequent random values and sampled batches are identical.
    assert random.random() == expected_python  # noqa: S311
    assert float(np.random.random()) == expected_numpy  # noqa: NPY002
    assert torch.equal(torch.rand(3), expected_torch)
    actual_batch = train_batcher.next_batch()
    assert torch.equal(actual_batch[0], expected_batch[0])
    assert torch.equal(actual_batch[1], expected_batch[1])
    actual_sample = torch.multinomial(
        sample_probabilities,
        num_samples=1,
        generator=resources.sample_generator,
    )
    assert torch.equal(actual_sample, expected_sample)


def test_v1_checkpoint_supports_config_and_model_loading_only(tmp_path: Path) -> None:
    # Given: a legacy checkpoint without v2 sample or dataset identity state.
    write_dataset_artifacts(tmp_path)
    config = experiment_config(tmp_path)
    gpt = GPT(config.model.to_gpt_config(config.data.block_size))
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    resources = build_checkpoint_resources(
        config,
        model=gpt,
        optimizer=optimizer,
        train_batcher=TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=1),
        val_batcher=TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=2),
    )
    checkpoint_path = tmp_path / "legacy.pt"
    write_v1_checkpoint(checkpoint_path, config=config, resources=resources)
    saved_parameter = next(gpt.parameters()).detach().clone()
    fresh_model = GPT(config.model.to_gpt_config(config.data.block_size))

    # When: inference metadata and weights are loaded.
    loaded_config = checkpoint.load_checkpoint_config(checkpoint_path)
    checkpoint.load_model_state(checkpoint_path, fresh_model)

    # Then: v1 defaults are normalized and model weights remain usable.
    assert loaded_config.optimizer.optimizer_type == "adamw"
    assert loaded_config.training.lr_decay_steps == loaded_config.training.max_steps
    assert torch.equal(next(fresh_model.parameters()), saved_parameter)

    # When/Then: training resume rejects v1 before mutating the current model.
    current_parameter = next(gpt.parameters()).detach().clone()
    with pytest.raises(RuntimeError, match=r"v1.*training resume") as error_info:
        _ = checkpoint.load_checkpoint(checkpoint_path, resources=resources, config=config)
    assert type(error_info.value).__name__ == "LegacyCheckpointResumeError"
    assert torch.equal(next(gpt.parameters()), current_parameter)


@pytest.mark.parametrize("field", _IMMUTABLE_CONFIG_FIELDS)
def test_resume_rejects_each_immutable_config_change_before_mutation(
    tmp_path: Path,
    field: str,
) -> None:
    # Given: a v2 checkpoint and live state changed after it was saved.
    config, resources, checkpoint_path = save_test_checkpoint(tmp_path)
    with torch.no_grad():
        _ = next(resources.model.parameters()).add_(1.0)
    parameter_before_load = next(resources.model.parameters()).detach().clone()
    generator_before_load = resources.sample_generator.get_state().clone()

    # When/Then: each trajectory-defining change is named and rejected before mutation.
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        _ = checkpoint.load_checkpoint(
            checkpoint_path,
            resources=resources,
            config=change_immutable_field(config, field),
        )
    assert torch.equal(next(resources.model.parameters()), parameter_before_load)
    assert torch.equal(resources.sample_generator.get_state(), generator_before_load)


@pytest.mark.parametrize(
    ("artifact_name", "expected_field"),
    [
        ("tokenizer.json", "dataset.tokenizer_sha256"),
        ("train.npy", "dataset.train_sha256"),
        ("val.npy", "dataset.val_sha256"),
    ],
)
def test_resume_rejects_each_dataset_fingerprint_change_before_mutation(
    tmp_path: Path,
    artifact_name: str,
    expected_field: str,
) -> None:
    # Given: a v2 checkpoint whose current dataset artifact has changed.
    config, resources, checkpoint_path = save_test_checkpoint(tmp_path)
    _ = (tmp_path / artifact_name).write_bytes(b"changed")
    changed_resources = replace(
        resources,
        dataset_fingerprints=checkpoint.compute_dataset_fingerprints(config.data),
    )
    parameter_before_load = next(resources.model.parameters()).detach().clone()

    # When/Then: the exact changed identity is rejected before model mutation.
    with pytest.raises(ValueError, match=expected_field.replace(".", r"\.")):
        _ = checkpoint.load_checkpoint(
            checkpoint_path,
            resources=changed_resources,
            config=config,
        )
    assert torch.equal(next(resources.model.parameters()), parameter_before_load)


def test_resume_allows_operational_and_sampling_config_changes(tmp_path: Path) -> None:
    # Given: byte-identical data in a new directory and changed non-trajectory settings.
    config, resources, checkpoint_path = save_test_checkpoint(tmp_path)
    copied_data = tmp_path / "copied-data"
    copied_data.mkdir()
    for artifact_name in ("tokenizer.json", "train.npy", "val.npy"):
        _ = (copied_data / artifact_name).write_bytes((tmp_path / artifact_name).read_bytes())
    changed_config = replace(
        config,
        data=replace(config.data, directory=copied_data),
        training=replace(
            config.training,
            output_dir=tmp_path / "different-output",
            checkpoint_dir=tmp_path / "different-checkpoints",
            tensorboard_dir=tmp_path / "different-runs",
            log_interval=2,
            checkpoint_interval=3,
            sample_interval=3,
            sample_tokens=5,
            sample_prompt="b",
        ),
    )
    changed_resources = replace(
        resources,
        dataset_fingerprints=checkpoint.compute_dataset_fingerprints(changed_config.data),
    )

    # When: only allowlisted operational and isolated-sampling fields change.
    resume_state = checkpoint.load_checkpoint(
        checkpoint_path,
        resources=changed_resources,
        config=changed_config,
    )

    # Then: the same experiment trajectory resumes at the next step.
    assert resume_state.next_step == 2


def test_checkpoint_metadata_exposes_validated_v2_identity(tmp_path: Path) -> None:
    # Given: a v2 checkpoint saved at completed step one.
    config, resources, checkpoint_path = save_test_checkpoint(tmp_path)

    # When: read-only checkpoint metadata is loaded.
    metadata = checkpoint.load_checkpoint_metadata(checkpoint_path)

    # Then: completion, resolved config, and all dataset identities are preserved.
    assert metadata.format_version == 2
    assert metadata.completed_step == 1
    assert metadata.config == config
    assert metadata.dataset_fingerprints == resources.dataset_fingerprints


def test_checkpoint_metadata_rejects_v1_training_evidence(tmp_path: Path) -> None:
    # Given: a legacy inference-only checkpoint.
    write_dataset_artifacts(tmp_path)
    config = experiment_config(tmp_path)
    model = GPT(config.model.to_gpt_config(config.data.block_size))
    resources = build_checkpoint_resources(
        config,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        train_batcher=TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=1),
        val_batcher=TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=2),
    )
    checkpoint_path = tmp_path / "legacy.pt"
    write_v1_checkpoint(checkpoint_path, config=config, resources=resources)

    # When/Then: v1 cannot serve as reference-training evidence.
    with pytest.raises(checkpoint.LegacyCheckpointResumeError):
        _ = checkpoint.load_checkpoint_metadata(checkpoint_path)
