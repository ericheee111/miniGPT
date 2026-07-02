import random
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

import minigpt.config as config
from minigpt.data import TokenBatcher
from minigpt.model import GPT, GPTConfig


def experiment_config(tmp_path: Path) -> config.ExperimentConfig:
    return config.ExperimentConfig(
        runtime=config.RuntimeSettings(seed=7, num_threads=1, device="cpu"),
        data=config.DataSettings(directory=tmp_path, block_size=4, batch_size=2),
        model=config.ModelSettings(
            vocab_size=11,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        ),
        optimizer=config.OptimizerSettings(
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.95,
            grad_clip=1.0,
        ),
        training=config.TrainingSettings(
            max_steps=4,
            warmup_steps=1,
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


def test_checkpoint_restores_model_optimizer_and_step(tmp_path: Path) -> None:
    # Given: a model and optimizer saved after one update.
    checkpoint = import_module("minigpt.checkpoint")
    gpt = GPT(GPTConfig(vocab_size=11, block_size=4, n_layer=1, n_head=1, n_embd=8))
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    train_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=1)
    val_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=2)
    checkpoint_path = tmp_path / "latest.pt"
    saved_parameter = next(gpt.parameters()).detach().clone()
    checkpoint.save_checkpoint(
        checkpoint_path,
        model=gpt,
        optimizer=optimizer,
        step=3,
        config=experiment_config(tmp_path),
        train_batcher=train_batcher,
        val_batcher=val_batcher,
    )
    with torch.no_grad():
        next(gpt.parameters()).add_(10.0)

    # When: the checkpoint is loaded into the existing objects.
    resume_state = checkpoint.load_checkpoint(
        checkpoint_path,
        model=gpt,
        optimizer=optimizer,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
    )

    # Then: parameters and the next training step are restored.
    assert torch.equal(next(gpt.parameters()), saved_parameter)
    assert resume_state.next_step == 4


def test_checkpoint_restores_random_and_batcher_states(tmp_path: Path) -> None:
    # Given: global RNGs and batchers captured in a checkpoint.
    checkpoint = import_module("minigpt.checkpoint")
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    gpt = GPT(GPTConfig(vocab_size=11, block_size=4, n_layer=1, n_head=1, n_embd=8))
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    train_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=13)
    val_batcher = TokenBatcher(np.arange(32), batch_size=2, block_size=4, seed=14)
    checkpoint_path = tmp_path / "latest.pt"
    checkpoint.save_checkpoint(
        checkpoint_path,
        model=gpt,
        optimizer=optimizer,
        step=0,
        config=experiment_config(tmp_path),
        train_batcher=train_batcher,
        val_batcher=val_batcher,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_batch = train_batcher.next_batch()

    # When: all random states are restored.
    checkpoint.load_checkpoint(
        checkpoint_path,
        model=gpt,
        optimizer=optimizer,
        train_batcher=train_batcher,
        val_batcher=val_batcher,
    )

    # Then: subsequent random values and sampled batches are identical.
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    actual_batch = train_batcher.next_batch()
    assert torch.equal(actual_batch[0], expected_batch[0])
    assert torch.equal(actual_batch[1], expected_batch[1])
