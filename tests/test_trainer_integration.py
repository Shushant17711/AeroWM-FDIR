"""Task 6.1: trainer integration, VRAM budget and loss decrease."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from configs.schema import TrainingConfig, WorldModelConfig
from train.trainer import WorldModelTrainer


@pytest.fixture()
def trainer(tmp_path) -> WorldModelTrainer:
    wm = WorldModelConfig(
        latent_dim=64, gru_dim=64, hidden_dim=64, simnorm_groups=8, simnorm_v=8,
        batch_size=8, horizon=6,
    )
    training = TrainingConfig(
        num_envs=16,
        sequence_length=4,
        replay_capacity=4_000,
        rollout_env_steps=256,
        log_interval=64,
        checkpoint_interval=100_000,
        output_dir=str(tmp_path),
    )
    return WorldModelTrainer(
        training=training,
        world_model=wm,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=str(tmp_path),
    )


def test_rollout_and_replay(trainer):
    stats = trainer.collect(6)
    assert "rollout_reward" in stats
    assert len(trainer.buffer) == 6 * trainer.env.num_envs
    batch = trainer.sample_batch(4)
    assert batch["obs"].shape == (4, 4, 18)
    assert batch["fault_context"].shape == (4, 4, 5)


def test_ten_gradient_updates_reduce_loss_on_fixed_batch(trainer):
    trainer.collect(8)
    batch = trainer.sample_batch(8)
    losses = []
    for _ in range(10):
        terms = trainer.loss_terms(batch)
        losses.append(float(terms["loss"].detach()))
        trainer.optimize(batch)
    print("\nlosses:", [round(x, 2) for x in losses])
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]


def test_peak_vram_below_budget(trainer):
    trainer.collect(8)
    batch = trainer.sample_batch(8)
    for _ in range(5):
        trainer.optimize(batch)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\npeak VRAM: {peak:.2f} GB")
        assert peak < 4.5


def test_checkpoint_roundtrip(trainer, tmp_path):
    trainer.collect(8)
    path = Path(tmp_path) / "checkpoint.pt"
    trainer.save(path)
    assert path.exists()
    trainer.load(path)
    assert trainer._total_steps == 8 * trainer.env.num_envs


def test_metrics_are_logged(trainer):
    trainer.collect(8)
    trainer.update(8)
    metrics = {"loss": 1.0, "env_steps": trainer._total_steps}
    trainer._log(metrics)
    with (Path(trainer.output_dir) / "metrics.jsonl").open() as fh:
        line = json.loads(fh.readline())
    assert line["env_steps"] == trainer._total_steps
