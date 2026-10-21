"""Task 3.2: pinned sequence replay buffer correctness and memory budget."""

from __future__ import annotations

import pytest
import torch

from train.replay_buffer import PinnedSequenceReplayBuffer


def make_buffer(capacity: int = 200, seq: int = 6, device="cpu") -> PinnedSequenceReplayBuffer:
    buf = PinnedSequenceReplayBuffer(
        capacity=capacity, sequence_length=seq, device=device, seed=0, pin_memory=False, num_envs=2
    )
    buf.reset_episode_tracking(2)
    return buf


def fill(buf: PinnedSequenceReplayBuffer, steps: int, episode_len: int = 9) -> None:
    for t in range(steps):
        done = torch.zeros(2, dtype=torch.bool)
        if t % episode_len == episode_len - 1:
            done[:] = True
        buf.add(
            torch.randn(2, 18),
            torch.randn(2, 3),
            torch.randn(2),
            done,
            torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), 5).float(),
        )


def test_shapes_and_types():
    buf = make_buffer()
    fill(buf, 40)
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, 6, 18)
    assert batch["action"].shape == (8, 6, 3)
    assert batch["reward"].shape == (8, 6)
    assert batch["done"].shape == (8, 6)
    assert batch["fault_context"].shape == (8, 6, 5)


def test_chunks_never_cross_episode_boundaries():
    buf = make_buffer(capacity=500, seq=6)
    fill(buf, 300, episode_len=9)
    offsets = torch.arange(buf.sequence_length)
    for _ in range(20):
        batch = buf.sample(32)
        ep = batch["episode_id"]
        step = batch["step_in_episode"]
        assert torch.all(ep == ep[:, :1])
        assert torch.all(step == step[:, :1] + offsets.unsqueeze(0))


def test_terminal_transition_is_last_in_chunk():
    buf = make_buffer(capacity=500, seq=6)
    fill(buf, 200, episode_len=7)
    batch = buf.sample(64)
    done = batch["done"]
    # If a chunk contains a terminal step it must be its final element.
    for row in range(done.shape[0]):
        idx = torch.nonzero(done[row]).flatten()
        if idx.numel():
            assert int(idx[0]) == done.shape[1] - 1


def test_ring_buffer_wraparound():
    buf = make_buffer(capacity=50, seq=5)
    fill(buf, 500, episode_len=11)
    assert len(buf) == 50
    batch = buf.sample(16)
    assert batch["obs"].shape == (16, 5, 18)
    assert torch.isfinite(batch["obs"]).all()
    assert torch.isfinite(batch["reward"]).all()


def test_requires_enough_transitions():
    buf = make_buffer(capacity=50, seq=8)
    fill(buf, 3)
    with pytest.raises(RuntimeError):
        buf.sample(2)


def test_sequence_length_validation():
    with pytest.raises(ValueError):
        PinnedSequenceReplayBuffer(sequence_length=1)
    with pytest.raises(ValueError):
        PinnedSequenceReplayBuffer(capacity=1)
    with pytest.raises(ValueError):
        PinnedSequenceReplayBuffer(capacity=20, sequence_length=8, num_envs=4)


def test_state_dict_roundtrip():
    buf = make_buffer(capacity=100, seq=4)
    fill(buf, 80, episode_len=10)
    state = buf.state_dict()
    other = make_buffer(capacity=100, seq=4)
    other.load_state_dict(state)
    assert len(other) == len(buf)
    assert torch.equal(other.episode_id, buf.episode_id)


@pytest.mark.gpu
def test_sampling_stays_within_vram_budget(gpu_device):
    device = gpu_device
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    buf = PinnedSequenceReplayBuffer(
        capacity=10_000, sequence_length=16, device=device, pin_memory=True, num_envs=64
    )
    buf.reset_episode_tracking(64)
    for t in range(50):
        buf.add(torch.randn(64, 18), torch.randn(64, 3), torch.randn(64), torch.zeros(64, dtype=torch.bool))
    for _ in range(5):
        batch = buf.sample(256)
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert batch["obs"].device.type == "cuda"
    assert (peak - base) < 500 * 1024**2
