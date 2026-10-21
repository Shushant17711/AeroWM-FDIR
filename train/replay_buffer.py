"""Episode-aware pinned-RAM replay buffer.

Transitions live in page-locked host memory so batches can be copied to the
GPU with ``non_blocking=True``; GPU VRAM usage is limited to the batches that
are actually being trained on.  Sampling returns contiguous chunks of length
``T`` that never cross an episode boundary.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from configs.schema import ActionSchema, ObservationSchema


class PinnedSequenceReplayBuffer:
    """Ring buffer of transitions with sequence sampling."""

    def __init__(
        self,
        capacity: int = 1_000_000,
        observation: Optional[ObservationSchema] = None,
        action: Optional[ActionSchema] = None,
        fault_context_dim: int = 5,
        sequence_length: int = 16,
        device: torch.device | str = "cuda",
        seed: int = 0,
        pin_memory: bool = True,
        num_envs: int = 1,
    ) -> None:
        if capacity <= 1:
            raise ValueError("capacity must be > 1")
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        # Keep the ring aligned to the environment stride so that a slot and
        # the same environment's next slot are always ``num_envs`` apart.
        self.capacity = int(capacity) - int(capacity) % self.num_envs
        if self.capacity < self.num_envs * sequence_length:
            raise ValueError("capacity too small for the requested sequence length")
        self.obs_dim = (observation or ObservationSchema()).dim
        self.action_dim = (action or ActionSchema()).dim
        self.fault_context_dim = int(fault_context_dim)
        self.sequence_length = int(sequence_length)
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()

        def alloc(*shape, dtype=torch.float32):
            t = torch.empty(*shape, dtype=dtype)
            return t.pin_memory() if self.pin_memory else t

        self.obs = alloc(self.capacity, self.obs_dim)
        self.action = alloc(self.capacity, self.action_dim)
        self.reward = alloc(self.capacity)
        self.done = alloc(self.capacity, dtype=torch.bool)
        self.fault_context = alloc(self.capacity, self.fault_context_dim)
        self.expert = alloc(self.capacity, dtype=torch.bool)
        self.valid = alloc(self.capacity, dtype=torch.bool)
        self.episode_id = alloc(self.capacity, dtype=torch.long)
        self.step_in_episode = alloc(self.capacity, dtype=torch.long)

        self._ptr = 0
        self._size = 0
        self._env_episode = torch.zeros(1, dtype=torch.long)
        self._env_step = torch.zeros(1, dtype=torch.long)
        self._episode_counter = 0
        # Number of contiguous environment steps whose transitions are all
        # demonstrations (used for expert-oversampled BC batches).
        self.expert_steps = 0
        self.generator = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size >= self.capacity

    def reset_episode_tracking(self, num_envs: int = 1, device: Optional[torch.device] = None) -> None:
        """Reset per-environment episode bookkeeping (call on env reset).

        Episode ids come from a monotonic global counter, so ids stay unique
        across resets and buffer wrap-around.
        """
        # Bookkeeping stays in host memory alongside the ring buffer.
        self._env_episode = torch.arange(
            self._episode_counter, self._episode_counter + num_envs, dtype=torch.long
        )
        self._episode_counter += num_envs
        self._env_step = torch.zeros(num_envs, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Insertion
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        fault_context: Optional[torch.Tensor] = None,
        expert: Optional[torch.Tensor] = None,
    ) -> None:
        """Append one transition per environment (``[B, ...]``).

        ``expert`` marks demonstration transitions (used by the behavioural
        cloning term); it defaults to ``False``.
        """
        batch = obs.shape[0]
        if action.shape[0] != batch or reward.shape[0] != batch:
            raise ValueError("batch dimensions must agree")
        if batch != self.num_envs:
            raise ValueError(f"expected {self.num_envs} environments per add, got {batch}")
        if self._env_episode.numel() != batch:
            self.reset_episode_tracking(batch, obs.device)

        obs_c = obs.detach().to("cpu", torch.float32)
        action_c = action.detach().to("cpu", torch.float32)
        reward_c = reward.detach().to("cpu", torch.float32)
        done_c = done.detach().to("cpu", torch.bool)
        fault_c = (
            fault_context.detach().to("cpu", torch.float32)
            if fault_context is not None
            else torch.zeros(batch, self.fault_context_dim)
        )

        expert_c = (
            expert.detach().to("cpu", torch.bool)
            if expert is not None
            else torch.zeros(batch, dtype=torch.bool)
        )

        idx = (self._ptr + torch.arange(batch)) % self.capacity
        self.obs[idx] = obs_c
        self.action[idx] = action_c
        self.reward[idx] = reward_c
        self.done[idx] = done_c
        self.fault_context[idx] = fault_c
        self.expert[idx] = expert_c
        if bool(expert_c.all()) and int(idx.max()) + 1 > self.expert_steps * self.num_envs:
            self.expert_steps = (int(idx.max()) + 1) // self.num_envs
        self.valid[idx] = True
        self.episode_id[idx] = self._env_episode
        self.step_in_episode[idx] = self._env_step

        # Advance per-environment episode bookkeeping.
        self._env_step = torch.where(done_c, torch.zeros_like(self._env_step), self._env_step + 1)
        n_new = int(done_c.sum())
        if n_new:
            fresh = torch.arange(self._episode_counter, self._episode_counter + n_new)
            self._env_episode[done_c] = fresh
            self._episode_counter += n_new
        self._ptr = (self._ptr + batch) % self.capacity
        self._size = min(self._size + batch, self.capacity)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, batch_size: int, expert_fraction: float = 0.0) -> Dict[str, torch.Tensor]:
        """Sample ``[B, T]`` chunks without crossing episode boundaries."""
        T = self.sequence_length
        if self._size < T:
            raise RuntimeError(f"buffer holds {self._size} transitions, need at least {T}")

        steps_available = self._size // self.num_envs
        if steps_available < T:
            raise RuntimeError("not enough full environment steps to sample a sequence")
        high = steps_available - T + 1
        # Pick a random environment and a random starting step for each sample.
        env_idx = torch.randint(0, self.num_envs, (batch_size,), generator=self.generator)
        step_start = torch.randint(0, high, (batch_size,), generator=self.generator)
        n_expert = int(round(batch_size * max(0.0, min(1.0, expert_fraction))))
        if n_expert > 0 and self.expert_steps >= T:
            expert_high = self.expert_steps - T + 1
            step_start[:n_expert] = torch.randint(0, expert_high, (n_expert,), generator=self.generator)
        offsets = torch.arange(T)
        idx = (
            step_start.unsqueeze(1) * self.num_envs
            + env_idx.unsqueeze(1)
            + offsets.unsqueeze(0) * self.num_envs
        ) % self.capacity

        for _ in range(16):
            ep = self.episode_id[idx]
            step = self.step_in_episode[idx]
            ok = (
                self.valid[idx]
                & (ep == ep[:, :1])
                & (step == (step[:, :1] + offsets.unsqueeze(0)))
            ).all(dim=1)
            if n_expert > 0:
                ok[:n_expert] = ok[:n_expert] & self.expert[idx[:n_expert]].all(dim=1)
            if bool(ok.all()):
                break
            bad = ~ok
            n_bad = int(bad.sum())
            step_start[bad] = torch.randint(0, high, (n_bad,), generator=self.generator)
            env_idx[bad] = torch.randint(0, self.num_envs, (n_bad,), generator=self.generator)
            idx = (
                step_start.unsqueeze(1) * self.num_envs
                + env_idx.unsqueeze(1)
                + offsets.unsqueeze(0) * self.num_envs
            ) % self.capacity

        def gather(t: torch.Tensor) -> torch.Tensor:
            return t[idx].to(self.device, non_blocking=True)

        return {
            "obs": gather(self.obs),
            "action": gather(self.action),
            "reward": gather(self.reward),
            "done": gather(self.done),
            "fault_context": gather(self.fault_context),
            "expert": gather(self.expert),
            "episode_id": gather(self.episode_id),
            "step_in_episode": gather(self.step_in_episode),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        return {
            "obs": self.obs.clone(),
            "action": self.action.clone(),
            "reward": self.reward.clone(),
            "done": self.done.clone(),
            "fault_context": self.fault_context.clone(),
            "expert": self.expert.clone(),
            "valid": self.valid.clone(),
            "episode_id": self.episode_id.clone(),
            "step_in_episode": self.step_in_episode.clone(),
            "ptr": self._ptr,
            "size": self._size,
            "episode_counter": self._episode_counter,
            "env_episode": self._env_episode.clone(),
            "env_step": self._env_step.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        n = int(state["size"])
        if n > self.capacity:
            raise ValueError("checkpoint larger than buffer capacity")
        self.obs.copy_(state["obs"])
        self.action.copy_(state["action"])
        self.reward.copy_(state["reward"])
        self.done.copy_(state["done"])
        self.fault_context.copy_(state["fault_context"])
        if "expert" in state:
            self.expert.copy_(state["expert"])
        else:
            self.expert.zero_()
        self.valid.copy_(state["valid"])
        self.episode_id.copy_(state["episode_id"])
        self.step_in_episode.copy_(state["step_in_episode"])
        self._ptr = int(state["ptr"])
        self._size = n
        self._episode_counter = int(state["episode_counter"])
        self._env_episode = state["env_episode"].clone()
        self._env_step = state["env_step"].clone()
