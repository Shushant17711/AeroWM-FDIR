"""Task 1.2: controller/environment interfaces and tensor containers."""

from __future__ import annotations

import torch

from agents.base import BaseController, ControllerMixin
from configs.schema import ActionSchema, FaultParamsLayout
from envs.base_types import EnvStepReturn, FaultState, RocketState, STATE_DIM
from envs.rocket_vec_env import VectorizedRocketEnv


class DummyController(ControllerMixin):
    action_schema = ActionSchema()

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self, batch_size: int = 1, device=None) -> None:
        self.batch_size = batch_size

    @torch.no_grad()
    def act(self, obs: torch.Tensor, fault_context=None) -> torch.Tensor:
        obs = self._check_obs(obs)
        return self._clip_action(torch.zeros(obs.shape[0], 3, device=obs.device))


def test_dummy_controller_satisfies_protocol():
    controller = DummyController()
    assert isinstance(controller, BaseController)
    controller.reset(4)
    out = controller.act(torch.zeros(4, 18))
    assert out.shape == (4, 3)
    assert torch.all(out.abs() <= 1.0)


def test_controller_mixin_rejects_bad_shapes():
    controller = DummyController()
    try:
        controller.act(torch.zeros(4, 17))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for wrong observation shape")
    clipped = controller._clip_action(torch.full((2, 3), 5.0))
    assert torch.all(clipped == 1.0)


def test_rocket_state_roundtrip():
    batch = 5
    state = RocketState(
        r=torch.randn(batch, 3),
        v=torch.randn(batch, 3),
        q=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(batch, 4).clone(),
        omega=torch.randn(batch, 3),
        mass=torch.full((batch,), 30_000.0),
        gimbal=torch.zeros(batch, 2),
        thrust=torch.full((batch,), 1e5),
    )
    packed = state.to_tensor()
    assert packed.shape == (batch, STATE_DIM)
    recovered = RocketState.from_tensor(packed)
    for field in ("r", "v", "q", "omega", "mass", "gimbal", "thrust"):
        assert torch.allclose(getattr(state, field), getattr(recovered, field))
    assert state.batch_size == batch


def test_fault_state_one_hot():
    fault = FaultState(
        mode=torch.tensor([0, 1, 4]),
        onset=torch.tensor([10, 20, 30]),
        severity=torch.tensor([0.0, 0.3, 0.5]),
        active=torch.tensor([False, True, True]),
        params=torch.zeros(3, FaultParamsLayout.DIM),
    )
    ctx = fault.one_hot()
    assert ctx.shape == (3, 5)
    assert torch.allclose(ctx.sum(dim=-1), torch.ones(3))
    assert ctx[1, 1].item() == 1.0
    assert ctx[2, 4].item() == 1.0


def test_env_step_return_contract():
    env = VectorizedRocketEnv(num_envs=3, device="cpu", seed=1)
    out = env.step(torch.zeros(3, 3))
    assert isinstance(out, EnvStepReturn)
    assert out.obs.shape == (3, 18)
    assert out.reward.shape == (3,)
    assert out.terminated.dtype == torch.bool
    assert out.truncated.dtype == torch.bool
    for key in ("fault_context", "fault_mode", "fault_active", "fault_onset", "state"):
        assert key in out.info, key
    assert out.info["fault_context"].shape == (3, 5)
