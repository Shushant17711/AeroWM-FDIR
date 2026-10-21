"""Task 3.1: cascade PID baseline achieves nominal landings and degrades on faults."""

from __future__ import annotations

import pytest
import torch

from agents.base import BaseController
from agents.pid_baseline import CascadePIDController
from configs.schema import FaultMode
from envs.rocket_vec_env import VectorizedRocketEnv


@torch.no_grad()
def run_episodes(env, controller, episodes: int, mode: int = 0, onset: int = 60) -> dict:
    """Per-mode landing rate over ``episodes`` completed episodes.

    ``fixed_fault_mode`` pins the *auto-resets* too.  Without it ``step()``
    re-draws the fault at the default 65 % probability on every auto-reset, so
    a "nominal" measurement is silently a 65 %-fault mixture (and a "stuck"
    measurement is diluted).  The benchmark uses the same mechanism.
    """
    previous = env.fixed_fault_mode
    env.fixed_fault_mode = mode
    success = crash = timeout = 0
    while success + crash + timeout < episodes:
        env.reset(mode=mode, probability=1.0 if mode else 0.0, onset=onset, severity=0.35)
        controller.reset(env.num_envs, env.device)
        obs = env._observation()
        for _ in range(env.env_cfg.max_episode_steps):
            out = env.step(controller.act(obs))
            obs = out.obs
            done = out.terminated | out.truncated
            if done.any():
                landed = out.info["landing_success"] & out.terminated
                success += int(landed.sum())
                crash += int((out.terminated & ~out.info["landing_success"]).sum())
                timeout += int((out.truncated & ~out.terminated).sum())
            if success + crash + timeout >= episodes:
                break
    env.fixed_fault_mode = previous
    total = max(1, success + crash + timeout)
    return {"success": success, "crash": crash, "timeout": timeout, "rate": success / total}


@pytest.fixture(scope="module")
def env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=16, device=device, seed=11)
    yield env


def test_controller_contract(env):
    controller = CascadePIDController(device=env.device)
    assert isinstance(controller, BaseController)
    assert controller.action_dim == 3
    controller.reset(env.num_envs, env.device)
    obs, _ = env.reset(probability=0.0)
    action = controller.act(obs)
    assert action.shape == (env.num_envs, 3)
    assert torch.isfinite(action).all()
    assert torch.all(action.abs() <= 1.0 + 1e-6)


def test_controller_is_deterministic_from_a_given_state(env):
    """The PID carries slew/integrator state, so determinism is per-episode."""
    controller = CascadePIDController(device=env.device)
    obs, _ = env.reset(probability=0.0)
    controller.reset(env.num_envs, env.device)
    a1 = controller.act(obs)
    controller.reset(env.num_envs, env.device)
    a2 = controller.act(obs)
    assert torch.allclose(a1, a2)


@pytest.mark.slow
def test_nominal_landing_rate_above_90_percent(env):
    controller = CascadePIDController(device=env.device)
    result = run_episodes(env, controller, episodes=48, mode=0)
    print(f"\nPID nominal: {result}")
    assert result["rate"] >= 0.90, result


@pytest.mark.slow
def test_stuck_gimbal_defeats_pid(env):
    controller = CascadePIDController(device=env.device)
    result = run_episodes(env, controller, episodes=16, mode=int(FaultMode.GIMBAL_STUCK))
    print(f"\nPID stuck gimbal: {result}")
    assert result["rate"] <= 0.30, result


@pytest.mark.slow
def test_gimbal_bias_defeats_pid(env):
    controller = CascadePIDController(device=env.device)
    result = run_episodes(env, controller, episodes=16, mode=int(FaultMode.GIMBAL_BIAS))
    print(f"\nPID gimbal bias: {result}")
    assert result["rate"] <= 0.50, result


@pytest.mark.slow
def test_com_shift_defeats_pid(env):
    controller = CascadePIDController(device=env.device)
    result = run_episodes(env, controller, episodes=16, mode=int(FaultMode.COM_SHIFT))
    print(f"\nPID CoM shift: {result}")
    assert result["rate"] <= 0.50, result


@pytest.mark.slow
def test_loe_degrades_but_remains_controllable(env):
    """Loss of engine effectiveness is partially absorbed by the velocity
    feedback of the non-adaptive baseline; it must still be no better than
    nominal flight."""
    controller = CascadePIDController(device=env.device)
    nominal = run_episodes(env, controller, episodes=48, mode=0)
    loe = run_episodes(env, controller, episodes=16, mode=int(FaultMode.LOE_THRUST))
    print(f"\nPID LOE: {loe} (nominal {nominal})")
    assert loe["rate"] < nominal["rate"]
