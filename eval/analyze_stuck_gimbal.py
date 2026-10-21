"""Why GIMBAL_STUCK is not recoverable with the frozen 3-D action contract.

This is an *analysis* tool, not part of the benchmark. It quantifies the
uncontrolled-axis torque and measures how quickly the attitude diverges even
with a fault-aware oracle that knows the stuck axis and angle exactly.
"""

from __future__ import annotations

import argparse

import torch

from agents.pid_baseline import CascadePIDController
from configs.schema import RocketPhysicsConfig
from envs.rocket_vec_env import VectorizedRocketEnv
from envs.rocket_dynamics import tilt_angle_deg


def theoretical_analysis(cfg: RocketPhysicsConfig, mass: float, thrust: float, stuck_angle: float) -> dict:
    inertia = cfg.specific_inertia[0] * mass
    torque = cfg.engine_moment_arm * thrust * abs(stuck_angle)
    alpha = torque / inertia
    damping = cfg.angular_damping[0]
    omega_ss = torque / damping
    return {
        "stuck_angle_deg": stuck_angle * 180.0 / 3.141592653589793,
        "torque_Nm": torque,
        "angular_accel_rad_s2": alpha,
        "damping_limited_rate_rad_s": omega_ss,
    }


@torch.no_grad()
def measure_divergence(env, controller, episodes: int = 32) -> dict:
    env.fixed_fault_mode = 3
    env.balanced_fault_mix = False
    controller.oracle_fault_compensation = True
    successes = 0
    total = 0
    while total < episodes:
        env.reset(mode=3, probability=1.0)
        controller.reset(env.num_envs, env.device)
        obs = env._observation()
        onset_angles = torch.zeros(env.num_envs, device=env.device)
        onset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        t20 = torch.full((env.num_envs,), -1.0, device=env.device)
        t60 = torch.full((env.num_envs,), -1.0, device=env.device)
        onset_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        for t in range(env.env_cfg.max_episode_steps):
            out = env.step(controller.act(obs, fault_state=env.fault))
            obs = out.obs
            active = out.info["fault_active"]
            new = active & ~onset_seen
            if new.any():
                onset_seen |= new
                onset_angles[new] = env.state[new, 14:16].abs().max(dim=1).values
                onset_step[new] = t
            tilt = tilt_angle_deg(env.state)
            elapsed = (t - onset_step).float() * env.physics.dt
            hit20 = (t20 < 0) & onset_seen & (tilt > 20.0)
            t20[hit20] = elapsed[hit20]
            hit60 = (t60 < 0) & onset_seen & (tilt > 60.0)
            t60[hit60] = elapsed[hit60]
            done = out.terminated | out.truncated
            if done.any():
                successes += int((out.info["landing_success"] & out.terminated).sum())
                total += int(done.sum())
            if total >= episodes:
                break
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / max(1, total),
        "median_onset_angle_deg": float(onset_angles[onset_seen].median() * 180 / 3.141592653589793)
        if onset_seen.any()
        else 0.0,
        "median_time_to_20deg_s": float(t20[t20 >= 0].median()) if (t20 >= 0).any() else None,
        "median_time_to_60deg_s": float(t60[t60 >= 0].median()) if (t60 >= 0).any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = RocketPhysicsConfig()
    print("Theoretical stuck-gimbal disturbance (nominal 300 kN thrust, 30 t):")
    for angle_deg in (0.5, 1.0, 2.0, 4.0):
        info = theoretical_analysis(cfg, 30_000.0, 300_000.0, angle_deg * 3.141592653589793 / 180.0)
        print(
            f"  stuck {angle_deg:4.1f} deg -> torque {info['torque_Nm']:9.0f} Nm, "
            f"alpha {info['angular_accel_rad_s2']:.4f} rad/s^2, "
            f"damping-limited rate {info['damping_limited_rate_rad_s']:.3f} rad/s"
        )
    env = VectorizedRocketEnv(num_envs=args.batch, device=device, seed=0)
    controller = CascadePIDController(device=device, oracle_fault_compensation=True)
    result = measure_divergence(env, controller, args.episodes)
    print("Oracle (ground-truth fault knowledge) stuck-gimbal recovery:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
