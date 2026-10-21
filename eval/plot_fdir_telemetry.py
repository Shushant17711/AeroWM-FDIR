"""Four-panel FDIR telemetry figure: trajectory, surprise, isolation, actuation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.schema import FaultMode, WorldModelConfig
from envs.rocket_vec_env import VectorizedRocketEnv
from envs.rocket_dynamics import tilt_angle_deg


@torch.no_grad()
def record_episode(
    env: VectorizedRocketEnv,
    controller,
    mode: int = 0,
    onset: int = 120,
    is_world_model: bool = False,
    threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Roll out one episode and record telemetry for plotting."""
    env.reset(mode=mode, probability=1.0 if mode else 0.0, onset=onset, severity=0.5)
    controller.reset(env.num_envs, env.device)
    obs = env._observation()
    log = {k: [] for k in ("t", "x", "y", "z", "vz", "tilt", "gimbal_y", "gimbal_z", "throttle", "surprise")}
    probs = []
    for t in range(env.env_cfg.max_episode_steps):
        action = controller.act(obs)
        out = env.step(action)
        obs = out.obs
        s = out.info["state"]
        log["t"].append(t * env.physics.dt)
        log["x"].append(float(s[0, 0]))
        log["y"].append(float(s[0, 1]))
        log["z"].append(float(s[0, 2]))
        log["vz"].append(float(s[0, 5]))
        log["tilt"].append(float(tilt_angle_deg(s)[0]))
        log["gimbal_y"].append(float(s[0, 14]))
        log["gimbal_z"].append(float(s[0, 15]))
        log["throttle"].append(float(s[0, 16] / env.physics.max_thrust))
        if is_world_model and hasattr(controller, "surprise"):
            log["surprise"].append(float(controller.surprise[0]))
            logits = controller.last_debug.get("fault_logits")
            probs.append(torch.softmax(logits[0].float(), dim=-1).cpu().numpy() if logits is not None else np.zeros(5))
        else:
            log["surprise"].append(0.0)
            probs.append(np.zeros(5))
        if (out.terminated | out.truncated).any():
            break
    data = {k: np.asarray(v) for k, v in log.items()}
    data["probs"] = np.asarray(probs)
    data["fault_onset"] = onset * env.physics.dt
    data["mode"] = mode
    data["threshold"] = threshold
    return data


def plot_fdir_telemetry(
    nominal: Dict[str, np.ndarray],
    fault: Dict[str, np.ndarray],
    path: Path | str = "results/fdir_recovery_demo.png",
    title: str = "AeroWM-FDIR fault detection, isolation and recovery",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title, fontsize=14)

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(nominal["x"], nominal["y"], nominal["z"], label="Nominal", color="#4C72B0")
    ax1.plot(fault["x"], fault["y"], fault["z"], label="Fault + FDIR recovery", color="#C44E52")
    ax1.scatter([0], [0], [0], color="k", marker="*", s=80, label="Pad")
    onset_idx = min(int(fault["fault_onset"] / 0.02), len(fault["x"]) - 1)
    ax1.scatter([fault["x"][onset_idx]], [fault["y"][onset_idx]], [fault["z"][onset_idx]],
                color="#C44E52", marker="x", s=60, label="Fault onset")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("z [m]")
    ax1.set_title("Panel 1 — 3D descent trajectory")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(2, 2, 2)
    # The anomaly score spans three decades (nominal ~1, a 40-sigma residual
    # signature ~10^2), so the axis is logarithmic -- otherwise the calibrated
    # alarm threshold is invisible next to the fault transient.
    ax2.set_yscale("log")
    ax2.plot(nominal["t"], np.maximum(nominal["surprise"], 1e-3), color="#4C72B0",
              lw=1.0, label="Nominal $S_t$")
    ax2.plot(fault["t"], np.maximum(fault["surprise"], 1e-3), color="#C44E52",
              label="Fault $S_t$")
    if fault.get("threshold") is not None:
        ax2.axhline(max(fault["threshold"], 1e-3), color="k", ls="--", lw=1,
                    label=r"$\tau_{alarm}$")
    ax2.axvline(fault["fault_onset"], color="gray", ls=":", label="fault onset")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("$S_t$ (log scale)")
    ax2.set_title("Panel 2 — real-time anomaly score (KL + innovation + residuals)")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 2, 3)
    labels = [FaultMode(i).name for i in range(5)]
    for i, label in enumerate(labels):
        ax3.plot(fault["t"], fault["probs"][: len(fault["t"]), i], label=label)
    ax3.axvline(fault["fault_onset"], color="gray", ls=":")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("softmax probability")
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_title("Panel 3 — fault isolation")
    ax3.legend(fontsize=7, ncol=2)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(fault["t"], np.rad2deg(fault["gimbal_y"]), label=r"$\delta_y$", color="#55A868")
    ax4.plot(fault["t"], np.rad2deg(fault["gimbal_z"]), label=r"$\delta_z$", color="#8172B2")
    ax4.plot(fault["t"], fault["throttle"], label="throttle", color="#937860", alpha=0.7)
    ax4.axvline(fault["fault_onset"], color="gray", ls=":")
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("deg / throttle fraction")
    ax4.set_title("Panel 4 — TVC and throttle compensation")
    ax4.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the FDIR telemetry figure")
    parser.add_argument("--output", default="results/fdir_recovery_demo.png")
    parser.add_argument("--mode", type=int, default=int(FaultMode.GIMBAL_BIAS))
    parser.add_argument("--onset", type=int, default=120)
    parser.add_argument("--checkpoint", default="runs/aerowm_fdir/checkpoint_final.pt")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    from agents.world_model_agent import WorldModelAgent
    from models.world_model import RocketWorldModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=1, device=device, seed=0)
    cfg = WorldModelConfig()
    model = RocketWorldModel(cfg).to(device).eval()
    agent = WorldModelAgent(cfg, model).to(device).eval()
    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        agent.actor.load_state_dict(state["actor"])
    from agents.pid_baseline import CascadePIDController
    from agents.residual_control import FDIRResidualController

    controller = FDIRResidualController(
        model,
        agent,
        device=device,
        base=CascadePIDController(device=device, fault_estimation_mode=True),
        mode="actor",
    )

    nominal = record_episode(env, controller, mode=0, is_world_model=True)
    # Calibrate the alarm threshold on the *measured* nominal surprise of this
    # flight (skipping the first 20 samples, where the recurrent state is
    # still warming up), exactly like the benchmark does.
    nominal_surprise = torch.tensor(
        nominal["surprise"][20:], dtype=torch.float32, device=device
    )
    if nominal_surprise.numel() > 0:
        model.fdir.detector.calibrate(nominal_surprise)
    threshold = float(model.fdir.detector.threshold)
    fault = record_episode(env, controller, mode=args.mode, onset=args.onset, is_world_model=True, threshold=threshold)
    path = plot_fdir_telemetry(nominal, fault, args.output)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
