"""Task 6.2: benchmark schema completeness and telemetry figure generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from configs.schema import MPPIConfig, WorldModelConfig
from eval.benchmark_fdir import run_benchmark
from eval.plot_fdir_telemetry import plot_fdir_telemetry, record_episode
from envs.rocket_vec_env import VectorizedRocketEnv

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def micro_result(tmp_path_factory):
    output = tmp_path_factory.mktemp("bench") / "eval_results.json"
    result = run_benchmark(
        episodes_per_mode=1,
        num_envs=2,
        seed=0,
        onset=80,
        mppi=MPPIConfig(num_samples=32, num_policy_samples=4, horizon=4, iterations=1, num_elites=4),
        world_model_cfg=WorldModelConfig(
            latent_dim=64, gru_dim=64, hidden_dim=64, simnorm_groups=8, simnorm_v=8
        ),
        use_mppi=False,
        output=str(output),
        verbose=False,
    )
    result["_path"] = str(output)
    return result


def test_result_schema(micro_result):
    payload = micro_result
    for key in ("generated_at", "device", "seed", "episodes_per_mode", "dt_ms", "controllers", "results", "summary"):
        assert key in payload
    assert set(payload["controllers"]) == {"Cascaded-PID", "PPO (model-free)", "AeroWM-FDIR (actor)"}
    for name in payload["controllers"]:
        modes = payload["results"][name]
        assert set(modes) == {"NOMINAL", "LOE_THRUST", "GIMBAL_BIAS", "GIMBAL_STUCK", "COM_SHIFT"}
        for mode, metrics in modes.items():
            for field in (
                "episodes",
                "success_rate",
                "crashes",
                "timeouts",
                "mean_final_vz",
                "mean_final_vlat",
                "mean_final_lateral",
                "mean_final_tilt",
                "mean_propellant_margin",
                "detection_latency_steps",
                "detection_latency_ms",
                "detection_rate",
                "fault_classification_accuracy",
            ):
                assert field in metrics, (name, mode, field)
            assert metrics["episodes"] == 1
            assert 0.0 <= metrics["success_rate"] <= 1.0


def test_world_model_reports_fdir_metrics(micro_result):
    wm = micro_result["results"]["AeroWM-FDIR (actor)"]
    assert wm["GIMBAL_BIAS"]["detection_rate"] is not None
    assert wm["NOMINAL"]["detection_latency_steps"] is None


def test_results_file_written(micro_result):
    path = Path(micro_result["_path"])
    assert path.exists()
    payload = json.loads(path.read_text())
    assert "summary" in payload


def test_plot_telemetry(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = VectorizedRocketEnv(num_envs=1, device=device, seed=0)
    from agents.pid_baseline import CascadePIDController

    controller = CascadePIDController(device=device)
    nominal = record_episode(env, controller, mode=0)
    fault = record_episode(env, controller, mode=2, onset=60)
    out = plot_fdir_telemetry(nominal, fault, tmp_path / "demo.png")
    assert out.exists()
    assert out.stat().st_size > 10_000
