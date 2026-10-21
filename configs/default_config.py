"""Concrete default config instances and filesystem paths.

Import from here when you need *a* configuration; import from
``configs.schema`` when you need the contract itself.
"""

from __future__ import annotations

from pathlib import Path

from configs.schema import (
    ActionSchema,
    EnvironmentConfig,
    FaultInjectionSchema,
    MPPIConfig,
    ObservationSchema,
    PPOConfig,
    RewardConfig,
    RocketPhysicsConfig,
    TrainingConfig,
    WorldModelConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = PROJECT_ROOT / "runs"

PHYSICS = RocketPhysicsConfig()
OBSERVATION = ObservationSchema()
ACTION = ActionSchema()
FAULT = FaultInjectionSchema()
ENVIRONMENT = EnvironmentConfig()
REWARD = RewardConfig()
WORLD_MODEL = WorldModelConfig()
MPPI = MPPIConfig()
PPO = PPOConfig()
TRAINING = TrainingConfig()

ALL = {
    "physics": PHYSICS,
    "observation": OBSERVATION,
    "action": ACTION,
    "fault": FAULT,
    "environment": ENVIRONMENT,
    "reward": REWARD,
    "world_model": WORLD_MODEL,
    "mppi": MPPI,
    "ppo": PPO,
    "training": TRAINING,
}
