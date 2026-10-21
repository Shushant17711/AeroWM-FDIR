"""Vectorized environment, dynamics, fault injection and quaternion math."""

from envs.base_types import (  # noqa: F401
    ControllerContext,
    EnvResetReturn,
    EnvStepReturn,
    FaultState,
    RocketState,
    STATE_DIM,
)
from envs.fault_injector import FaultInjector  # noqa: F401
from envs.quaternion_ops import (  # noqa: F401
    quat_conjugate,
    quat_derivative,
    quat_error,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_rotate_inverse,
    quat_slerp,
    quat_tilt_cosine,
    quat_to_rotmat,
    rotation_vector,
)
from envs.rk4_integrator import post_step_clamp, rk4_step, rk4_substeps  # noqa: F401
from envs.rocket_dynamics import (  # noqa: F401
    actuator_commands,
    atmosphere_density,
    compute_derivatives,
    dynamic_pressure,
    gimbal_from_torque,
    inertia_tensor,
    thrust_vector_body,
)
from envs.rocket_vec_env import VectorizedRocketEnv  # noqa: F401

__all__ = [
    "RocketState",
    "FaultState",
    "EnvStepReturn",
    "EnvResetReturn",
    "ControllerContext",
    "STATE_DIM",
    "FaultInjector",
    "VectorizedRocketEnv",
    "quat_normalize",
    "quat_multiply",
    "quat_conjugate",
    "quat_to_rotmat",
    "quat_derivative",
    "quat_rotate",
    "quat_rotate_inverse",
    "quat_from_axis_angle",
    "quat_tilt_cosine",
    "rotation_vector",
    "quat_error",
    "quat_slerp",
    "compute_derivatives",
    "actuator_commands",
    "thrust_vector_body",
    "atmosphere_density",
    "dynamic_pressure",
    "inertia_tensor",
    "gimbal_from_torque",
    "post_step_clamp",
    "rk4_step",
    "rk4_substeps",
]
