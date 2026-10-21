# Implementation Plan — AeroWM-FDIR (6-DOF Rocket World Model & Fault-Tolerant Control)

This plan sequences the build of **AeroWM-FDIR** from frozen data contracts and a GPU-vectorized 6-DOF Runge-Kutta physics engine up through a Recurrent Latent World Model, a KL-divergence Fault Detection, Isolation, and Recovery (FDIR) module, and a Latent MPPI trajectory planner optimized for a single RTX 4060 (8GB VRAM) laptop. "Done" means all unit and integration tests pass via `pytest`, vectorized simulation throughput exceeds 200,000 steps per second (SPS), peak training VRAM stays below 4.5 GB using `bfloat16`, fault detection latency is at most 3 simulation steps (60 ms), and the automated benchmark suite (`eval/benchmark_fdir.py`) verifies at least an 85% fault-recovery landing rate across all injected failure modes.

## Out-of-band prerequisites
- NVIDIA proprietary driver (`>=550.x`) and CUDA 12.x toolkit installed on host Arch Linux system so `torch.cuda.is_available()` returns `True` on the RTX 4060.
- Optional Weights & Biases API key exported in shell environment (`WANDB_API_KEY`) if cloud experiment syncing is desired (offline mode works without credentials).

## Task list

- [x] 1. Freeze shared configuration schemas, tensor contracts, and project scaffolding
- [x] 1.1 Create project package layout and frozen configuration dataclasses
  - Create `pyproject.toml`, directory packages (`configs/`, `envs/`, `models/`, `agents/`, `train/`, `eval/`, `tests/`), and `configs/schema.py`.
  - Implement frozen `@dataclass(frozen=True)` contracts in `configs/schema.py`: `RocketPhysicsConfig` (Falcon-9-class scaled parameters, $\Delta t = 0.02\text{ s}$, throttle bounds $[0.40, 1.00]$, gimbal limits $\pm 8^\circ$), `ObservationSchema` ($o_t \in \mathbb{R}^{18}$), `ActionSchema` ($a_t \in [-1, 1]^3$), `FaultInjectionSchema` ($c_t \in \mathbb{R}^5$ with `FaultMode` IntEnum), and `WorldModelConfig` (`latent_dim=256`, `gru_dim=256`, `horizon=15`).
  - Write `tests/test_schemas.py` verifying immutability, tensor shape constants, and physical parameter bounds.
  - _Design: §1, §2 AC-2.1, AC-2.2, AC-2.3, AC-2.4_

- [x] 1.2 Define abstract controller and environment interfaces
  - Create `agents/base.py` defining the `BaseController` protocol (`reset(batch_size)`, `act(obs, fault_context=None) -> Tensor[B, 3]`) shared by PID, PPO, and World-Model MPPI controllers.
  - Create `envs/base_types.py` defining named tuples `RocketState` (internal 16-D physics state + fault state) and `EnvStepReturn` (`obs`, `reward`, `terminated`, `truncated`, `info` dict with ground-truth `fault_context`).
  - Add contract validation helpers in `tests/test_interfaces.py` ensuring dummy tensors conform to `BaseController` and `EnvStepReturn`.
  - _Design: §2 AC-2.2, AC-2.3, §5 AC-5.3_

- [x] 2. Implement GPU-vectorized 6-DOF rocket physics engine and fault injection
- [x] 2.1 Implement batched quaternion math and 6-DOF rigid-body equations of motion
  - Create `envs/quaternion_ops.py` with JIT-compilable batched functions: `quat_multiply`, `quat_to_rotmat`, `quat_derivative(q, omega)`, and `quat_normalize(q)` with epsilon safety.
  - Create `envs/rocket_dynamics.py` implementing continuous-time state derivatives $\dot{x} = f(x, u, c_{\text{fault}})$ covering variable mass depletion $\dot{m} = -T / (I_{\text{sp}} g_0)$, dynamic inertia tensor $\mathbf{J}(m)$ updates, exponential atmospheric density $\rho(z)$, aerodynamic drag, TVC torque with CoM offset, and first-order gimbal lag $\dot{\delta} = (\delta_{\text{cmd}} - \delta)/\tau_{\text{act}}$.
  - Write `tests/test_rocket_dynamics.py` checking free-fall acceleration under zero thrust, mass conservation at zero throttle, and quaternion norm preservation (`max | ||q|| - 1.0 | < 1e-6`).
  - _Design: §3 AC-3.2, AC-3.3_

- [x] 2.2 Build batched RK4 numerical integrator and throughput test
  - Create `envs/rk4_integrator.py` implementing a fourth-order Runge-Kutta step `rk4_step(dynamics_fn, state, action, fault_params, dt)` operating over `[B, D]` CUDA/CPU tensors with post-step quaternion renormalization and propellant floor clamping ($m \ge m_{\text{dry}}$).
  - Write `tests/test_rk4_integrator.py` comparing numerical ballistic trajectories against analytical parabolic equations (`atol=1e-3`) and asserting throughput $\ge 200,000$ SPS at `B=1024` on CUDA.
  - _Design: §3 AC-3.1, AC-3.2_

- [x] 2.3 Implement mid-flight fault injection engine
  - Create `envs/fault_injector.py` implementing `FaultInjector` to sample and apply mid-episode failure modes at $t \ge t_{\text{fault}}$: `LOE_THRUST` ($20\%\text{–}40\%$ thrust scaling), `GIMBAL_BIAS` ($\pm 3.5^\circ$ additive offset), `GIMBAL_STUCK` (freezing $\delta_y$ or $\delta_z$ at onset angle), and `COM_SHIFT` (lateral shift in center-of-mass moment arm).
  - Write unit tests in `tests/test_fault_injector.py` verifying that each fault mode alters thrust or torque exactly at $t = t_{\text{fault}}$ and remains inactive for $t < t_{\text{fault}}$.
  - _Design: §2 AC-2.4, §3 AC-3.4_

- [x] 2.4 Wire vectorized Gymnasium environment and shaped guidance reward
  - Create `envs/rocket_vec_env.py` implementing `VectorizedRocketEnv` with auto-resetting sub-environments, domain randomization of initial altitude/velocity/attitude, 18-D observation normalization, and the composite reward function (glide-slope cone penalty, attitude tilt penalty, angular rate damping, fuel penalty, and $+100$ soft-landing terminal bonus).
  - Write `tests/test_rocket_vec_env.py` testing reset/step shapes (`[B, 18]`, `[B, 3]`), auto-reset on ground contact ($z \le 0$), and reward finite-value assertions over 1,000 random rollouts.
  - _Design: §3 AC-3.1, AC-3.5_

- [x] 3. Implement end-to-end skeleton pipeline and classical PID baseline
- [x] 3.1 Implement cascade PID flight controller baseline
  - Create `agents/pid_baseline.py` implementing `CascadePIDController(BaseController)` with outer-loop velocity/altitude proportional-derivative control and inner-loop attitude quaternion error to TVC gimbal $\delta_{y}, \delta_{z}$ and gravity-compensated throttle $u_T$.
  - Write `tests/test_pid_baseline.py` verifying that `CascadePIDController` achieves a >90% soft-landing success rate under `NOMINAL` conditions and degrades measurably under `GIMBAL_BIAS` and `LOE_THRUST`.
  - _Design: §5 AC-5.3_

- [x] 3.2 Implement pinned-RAM replay buffer and end-to-end rollout smoke test
  - Create `train/replay_buffer.py` implementing `PinnedSequenceReplayBuffer` pre-allocating contiguous CPU pinned memory (`pin_memory=True`) for `1,000,000` transitions (`obs`, `action`, `reward`, `done`, `fault_context`) and sampling sequence chunks `[B, T, ...]` with `non_blocking=True` CUDA transfers.
  - Write `tests/test_replay_buffer.py` verifying ring-buffer wrap-around, sequence episode-boundary masking, and GPU VRAM delta remaining under 500 MB during batch sampling.
  - _Design: §1, §6 AC-6.1_

- [x] 4. Build Recurrent Latent World Model and KL Fault Detector (FDIR)
- [x] 4.1 Implement Recurrent State-Space Model (RSSM) encoder, dynamics prior, and heads
  - Create `models/networks.py` with `NormMLP` blocks (`Linear -> LayerNorm -> Mish`) and orthogonal weight initialization.
  - Create `models/world_model.py` implementing `RocketWorldModel`: Posterior Encoder $q_\phi(z_t | h_t, o_t)$, GRU Recurrent Core $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$, Stochastic Prior Predictor $p_\theta(\hat{z}_t | h_t)$, Reward Predictor $R_\theta(h_t, z_t, a_t)$, and Twin Q-Value Ensemble $Q_{\psi_1, \psi_2}(h_t, z_t, a_t)$.
  - Write `tests/test_world_model.py` verifying multi-step latent imagination rollout `imagine_rollout(h_0, z_0, action_seq)` over $H=15$ steps under `torch.amp.autocast(dtype=torch.bfloat16)`.
  - _Design: §4 AC-4.1_

- [x] 4.2 Implement analytical KL-divergence anomaly detector and fault context isolator
  - Create `models/fdir_detector.py` implementing `FDIRModule`: exact Gaussian KL-divergence $D_{\text{KL}}(q_\phi(z_t | h_t, o_t) \| p_\theta(\hat{z}_t | h_t))$ with exponential moving average (EMA) threshold calibration, plus Fault Context Classifier/Regressor $\hat{c}_t = g_\omega(h_t, z_t)$ predicting fault mode logits and severity scalar.
  - Write `tests/test_fdir_detector.py` checking non-negative KL output, gradient flow with free-bits clipping (`free_bits=1.0`), and fault context tensor shapes `[B, 5]`.
  - _Design: §4 AC-4.2_

- [x] 5. Implement Fault-Conditioned Actor-Critic and Latent MPPI Planner
- [x] 5.1 Implement fault-conditioned squashed Gaussian Actor and TD($\lambda$) Critic loss
  - Create `agents/world_model_agent.py` implementing the Actor network $\pi_\theta(a_t | h_t, z_t, \hat{c}_t)$ with tanh-squashed log-probability Jacobian correction and Polyak-averaged target Twin Critics $\bar{Q}_{\psi_1, \psi_2}$.
  - Implement latent imagination policy gradient and TD($\lambda$) value target calculation in `agents/world_model_agent.py`.
  - Write `tests/test_world_model_agent.py` verifying actor entropy computation, target network soft updates (`tau=0.005`), and finite gradients during a single imagination step.
  - _Design: §5 AC-5.1_

- [x] 5.2 Implement GPU-parallel Latent MPPI trajectory planner
  - Create `agents/mppi_planner.py` implementing `LatentMPPIPlanner(BaseController)` that samples `K=512` action trajectories of horizon `H=15` (warm-started with 64 trajectories from $\pi_\theta$ and 448 Gaussian perturbations), evaluates cumulative imagined return via `RocketWorldModel` + `FDIRModule` fault context $\hat{c}_t$, and updates mean/std iteratively over 4 CEM/MPPI iterations.
  - Write `tests/test_mppi_planner.py` asserting that `LatentMPPIPlanner.act(obs)` executes in $< 15\text{ ms}$ per batch step on RTX 4060 under `bfloat16` and respects action bounds $[-1, 1]^3$.
  - _Design: §5 AC-5.2_

- [x] 5.3 Implement Model-Free PPO baseline for comparison
  - Create `agents/ppo_baseline.py` implementing a CleanRL-style vectorized `PPOController(BaseController)` with an MLP Actor-Critic and GAE($\lambda=0.95$) update loop for fair model-free vs. world-model comparison.
  - Write `tests/test_ppo_baseline.py` testing one rollout-and-update epoch on `VectorizedRocketEnv`.
  - _Design: §5 AC-5.3_

- [x] 6. Wire training pipeline and automated FDIR evaluation suite
- [x] 6.1 Implement joint World-Model + TD-MPC2 training loop
  - Create `train/trainer.py` wiring `VectorizedRocketEnv(num_envs=1024)`, `PinnedSequenceReplayBuffer`, `RocketWorldModel`, `FDIRModule`, and `LatentMPPIPlanner` with `torch.compile` and `bfloat16` mixed precision.
  - Implement the composite loss in `train/trainer.py`: $\mathcal{L}_{\text{total}} = \beta_1 \mathcal{L}_{\text{consistency}} + \beta_2 \mathcal{L}_{\text{KL}} + \beta_3 \mathcal{L}_{\text{reward}} + \beta_4 \mathcal{L}_{\text{value}} + \beta_5 \mathcal{L}_{\text{fault}}$, checkpoint saving, and JSONL metric logging.
  - Write integration test `tests/test_trainer_integration.py` executing 50 environment steps and 10 gradient updates while asserting peak CUDA memory `< 4.5 GB` and monotonic loss decrease on a fixed toy batch.
  - _Design: §1, §6 AC-6.1, AC-6.2_

- [x] 6.2 Build deterministic FDIR benchmark CLI and trajectory telemetry exporter
  - Create `eval/benchmark_fdir.py` to evaluate `CascadePIDController`, `PPOController`, and `LatentMPPIPlanner` across 100 episodes per fault mode (`NOMINAL`, `LOE_THRUST`, `GIMBAL_BIAS`, `GIMBAL_STUCK`, `COM_SHIFT`), computing landing success rate, mean KL detection latency (in steps and ms), fault classification accuracy, and propellant margin.
  - Export structured benchmark results to `eval_results.json` and generate publication-ready multi-panel comparison plots (`eval/plot_fdir_telemetry.py`) showing altitude, gimbal deflection, and KL-divergence surprise spikes at $t = t_{\text{fault}}$.
  - Write `tests/test_benchmark_fdir.py` running a 5-episode micro-benchmark across all three controllers and verifying schema completeness of `eval_results.json`.
  - _Design: §4 AC-4.2, §6 AC-6.3_

## Milestones / demo gates

> Implementation status: all implementation tasks are complete and covered by 119 passing
> tests. Revisions 2-5 addressed the update-to-data ratio (16 updates / 4096 transitions →
> 20k updates), free bits (1.0 → 0.05), the observation-innovation detector, explicit
> `a_{t-1}` + innovation features in the fault isolator, balanced 50% fault sampling,
> oracle fault-tolerant demonstrations, iterative DAgger and BC-regularised actor
> learning, a PPO training pipeline, the Section 6.3 terminal divert, exact physics
> residuals, the rescaled reward and the fault-conditioned residual controller.
>
> **All four gates pass (Revision 6, `results/TRAINING_REPORT.md` §9).**
> SPS 262 046 (B=1024); the classical baseline lands 95 % nominally and 0-40 % under
> faults (0-12.5 % on the 48/16-episode unit-test protocol); the FDIR detector alarms
> 0.90 steps (18 ms) after the onset with 100 % detection and 99-100 % isolation; and
> `AeroWM-FDIR` lands 95-100 % (actor) / 90-95 % (MPPI) in every fault mode. The
> mechanisms: exact one-step physics residuals replace the latent KL as the dominant
> anomaly signal, the fault class is latched, the reward is rescaled and the policy is
> a bounded residual around the fault-conditioned law, the CoM offset is read out from
> the *averaged* rotational residual, and a backup ACS takes over a frozen gimbal axis.

- **Gate 1 (After Task 2.4):** Physics Engine Demo — 1,024 rockets simulated in parallel at $>200,000$ SPS on RTX 4060 with mid-flight fault injection working.
- **Gate 2 (After Task 3.2):** Baseline Control Demo — Classical Cascade PID lands nominal rockets and fails reproducibly on stuck-gimbal and 35% thrust-loss faults.
- **Gate 3 (After Task 4.2):** Real-Time FDIR Anomaly Demo — World Model KL-divergence spikes within $\le 3$ steps ($60\text{ ms}$) of fault injection and isolates the fault class.
- **Gate 4 (After Task 6.2):** Full Portfolio Benchmark — Latent MPPI World Model recovers from mid-flight actuator/engine faults and outputs `eval_results.json` + telemetry plots comparing PID vs. PPO vs. AeroWM-FDIR.

## Coverage table

| Design Section & Criterion | Covered by Task(s) |
| :--- | :--- |
| `§1` Hardware & VRAM Constraints (`< 6.5 GB`, `bfloat16`) | `1.1`, `3.2`, `6.1` |
| `§2 AC-2.1` `RocketPhysicsConfig` schema | `1.1` |
| `§2 AC-2.2` `ObservationSchema` ($\mathbb{R}^{18}$) | `1.1`, `1.2` |
| `§2 AC-2.3` `ActionSchema` ($[-1, 1]^3$) | `1.1`, `1.2` |
| `§2 AC-2.4` `FaultInjectionSchema` ($\mathbb{R}^5$) | `1.1`, `2.3` |
| `§3 AC-3.1` Batched RK4 solver (`>= 200,000` SPS) | `2.2`, `2.4` |
| `§3 AC-3.2` Quaternion kinematics & renormalization | `2.1`, `2.2` |
| `§3 AC-3.3` Variable mass, inertia, drag, actuator lag | `2.1` |
| `§3 AC-3.4` Mid-flight fault injection engine | `2.3` |
| `§3 AC-3.5` Vectorized Gymnasium wrapper & shaped reward | `2.4` |
| `§4 AC-4.1` RSSM World Model (`Encoder`, `GRU`, `Prior`, `Reward`, `Q`) | `4.1` |
| `§4 AC-4.2` Analytical KL Fault Detector & Context Classifier (`<= 60 ms`) | `4.2`, `6.2` |
| `§5 AC-5.1` Fault-conditioned Actor & TD($\lambda$) Twin Critics | `5.1` |
| `§5 AC-5.2` Latent MPPI Planner (`K=512, H=15`) | `5.2` |
| `§5 AC-5.3` Cascade PID & PPO Baselines (`BaseController`) | `1.2`, `3.1`, `5.3` |
| `§6 AC-6.1` Pinned-RAM Replay Buffer (`1M` transitions) | `3.2`, `6.1` |
| `§6 AC-6.2` Joint World-Model + TD-MPC2 training loop | `6.1` |
| `§6 AC-6.3` Automated FDIR evaluation suite (`eval_results.json`) | `6.2` |
