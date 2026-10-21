# Design Document — AeroWM-FDIR: 6-DOF Rocket World Model & Fault-Tolerant Control

## §1 System Overview & Hardware Constraints
- **Target Hardware:** Single Laptop (Intel Core i7 13th Gen, 24GB RAM, NVIDIA RTX 4060 8GB VRAM, Arch Linux).
- **Compute & Memory Budget:** Peak GPU VRAM usage MUST remain strictly below 6.5 GB (`bfloat16` mixed precision + `torch.compile`). Vectorized physics rollouts (`N=1024` environments) execute on GPU tensors; replay buffer (`1,000,000` transitions) resides in pinned host RAM (24GB).
- **Objective:** Train a decoder-free Latent World Model (TD-MPC2 + Recurrent State-Space KL Anomaly Detector) capable of guiding a 6-DOF reusable rocket through powered ascent/descent while detecting, isolating, and recovering from actuator and propulsion faults in under 50 ms (2–3 integration steps).

## §2 Shared Contracts, Schemas & Configuration (`configs/schema.py`)
- **AC-2.1:** Frozen `RocketPhysicsConfig` dataclass specifying dry mass ($m_{\text{dry}} = 22,200\text{ kg}$), propellant mass ($m_{\text{prop}} = 15,000\text{ kg}$), max thrust ($T_{\max} = 845,000\text{ N}$), throttle bounds ($u_T \in [0.40, 1.00]$), specific impulse ($I_{\text{sp}} = 282\text{ s}$), gimbal limits ($\delta_{y}, \delta_{z} \in [-8^\circ, +8^\circ]$), actuator time constant ($\tau_{\text{act}} = 0.035\text{ s}$), integration timestep ($\Delta t = 0.02\text{ s}$).
- **AC-2.2:** Frozen `ObservationSchema` ($o_t \in \mathbb{R}^{18}$): position $\mathbf{r} \in \mathbb{R}^3$, velocity $\mathbf{v} \in \mathbb{R}^3$, unit quaternion $\mathbf{q} \in \mathbb{R}^4$, body angular rates $\boldsymbol{\omega} \in \mathbb{R}^3$, normalized mass $\tilde{m} \in \mathbb{R}^1$, actual gimbal angles $[\delta_y, \delta_z] \in \mathbb{R}^2$, measured chamber thrust $\tilde{T} \in \mathbb{R}^1$, dynamic pressure $q_{\text{dyn}} \in \mathbb{R}^1$.
- **AC-2.3:** Frozen `ActionSchema` ($a_t \in [-1, 1]^3$): normalized pitch gimbal command, yaw gimbal command, and throttle command.
- **AC-2.4:** Frozen `FaultInjectionSchema` ($c_t \in \mathbb{R}^5$): fault mode enum (`NOMINAL=0`, `LOE_THRUST=1`, `GIMBAL_BIAS=2`, `GIMBAL_STUCK=3`, `COM_SHIFT=4`), fault severity scalar, onset timestep $t_{\text{fault}}$.

## §3 Vectorized 6-DOF Rocket Physics Engine (`envs/`)
- **AC-3.1:** Batched Runge-Kutta 4th Order (RK4) solver in PyTorch (`envs/rk4_integrator.py`) supporting `N >= 1024` parallel environments on CUDA with zero Python loops across batch dimension and throughput $\ge 200,000$ SPS on RTX 4060.
- **AC-3.2:** Quaternion kinematics $\dot{\mathbf{q}} = \frac{1}{2} \mathbf{\Omega}(\boldsymbol{\omega}) \mathbf{q}$ with explicit unit-norm renormalization after each RK4 step (`max | ||q|| - 1.0 | < 1e-6`).
- **AC-3.3:** Variable mass depletion $\dot{m} = -T / (I_{\text{sp}} g_0)$, time-varying diagonal inertia tensor $\mathbf{J}(m)$, aerodynamic drag with altitude-dependent exponential atmosphere $\rho(z) = \rho_0 \exp(-z/H)$, and first-order actuator lag $\dot{\delta} = (\delta_{\text{cmd}} - \delta) / \tau_{\text{act}}$.
- **AC-3.4:** Fault injection wrapper (`envs/fault_injector.py`) capable of injecting mid-flight Loss of Engine Effectiveness (20%-40% thrust drop), TVC pitch/yaw bias ($\pm 3.5^\circ$), stuck gimbal actuator, and Center-of-Mass shift.
- **AC-3.5:** Gymnasium VectorEnv-compatible wrapper (`envs/rocket_vec_env.py`) with shaped reward function combining glide-slope tracking, attitude stability penalty, angular rate damping, fuel conservation, and terminal landing bonus.

## §4 Latent World Model & KL Fault Detector (`models/`)
- **AC-4.1:** Recurrent State-Space World Model (`models/world_model.py`) with Encoder $q_\phi(z_t | h_t, o_t)$, GRU Recurrent Backbone $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$, Stochastic Prior Dynamics Predictor $p_\theta(\hat{z}_t | h_t)$, Reward Predictor $R_\theta(h_t, z_t, a_t)$, and Terminal Value/Q-Ensemble $Q_{\psi_1, \psi_2}(h_t, z_t, a_t)$ using LayerNorm + Mish activations (`latent_dim=256`, `gru_dim=256`).
- **AC-4.2:** Real-time FDIR Anomaly Detector (`models/fdir_detector.py`) computing exact analytical KL-divergence $D_{\text{KL}}(q_\phi(z_t | h_t, o_t) \| p_\theta(\hat{z}_t | h_t))$ and supervised/contrastive Fault Context Head $\hat{c}_t = g_\omega(h_t, z_t)$ achieving fault detection latency $\le 3$ timesteps ($60\text{ ms}$) and fault classification accuracy $\ge 92\%$.

## §5 Latent MPPI Planner, Actor-Critic & Baselines (`agents/`)
- **AC-5.1:** Fault-conditioned Actor Policy $\pi_\theta(a_t | h_t, z_t, \hat{c}_t)$ outputting squashed Gaussian actions and Critic ensemble trained with temporal-difference targets ($\lambda$-returns).
- **AC-5.2:** Latent Model Predictive Path Integral (`agents/mppi_planner.py`) rolling out `K=512` candidate action sequences over horizon `H=15` inside the learned World Model under `torch.amp.autocast(dtype=torch.bfloat16)`, warm-started by the Actor policy.
- **AC-5.3:** Classical Cascade PID / LQR flight controller (`agents/pid_baseline.py`) and Model-Free PPO agent (`agents/ppo_baseline.py`) implementing a shared `BaseController` interface for apples-to-apples fault recovery benchmarking.

## §6 Training Pipeline & Evaluation Benchmark (`train/` & `eval/`)
- **AC-6.1:** Pinned-RAM Replay Buffer (`train/replay_buffer.py`) storing `1,000,000` transitions in CPU RAM with asynchronous non-blocking batch transfers to GPU (`pin_memory=True`, `non_blocking=True`) keeping GPU VRAM $< 4.5\text{ GB}$.
- **AC-6.2:** Joint World Model + TD-MPC2 trainer (`train/trainer.py`) optimizing latent consistency loss, KL balancing loss (free bits = 1.0), reward/value loss, and fault identification loss.
- **AC-6.3:** Automated evaluation suite (`eval/benchmark_fdir.py`) running deterministic Monte Carlo episodes across all 5 fault modes and exporting structured metrics (`eval_results.json`) verifying safe landing rate $\ge 85\%$ under fault injection vs. PID baseline.
