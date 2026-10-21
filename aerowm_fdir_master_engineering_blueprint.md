# Master Engineering Blueprint: AeroWM-FDIR
**Latent World Model & Model Predictive Control for 6-DOF Rocket Powered Descent and Real-Time Fault Detection, Isolation, and Recovery (FDIR)**

---

## 0. AI Coding Assistant System Directives (READ FIRST)

> **CRITICAL INSTRUCTIONS FOR THE EXECUTING AI AGENT:**
> 1. **Hardware Target:** This codebase runs on a single Linux workstation/laptop equipped with an **NVIDIA RTX 4060 (8GB VRAM), 13th-Gen Intel Core i7 CPU, and 24GB System RAM**.
> 2. **Strict Memory Budget:** Total GPU VRAM consumption during training MUST NOT exceed **6.0 GB**. Always store the Experience Replay Buffer in **pinned CPU System RAM** (`pin_memory=True`) or pre-allocated GPU ring buffers of size $\le 250,000$ transitions.
> 3. **Vectorization Mandate:** Do NOT use standard Python `for` loops to step through multiple environments. All physics simulations (`Rocket6DOFEnv`) MUST be implemented in batched PyTorch tensors (`torch.Tensor` of shape `(B, ...)`) running natively on `cuda:0` with `torch.compile` and `torch.bfloat16` mixed precision where applicable.
> 4. **Zero Black-Box RL Wrappers:** Do NOT import `stable-baselines3`, `ray[rllib]`, or `tianshou`. Implement the **TD-MPC2 + Recurrent State-Space Fault Detector (RSSM-FDIR)** architecture cleanly in pure, modular **PyTorch** (CleanRL / TorchRL style) so every tensor operation is transparent and modifiable.
> 5. **Numerical Stability:** Quaternions must be normalized after every Runge-Kutta 4th-order (RK4) integration step. Avoid Euler angles inside the dynamics engine to prevent gimbal lock.

---

## 1. Executive Summary & Project Objective

**AeroWM-FDIR** is an end-to-end physics-informed Reinforcement Learning and World Modeling system for **6-Degrees-of-Freedom (6-DOF) Rocket Powered Descent Guidance ( Falcon-9 / Starship style landing)** under severe actuator, propulsion, and sensor failures.

Instead of relying on brittle model-free policies (like vanilla PPO), this project combines:
1. **A GPU-Vectorized 6-DOF Rigid-Body Rocket Simulator** with variable mass, aerodynamic drag, reaction control system (RCS), thrust vector control (TVC) actuator lag, and a programmable **Fault Injection Engine**.
2. **A Latent World Model with Surprise-Driven Anomaly Detection:** A Recurrent State-Space Model (RSSM) combined with a **TD-MPC2 (Temporal Difference Model Predictive Control)** planner.
3. **Real-Time FDIR (Fault Detection, Isolation, and Recovery):** Uses the Kullback-Leibler ($D_{\text{KL}}$) divergence and observation prediction residual between the World Model's prior imagination $\hat{z}_{t+1}$ and posterior sensor encoding $z_{t+1}$ to detect and classify engine/gimbal faults within $< 60\text{ ms}$ (3 simulation steps) and dynamically replan the landing trajectory.

---

## 2. Repository Directory Structure

The AI coding assistant must generate and maintain the following exact modular project layout:

```text
aerowm_fdir/
├── pyproject.toml                 # Dependencies: torch>=2.2, gymnasium, numpy, wandb, matplotlib, typer
├── README.md                      # Documentation, architecture diagrams, and benchmark tables
├── configs/
│   └── default_config.py          # Dataclass containing all physics, world-model, and MPC hyperparameters
├── envs/
│   ├── __init__.py
│   ├── rocket_physics.py          # Batched PyTorch 6-DOF RK4 differential equations & quaternion math
│   ├── fault_injector.py          # Stochastic fault generation (LOE, gimbal lock, CoM shift, IMU noise)
│   └── vec_rocket_env.py          # GPU-vectorized Gymnasium-compatible wrapper (runs 512 envs in parallel)
├── models/
│   ├── __init__.py
│   ├── networks.py                # SimNorm, MLP blocks with LayerNorm + Mish activations, GRU cell
│   ├── world_model.py             # Latent Encoder, Recurrent Prior/Posterior Dynamics, Reward & Value Heads
│   └── fdir_head.py               # Anomaly score computation (KL Surprise) + Supervised/Self-Supervised Fault Classifier
├── planners/
│   ├── __init__.py
│   ├── mppi_planner.py            # Batched Model Predictive Path Integral (MPPI) trajectory optimizer in latent space
│   └── pid_baseline.py            # Classical cascaded PID / LQR flight controller for benchmark comparison
├── training/
│   ├── __init__.py
│   ├── replay_buffer.py           # Episode-aware vectorized ring buffer in CPU/GPU RAM
│   └── train_world_model.py       # Main training script (jointly trains World Model, Policy Prior, Q-Ensemble, FDIR)
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_fdir.py           # Evaluates ROC-AUC of fault detection latency and landing success rate
│   └── visualize_trajectory.py    # Generates 3D landing trajectory plots, gimbal traces, and KL-surprise spikes
└── tests/
    ├── test_physics_rk4.py        # Unit tests for quaternion norm preservation, mass depletion, and free-fall
    └── test_world_model_shapes.py # Unit tests for tensor shapes across imagination horizon H
```

---

## 3. Exact Mathematical Formulation of the 6-DOF Rocket Environment

### 3.1 Coordinate Frames
* **Inertial Frame ($\mathcal{I}$):** East-North-Up (ENU) surface landing pad frame where origin $(0,0,0)$ is the center of the landing pad and $+z$ points vertically upward against gravity $\mathbf{g}_{\mathcal{I}} = [0, 0, -g_0]^T$, with $g_0 = 9.80665\text{ m/s}^2$.
* **Body Frame ($\mathcal{B}$):** Origin at the rocket's instantaneous Center of Mass (CoM). The $+x_{\mathcal{B}}$ axis points along the rocket's longitudinal nose axis; $+y_{\mathcal{B}}$ and $+z_{\mathcal{B}}$ form the lateral pitch/yaw axes.

### 3.2 State Vector ($\mathbf{x}_t \in \mathbb{R}^{17}$)
Every parallel environment $i \in \{1, \dots, B\}$ maintains a ground-truth physical state vector $\mathbf{x}_t$:
$$\mathbf{x}_t = \begin{bmatrix} \mathbf{r}_{\mathcal{I}} \\ \mathbf{v}_{\mathcal{I}} \\ \mathbf{q}_{\mathcal{B}\to\mathcal{I}} \\ \boldsymbol{\omega}_{\mathcal{B}} \\ m \\ \delta_y \\ \delta_z \\ T_{\text{actual}} \end{bmatrix} \in \mathbb{R}^{17}$$

Where:
1. $\mathbf{r}_{\mathcal{I}} = [r_x, r_y, r_z]^T \in \mathbb{R}^3$: Position in Inertial Frame ($\text{m}$).
2. $\mathbf{v}_{\mathcal{I}} = [v_x, v_y, v_z]^T \in \mathbb{R}^3$: Velocity in Inertial Frame ($\text{m/s}$).
3. $\mathbf{q}_{\mathcal{B}\to\mathcal{I}} = [q_w, q_x, q_y, q_z]^T \in \mathbb{R}^4$: Unit quaternion representing orientation from Body to Inertial frame ($\|\mathbf{q}\|_2 = 1$).
4. $\boldsymbol{\omega}_{\mathcal{B}} = [\omega_x, \omega_y, \omega_z]^T \in \mathbb{R}^3$: Angular velocity vector in Body Frame ($\text{rad/s}$).
5. $m \in \mathbb{R}$: Instantaneous total vehicle mass ($\text{kg}$), bounded by $[m_{\text{dry}}, m_{\text{wet}}]$.
6. $\delta_y, \delta_z \in \mathbb{R}^2$: Actual physical TVC gimbal pitch and yaw deflection angles ($\text{rad}$), modeling first-order hydraulic actuator lag.
7. $T_{\text{actual}} \in \mathbb{R}$: Actual instantaneous main engine thrust ($\text{N}$), modeling turbopump spool-up lag.

### 3.3 Action Vector ($\mathbf{a}_t \in [-1, 1]^4$)
The agent outputs a normalized continuous action vector at $50\text{ Hz}$ ($\Delta t = 0.02\text{ s}$):
$$\mathbf{a}_t = \begin{bmatrix} u_{\delta_y} & u_{\delta_z} & u_{\text{throttle}} & u_{\text{rcs}} \end{bmatrix}^T \in [-1, 1]^4$$

Scaled to physical commands:
* **Commanded TVC Pitch Gimbal:** $\delta_{y,\text{cmd}} = u_{\delta_y} \cdot \delta_{\max}$, where $\delta_{\max} = 10^\circ \approx 0.1745\text{ rad}$.
* **Commanded TVC Yaw Gimbal:** $\delta_{z,\text{cmd}} = u_{\delta_z} \cdot \delta_{\max}$.
* **Commanded Main Engine Throttle:** Mapped linearly from $u_{\text{throttle}} \in [-1, 1]$ to $[T_{\min}, T_{\max}]$, where $T_{\min} = 0.40 \cdot T_{\max}$ (simulating realistic liquid rocket minimum deep-throttle limit).
* **Commanded Roll Cold-Gas RCS Torque:** $\tau_{x,\text{rcs}} = u_{\text{rcs}} \cdot \tau_{\text{rcs},\max}$.

### 3.4 6-DOF Equations of Motion (Solved via 4th-Order Runge-Kutta)
Let $\mathbf{R}(\mathbf{q}) \in SO(3)$ be the Direction Cosine Matrix converting vectors from Body frame $\mathcal{B}$ to Inertial frame $\mathcal{I}$:
$$\mathbf{R}(\mathbf{q}) = \begin{bmatrix} 1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_z q_w) & 2(q_x q_z + q_y q_w) \\ 2(q_x q_y + q_z q_w) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_x q_w) \\ 2(q_x q_z - q_y q_w) & 2(q_y q_z + q_x q_w) & 1 - 2(q_x^2 + q_y^2) \end{bmatrix}$$

1. **Actuator First-Order Lag Dynamics:**
   $$\dot{\delta}_y = \frac{\delta_{y,\text{cmd}} - \delta_y}{\tau_{\text{gimbal}}}, \quad \dot{\delta}_z = \frac{\delta_{z,\text{cmd}} - \delta_z}{\tau_{\text{gimbal}}}, \quad \dot{T}_{\text{actual}} = \frac{T_{\text{cmd}} - T_{\text{actual}}}{\tau_{\text{engine}}}$$
   where $\tau_{\text{gimbal}} = 0.05\text{ s}$ and $\tau_{\text{engine}} = 0.10\text{ s}$.

2. **Thrust Vector in Body Frame ($\mathbf{F}_{T,\mathcal{B}}$):**
   $$\mathbf{F}_{T,\mathcal{B}} = T_{\text{actual}} \begin{bmatrix} \cos(\delta_y)\cos(\delta_z) \\ -\sin(\delta_z) \\ -\cos(\delta_z)\sin(\delta_y) \end{bmatrix}$$

3. **Aerodynamic Drag Force in Inertial Frame ($\mathbf{F}_{\text{aero},\mathcal{I}}$):**
   $$\mathbf{v}_{\text{rel},\mathcal{I}} = \mathbf{v}_{\mathcal{I}} - \mathbf{v}_{\text{wind},\mathcal{I}}, \quad \rho(r_z) = \rho_0 \exp\left(-\frac{\max(0, r_z)}{H_{\text{scale}}}\right)$$
   $$\mathbf{F}_{\text{aero},\mathcal{I}} = -\frac{1}{2} \rho(r_z) \|\mathbf{v}_{\text{rel},\mathcal{I}}\|_2 C_D A_{\text{ref}} \mathbf{v}_{\text{rel},\mathcal{I}}$$

4. **Translational Dynamics:**
   $$\dot{\mathbf{r}}_{\mathcal{I}} = \mathbf{v}_{\mathcal{I}}, \quad \dot{\mathbf{v}}_{\mathcal{I}} = \frac{1}{m} \left( \mathbf{R}(\mathbf{q}) \mathbf{F}_{T,\mathcal{B}} + \mathbf{F}_{\text{aero},\mathcal{I}} \right) + \mathbf{g}_{\mathcal{I}}$$

5. **Torque & Rotational Dynamics (Euler's Equations):**
   Let $\mathbf{r}_{\text{engine},\mathcal{B}} = [-L_{\text{com}}, \Delta y_{\text{com}}, \Delta z_{\text{com}}]^T$ be the moment arm from the Center of Mass to the engine gimbal pivot (where $\Delta y_{\text{com}}, \Delta z_{\text{com}}$ are $0$ nominally, and non-zero under a CoM fault):
   $$\boldsymbol{\tau}_{\mathcal{B}} = \mathbf{r}_{\text{engine},\mathcal{B}} \times \mathbf{F}_{T,\mathcal{B}} + \begin{bmatrix} \tau_{x,\text{rcs}} \\ 0 \\ 0 \end{bmatrix} - \mathbf{d}_{\text{damp}} \odot \boldsymbol{\omega}_{\mathcal{B}}$$
   $$\dot{\boldsymbol{\omega}}_{\mathcal{B}} = \mathbf{J}^{-1} \left( \boldsymbol{\tau}_{\mathcal{B}} - \boldsymbol{\omega}_{\mathcal{B}} \times (\mathbf{J} \boldsymbol{\omega}_{\mathcal{B}}) \right)$$
   where $\mathbf{J} = \text{diag}(J_{xx}, J_{yy}, J_{zz})$ is the instantaneous inertia tensor scaled linearly with mass $m$.

6. **Quaternion Kinematics:**
   $$\dot{\mathbf{q}} = \frac{1}{2} \boldsymbol{\Omega}(\boldsymbol{\omega}_{\mathcal{B}}) \mathbf{q}, \quad \text{where } \boldsymbol{\Omega}(\boldsymbol{\omega}) = \begin{bmatrix} 0 & -\omega_x & -\omega_y & -\omega_z \\ \omega_x & 0 & \omega_z & -\omega_y \\ \omega_y & -\omega_z & 0 & \omega_x \\ \omega_z & \omega_y & -\omega_x & 0 \end{bmatrix}$$
   *(Re-normalize $\mathbf{q} \leftarrow \mathbf{q} / \|\mathbf{q}\|_2$ after every RK4 step).*

7. **Mass Depletion Dynamics:**
   $$\dot{m} = -\frac{T_{\text{actual}}}{I_{\text{sp}} g_0}$$

### 3.5 Nominal Vehicle Parameters (Sub-Scale Falcon-9 Landing Stage)
Use normalized/well-conditioned physical constants so neural network gradients remain stable:
* Wet Mass $m_{\text{wet}} = 25,000\text{ kg}$, Dry Mass $m_{\text{dry}} = 18,000\text{ kg}$
* Max Thrust $T_{\max} = 450,000\text{ N}$, Specific Impulse $I_{\text{sp}} = 282\text{ s}$
* Half-Length $L_{\text{com}} = 15.0\text{ m}$, Reference Area $A_{\text{ref}} = 10.75\text{ m}^2$, Drag Coefficient $C_D = 0.75$
* Initial Altitude $r_{z,0} \sim \mathcal{U}(800, 1200)\text{ m}$, Initial Lateral Offset $r_{x,0}, r_{y,0} \sim \mathcal{U}(-150, 150)\text{ m}$
* Initial Descent Velocity $v_{z,0} \sim \mathcal{U}(-85, -55)\text{ m/s}$, Lateral Velocity $v_{x,0}, v_{y,0} \sim \mathcal{U}(-20, 20)\text{ m/s}$

---

## 4. Programmable Fault Injection Engine (`envs/fault_injector.py`)

Each episode has a probability $p_{\text{fault}} = 0.65$ of experiencing a fault at a random timestep $t_{\text{fault}} \sim \mathcal{U}(30, 200)$. The environment maintains a ground-truth fault label $y_{\text{fault}} \in \{0, 1, 2, 3, 4\}$ and continuous fault parameter vector $\mathbf{c}_t \in \mathbb{R}^5$ for evaluation and auxiliary supervision:

| Fault ID | Fault Mode Name | Mathematical Injection Modification | Real-World Aerospace Equivalent |
| :--- | :--- | :--- | :---
| `0` | **Nominal Flight** | No modification ($\alpha_T = 1.0, b_\delta = 0$) | Nominal trajectory |
| `1` | **Loss of Engine Effectiveness (LOE)** | $T_{\text{cmd}} \leftarrow \alpha_T T_{\text{cmd}}$, where $\alpha_T \sim \mathcal{U}(0.60, 0.82)$ | Turbopump cavitation, fuel valve partial blockage |
| `2` | **TVC Gimbal Bias / Stuck Actuator** | $\delta_{y,\text{cmd}} \leftarrow \delta_{y,\text{cmd}} + b_{\delta_y}$ or frozen $\dot{\delta}_y = 0$, $b_{\delta_y} \sim \mathcal{U}(\pm 2.5^\circ, \pm 5.0^\circ)$ | Hydraulic servo valve jam or linkage offset |
| `3` | **Sudden CoM / Slosh Offset** | $\Delta y_{\text{com}} \sim \mathcal{U}(-0.35\text{ m}, 0.35\text{ m})$, inducing parasitic pitch/yaw torque | Propellant slosh baffle failure or asymmetric payload shift |
| `4` | **IMU Gyro Drift & Wind Gust Shear** | Sensor observation $\tilde{\boldsymbol{\omega}}_{\mathcal{B}} = \boldsymbol{\omega}_{\mathcal{B}} + \mathbf{b}_{\omega} + \mathcal{N}(0, \sigma^2)$ plus $25\text{ m/s}$ lateral wind step | Rate-gyro saturation or severe atmospheric shear layer |

### Sensor Observation Vector ($\mathbf{o}_t \in \mathbb{R}^{20}$)
The agent DOES NOT observe the ground-truth fault ID $y_{\text{fault}}$ or the hidden fault parameters. It observes:
$$\mathbf{o}_t = \text{Concat}\left(\tilde{\mathbf{r}}_{\mathcal{I}}, \tilde{\mathbf{v}}_{\mathcal{I}}, \tilde{\mathbf{q}}, \tilde{\boldsymbol{\omega}}_{\mathcal{B}}, \tilde{m}, \tilde{\delta}_y, \tilde{\delta}_z, \tilde{T}_{\text{meas}}, \mathbf{a}_{t-1}\right) \in \mathbb{R}^{21}$$
*(All observations must be normalized via running mean/std or fixed physical scale factors before entering the neural network).*

---

## 5. Reward Function & Safety Constraints

To prevent reward hacking (such as hovering indefinitely until fuel runs out or crashing horizontally at the pad origin), implement the following dense potential-based shaping reward plus terminal landing bonus:

1. **Glide-Slope & Distance Shaping ($r_{\text{pos}}$):**
   Define target velocity profile pointing toward the pad origin with desired descent rate:
   $$\mathbf{v}_{\text{targ}}(\mathbf{r}_{\mathcal{I}}) = -v_{\max} \frac{\mathbf{r}_{\mathcal{I}}}{\|\mathbf{r}_{\mathcal{I}}\|_2 + 10.0} \quad \implies \quad r_{\text{track},t} = -0.05 \|\mathbf{v}_{\mathcal{I}} - \mathbf{v}_{\text{targ}}(\mathbf{r}_{\mathcal{I}})\|_2 - 0.01 \|\mathbf{r}_{\mathcal{I}}\|_2$$

2. **Attitude Upright Penalty ($r_{\text{att}}$):**
   Let $\cos\theta_{\text{tilt}} = R_{33}(\mathbf{q}) = 1 - 2(q_x^2 + q_y^2)$ be the cosine of the tilt angle from vertical:
   $$r_{\text{att},t} = -0.5 (1 - \cos\theta_{\text{tilt}})^2 - 0.02 \|\boldsymbol{\omega}_{\mathcal{B}}\|_2^2$$

3. **Control Smoothness & Gimbal Slew Penalty ($r_{\text{ctrl}}$):**
   $$r_{\text{ctrl},t} = -0.02 \|\mathbf{a}_t - \mathbf{a}_{t-1}\|_2^2 - 0.01 \left(\delta_y^2 + \delta_z^2\right)$$

4. **Terminal Conditions (Evaluated when $r_z \le 0$ or $\cos\theta_{\text{tilt}} < \cos(60^\circ)$ or $m \le m_{\text{dry}}$):**
   * **Successful Soft Landing ($r_z \le 0.5\text{ m}$):**
     * Lateral error $\sqrt{r_x^2 + r_y^2} < 5.0\text{ m}$
     * Vertical touchdown speed $|v_z| < 2.5\text{ m/s}$ and lateral speed $\sqrt{v_x^2 + v_y^2} < 1.5\text{ m/s}$
     * Tilt angle $\theta_{\text{tilt}} < 8^\circ$ and $\|\boldsymbol{\omega}_{\mathcal{B}}\|_2 < 0.25\text{ rad/s}$
     * **Bonus:** $+200.0 + 10.0 \cdot \left(\frac{m - m_{\text{dry}}}{m_{\text{wet}} - m_{\text{dry}}}\right)$
   * **Crash / Out-of-Bounds Penalty:**
     * $-100.0 - 2.0 \|\mathbf{v}_{\mathcal{I}}\|_2$

---

## 6. Neural Architecture: RSSM-FDIR + TD-MPC2 World Model

This architecture combines the deterministic/stochastic state decomposition of **RSSM** (for sharp KL-divergence anomaly detection) with **SimNorm** latent stabilization and **TD-MPC2** trajectory planning.

```
                   +---------------------------------------------------+
                   |         6-DOF Rocket Environment (CUDA)           |
                   |   Fault Injector: LOE, Gimbal Lock, CoM, Wind     |
                   +-------------------------+-------------------------+
                                             | o_t (21-dim telemetry)
                                             v
+-----------------------+          +-----------------------------------+
| Recurrent History h_t | -------> | Posterior Encoder q(z_t | h_t,o_t)|
|  GRU(h_{t-1}, z_{t-1},|          |   MLP + SimNorm -> z_t (256-dim)  |
|      a_{t-1}) (256-d) |          +-----------------+-----------------+
+-----------+-----------+                            |
            |                                        |
            v                                        v
+-----------------------------------+      +---------------------------+
| Prior Predictor p(^z_t | h_t)     | ---> |   KL Surprise & FDIR Head |
|   MLP + SimNorm -> ^z_t (256-dim) |      | D_KL(q || p) -> Fault Cls |
+-----------------------------------+      +-------------+-------------+
                                                         | c_t (Fault Context)
                                                         v
                                           +---------------------------+
                                           |   TD-MPC2 MPPI Planner    |
                                           | Rolls out H=12 steps using|
                                           | Prior + Reward + Q-Ensemb.|
                                           +-------------+-------------+
                                                         | a_t (4-dim action)
                                                         v
                                               [ Applied to Rocket ]
```

### 6.1 Simplicial Normalization (`SimNorm`)
To prevent latent representation collapse or exploding embeddings during multi-step imagination without a pixel decoder, apply `SimNorm` to all latent vectors $z_t \in \mathbb{R}^{256}$. Reshape $z_t$ into $G = 32$ groups of dimension $V = 8$, and apply Softmax with temperature $\tau_{\text{sim}} = 1.0$ within each group:
$$\text{SimNorm}(z)_g = \text{Softmax}\left(\frac{z_g}{\tau_{\text{sim}}}\right), \quad \forall g \in \{1, \dots, 32\}$$

### 6.2 Core Modules (`models/world_model.py` & `models/fdir_head.py`)
All MLPs use `Linear -> LayerNorm -> Mish` blocks.

1. **Observation Encoder ($e_\phi$):**
   * Input: Normalized telemetry $\mathbf{o}_t \in \mathbb{R}^{21}$
   * Architecture: 2-layer MLP (`21 -> 256 -> 256`)
   * Output: Observation embedding $\mathbf{e}_t \in \mathbb{R}^{256}$

2. **Recurrent Deterministic State Model ($f_\theta$):**
   * Maintains temporal memory across sensor noise and partial observability (crucial for inferring hidden faults like gimbal bias or thrust loss over a few frames).
   * Input: Previous deterministic state $\mathbf{h}_{t-1} \in \mathbb{R}^{256}$ and projected $[\mathbf{z}_{t-1}, \mathbf{a}_{t-1}] \in \mathbb{R}^{256}$
   * Architecture: `nn.GRUCell(256, 256)`
   * Output: $\mathbf{h}_t \in \mathbb{R}^{256}$

3. **Posterior State Representation ($q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{e}_t)$):**
   * Input: $\text{Concat}(\mathbf{h}_t, \mathbf{e}_t) \in \mathbb{R}^{512}$
   * Output: Gaussian parameters $(\boldsymbol{\mu}_{\text{post}}, \log\boldsymbol{\sigma}_{\text{post}}) \in \mathbb{R}^{128 \times 2}$ AND a SimNorm latent projection $\mathbf{s}_t = \text{Concat}(\mathbf{h}_t, \mathbf{z}_t) \in \mathbb{R}^{384}$.
   * Reparameterization trick: $\mathbf{z}_t = \boldsymbol{\mu}_{\text{post}} + \boldsymbol{\sigma}_{\text{post}} \odot \boldsymbol{\epsilon}$, with $\sigma_{\min} = 0.05$.

4. **Prior Latent Transition Predictor ($p_\theta(\hat{\mathbf{z}}_t \mid \mathbf{h}_t)$):**
   * Predicts the distribution of $\mathbf{z}_t$ *before* seeing observation $\mathbf{o}_t$.
   * Input: $\mathbf{h}_t \in \mathbb{R}^{256}$
   * Output: Prior Gaussian $(\boldsymbol{\mu}_{\text{prior}}, \log\boldsymbol{\sigma}_{\text{prior}}) \in \mathbb{R}^{128 \times 2}$.

5. **FDIR Anomaly Detector & Fault Isolation Head (`models/fdir_head.py`):**
   * **Instantaneous Physics Surprise ($S_t$):**
     $$S_t = D_{\text{KL}}\Big( \mathcal{N}(\boldsymbol{\mu}_{\text{post},t}, \boldsymbol{\sigma}_{\text{post},t}) \,\Big\|\, \mathcal{N}(\boldsymbol{\mu}_{\text{prior},t}, \boldsymbol{\sigma}_{\text{prior},t}) \Big)$$
   * **Fault Context Encoder ($g_\psi$):**
     Takes $\text{Concat}(\mathbf{h}_t, \mathbf{z}_t, \boldsymbol{\mu}_{\text{post},t} - \boldsymbol{\mu}_{\text{prior},t}) \in \mathbb{R}^{512}$ and outputs:
     * Predicted fault class logits $\hat{\mathbf{y}}_{\text{fault}} \in \mathbb{R}^5$ (5 classes: Nominal, LOE, Gimbal Bias, CoM Shift, IMU/Wind)
     * Estimated continuous fault embedding $\hat{\mathbf{c}}_t \in \mathbb{R}^{16}$, which is concatenated into the Reward, Value, and Policy networks so the planner immediately adapts its trajectory to the detected fault.

6. **Reward Predictor ($R_\theta$), Twin Q-Value Ensemble ($Q_{\theta_1}, Q_{\theta_2}$), and Policy Prior ($\pi_\theta$):**
   * Conditioned on combined latent state $\tilde{\mathbf{s}}_t = \text{Concat}(\mathbf{h}_t, \mathbf{z}_t, \hat{\mathbf{c}}_t) \in \mathbb{R}^{400}$.
   * **Reward Head:** `MLP(400 + 4 -> 256 -> 256 -> 1)` predicting scalar reward $\hat{r}_t$.
   * **Value Ensemble:** Two independent MLPs $Q_{\theta_1}(\tilde{\mathbf{s}}_t, \mathbf{a}_t), Q_{\theta_2}(\tilde{\mathbf{s}}_t, \mathbf{a}_t)$ with EMA target networks $\bar{Q}_{\theta_1}, \bar{Q}_{\theta_2}$ ($\tau_{\text{ema}} = 0.005$).
   * **Policy Prior ($\pi_\theta$):** Squashed Gaussian actor `MLP(400 -> 256 -> 256 -> 8)` outputting $(\boldsymbol{\mu}_a, \log\boldsymbol{\sigma}_a) \in \mathbb{R}^{4 \times 2}$. Used both as a warm-start proposal distribution for MPPI planning and for fast policy evaluation.

---

## 7. Loss Functions & Joint Training Objective

Sample a batch of short trajectory segments of length $H = 12$ steps from the Replay Buffer: $\{\mathbf{o}_{t:t+H}, \mathbf{a}_{t:t+H}, r_{t:t+H}, d_{t:t+H}, y_{\text{fault}, t:t+H}\}$ with batch size $B = 256$.

Unroll the Recurrent World Model across $k = 0, \dots, H$:
1. **Latent Consistency & KL Balancing Loss ($\mathcal{L}_{\text{dyn}}$):**
   Use DreamerV3-style KL balancing ($\alpha_{\text{prior}} = 0.8, \alpha_{\text{post}} = 0.2$) with free bits $\beta_{\text{free}} = 1.0$:
   $$\mathcal{L}_{\text{KL}, k} = \alpha_{\text{prior}} \max\left(\beta_{\text{free}}, D_{\text{KL}}(\text{sg}(q_\phi) \| p_\theta)\right) + \alpha_{\text{post}} \max\left(\beta_{\text{free}}, D_{\text{KL}}(q_\phi \| \text{sg}(p_\theta))\right)$$
   Plus cosine similarity consistency between multi-step open-loop imagined state $\hat{\mathbf{s}}_{t+k}$ and closed-loop encoded target $\text{sg}(\mathbf{s}_{t+k})$.

2. **Reward Prediction Loss ($\mathcal{L}_{\text{rew}}$):**
   $$\mathcal{L}_{\text{rew}} = \sum_{k=0}^{H-1} \lambda^k \text{SmoothL1Loss}\left(\hat{r}_{t+k}, r_{t+k}\right), \quad \lambda = 0.9$$

3. **Temporal Difference Q-Value Loss ($\mathcal{L}_Q$):**
   With TD target $y_{t+k} = r_{t+k} + \gamma (1 - d_{t+k}) \min_{j=1,2} \bar{Q}_{\theta_j}(\tilde{\mathbf{s}}_{t+k+1}, \pi_\theta(\tilde{\mathbf{s}}_{t+k+1}))$:
   $$\mathcal{L}_Q = \sum_{k=0}^{H-1} \lambda^k \sum_{j=1,2} \text{SmoothL1Loss}\left(Q_{\theta_j}(\tilde{\mathbf{s}}_{t+k}, \mathbf{a}_{t+k}), \text{sg}(y_{t+k})\right)$$

4. **FDIR Fault Isolation Loss ($\mathcal{L}_{\text{fdir}}$):**
   $$\mathcal{L}_{\text{fdir}} = \sum_{k=0}^{H-1} \text{CrossEntropy}\left(\hat{\mathbf{y}}_{\text{fault}, t+k}, y_{\text{fault}, t+k}\right)$$

5. **Total World Model Loss:**
   $$\mathcal{L}_{\text{WM}} = c_{\text{dyn}} \mathcal{L}_{\text{dyn}} + c_{\text{rew}} \mathcal{L}_{\text{rew}} + c_Q \mathcal{L}_Q + c_{\text{fdir}} \mathcal{L}_{\text{fdir}}$$
   *(Default weights: $c_{\text{dyn}} = 1.0, c_{\text{rew}} = 1.0, c_Q = 0.5, c_{\text{fdir}} = 0.5$).*

6. **Policy Prior Max-Q + Entropy Loss ($\mathcal{L}_\pi$):**
   Updated separately by freezing $\theta$:
   $$\mathcal{L}_\pi = \sum_{k=0}^{H-1} \lambda^k \left[ -\min_{j=1,2} Q_{\theta_j}\left(\text{sg}(\tilde{\mathbf{s}}_{t+k}), \tilde{\mathbf{a}}_{t+k}\right) + \beta_{\mathcal{H}} \log \pi_\theta\left(\tilde{\mathbf{a}}_{t+k} \mid \text{sg}(\tilde{\mathbf{s}}_{t+k})\right) \right]$$

---

## 8. Online Latent Trajectory Planning: Batched MPPI (`planners/mppi_planner.py`)

During flight (and optionally during environment data collection), the agent does not blindly output $\pi_\theta(\tilde{\mathbf{s}}_t)$. It plans over horizon $H_{\text{plan}} = 8$ steps using **Model Predictive Path Integral (MPPI)** inside the learned World Model:

1. **Warm-Start Action Proposals:**
   * Sample $N_\pi = 24$ action trajectories of length $H_{\text{plan}}$ by rolling out the learned Policy Prior $\pi_\theta$ in the Prior World Model $p_\theta$.
   * Sample $N_{\text{gauss}} = 232$ action trajectories from a time-dependent Gaussian $\mathcal{N}(\boldsymbol{\mu}_{0:H_{\text{plan}}}^{(i)}, (\boldsymbol{\sigma}_{0:H_{\text{plan}}}^{(i)})^2)$ initialized from the shifted mean of the previous timestep's plan.
   * Total candidates evaluated in parallel on GPU: $N_{\text{total}} = 256$ per environment.

2. **Latent Imagination Rollout & Scoring ($J(\tau)$):**
   Unroll all $N_{\text{total}}$ candidate sequences through the Prior World Model $p_\theta(\hat{\mathbf{z}}_{t+k+1} \mid \hat{\mathbf{h}}_{t+k+1})$ and fault context $\hat{\mathbf{c}}_t$ (kept constant or updated across the horizon):
   $$J(\mathbf{a}_{t:t+H_{\text{plan}}}) = \sum_{k=0}^{H_{\text{plan}}-1} \gamma^k \hat{r}(\hat{\mathbf{s}}_{t+k}, \mathbf{a}_{t+k}) + \gamma^{H_{\text{plan}}} \min_{j=1,2} Q_{\theta_j}\left(\hat{\mathbf{s}}_{t+H_{\text{plan}}}, \pi_\theta(\hat{\mathbf{s}}_{t+H_{\text{plan}}})\right)$$

3. **Elite Update ($K = 32$ Elites, $I = 4$ Iterations):**
   Select top $K = 32$ trajectories with highest return $J_m$. Compute importance weights:
   $$w_m = \frac{\exp\left(\tau_{\text{mppi}} (J_m - \max_n J_n)\right)}{\sum_{l=1}^{K} \exp\left(\tau_{\text{mppi}} (J_l - \max_n J_n)\right)}, \quad \tau_{\text{mppi}} = 10.0$$
   Update $\boldsymbol{\mu}_{0:H_{\text{plan}}}$ and $\boldsymbol{\sigma}_{0:H_{\text{plan}}}$ using weighted elite mean/std, repeat for $I = 4$ iterations, and execute the first action $\mathbf{a}_t^* = \boldsymbol{\mu}_0$.

---

## 9. Laptop Hardware Optimization Checklist (RTX 4060 8GB VRAM)

To guarantee high training throughput ($> 25,000$ environment steps/sec during rollout, $< 45$ minutes total wall-clock training time) without Out-Of-Memory crashes:

| Hyperparameter / Setting | Value | Hardware Justification |
| :--- | :--- | :--- |
| `num_envs` (Training Rollout) | `256` | Fits comfortably in L2 cache / CUDA cores on RTX 4060 |
| `num_envs` (MPPI Evaluation) | `16` | $16 \times 256$ MPPI particles = $4,096$ parallel latent rollouts ($\approx 1.1\text{ GB}$ VRAM) |
| `replay_buffer_capacity` | `200,000` | Stored in `float32` on GPU or pinned CPU RAM ($\approx 180\text{ MB}$ total memory) |
| `batch_size` | `256` | With horizon $H=12$, processes $3,072$ transitions per gradient update |
| `precision` | `torch.bfloat16` | Native Ada Lovelace Tensor Core acceleration for MLPs; keep RK4 physics in `float32` |
| `gradient_clip_norm` | `10.0` | Prevents exploding gradients from recurrent horizon backpropagation |
| `learning_rate` | `3e-4` | AdamW optimizer with `weight_decay=1e-4` |

---

## 10. Step-by-Step Implementation Execution Plan (For AI Coding Assistant)

When building this repository, the AI coding assistant MUST execute the phases in strict sequential order and run verification tests after every phase:

### Phase 1: Vectorized 6-DOF Physics Engine & Fault Injector
1. Create `configs/default_config.py` with all physical constants and hyperparameters.
2. Implement `envs/rocket_physics.py` containing batched PyTorch functions `quat_to_rotmat(q)`, `quat_derivative(q, omega)`, `compute_derivatives(state, action, fault_params)`, and `rk4_step(state, action, fault_params, dt)`.
3. Implement `envs/fault_injector.py` supporting the 5 fault modes (`Nominal`, `LOE`, `Gimbal Bias`, `CoM Shift`, `IMU/Wind`).
4. Implement `envs/vec_rocket_env.py` with `.reset()` and `.step(actions)` operating purely on `(B, ...)` CUDA tensors, including auto-reset of terminated environments.
5. **Verification Gate 1:** Run `tests/test_physics_rk4.py` to verify that:
   * With zero thrust in vacuum, vertical acceleration equals $-9.80665\text{ m/s}^2$.
   * Quaternion norm $\|\mathbf{q}\|_2$ stays within $1.0 \pm 10^{-6}$ over 500 steps.
   * Stepping $B=512$ parallel environments achieves $> 50,000\text{ SPS}$ on CUDA.

### Phase 2: Classical Cascaded PID Baseline (`planners/pid_baseline.py`)
1. Implement a vectorized cascaded Attitude + Velocity PID flight controller that computes desired pitch/yaw angles from lateral position/velocity error and maps attitude error to TVC gimbal angles $[\delta_{y,\text{cmd}}, \delta_{z,\text{cmd}}]$ and vertical suicide-burn throttle $u_{\text{throttle}}$.
2. **Verification Gate 2:** Verify that the PID controller achieves $> 85\%$ landing success under `Fault 0 (Nominal)`, but drops below $30\%$ success when `Fault 1 (35% LOE)` or `Fault 2 (+4 deg Gimbal Bias)` is injected. This establishes the baseline gap for our World Model.

### Phase 3: RSSM-FDIR World Model & Replay Buffer
1. Implement `models/networks.py`, `models/world_model.py`, and `models/fdir_head.py`.
2. Implement `training/replay_buffer.py` capable of sampling contiguous trajectory chunks of length $H+1 = 13$ without crossing episode termination boundaries.
3. **Verification Gate 3:** Run `tests/test_world_model_shapes.py` with dummy batch $(B=32, H=12)$ and assert that all losses ($\mathcal{L}_{\text{dyn}}, \mathcal{L}_{\text{rew}}, \mathcal{L}_Q, \mathcal{L}_{\text{fdir}}, \mathcal{L}_\pi$) compute finite gradients and peak VRAM is $< 3.5\text{ GB}$.

### Phase 4: MPPI Latent Planner & Training Pipeline
1. Implement `planners/mppi_planner.py` to roll out $N=256$ candidate action sequences over $H_{\text{plan}}=8$ steps inside the learned World Model.
2. Implement `training/train_world_model.py` with Curriculum Learning:
   * **Steps $0 \to 150,000$:** Warm-up on $80\%$ nominal flights + $20\%$ mild faults using Policy Prior $\pi_\theta$ + exploration noise.
   * **Steps $150,000 \to 500,000$:** Full fault distribution ($65\%$ fault probability across all 4 failure modes) with MPPI planning enabled.
3. Log metrics every 2,500 steps: `episode_return`, `landing_success_rate`, `fdir_classification_accuracy`, `kl_surprise_ratio`, and `sps`.

### Phase 5: Evaluation Suite & Portfolio Visualization (`evaluation/`)
1. Implement `evaluation/evaluate_fdir.py` to generate a Markdown/CSV benchmark comparing:
   * **Cascaded PID Baseline**
   * **Model-Free Actor-Critic ($\pi_\theta$ without MPPI or FDIR context)**
   * **AeroWM-FDIR (Full World Model + KL Anomaly Detection + MPPI)**
2. Implement `evaluation/visualize_trajectory.py` to save a high-resolution 4-panel figure (`results/fdir_recovery_demo.png`):
   * **Panel 1:** 3D Rocket Descent Trajectory showing nominal path vs. fault injection point vs. corrected landing trajectory.
   * **Panel 2:** Real-time KL-Divergence Surprise Score $S_t$ showing the immediate spike at $t = t_{\text{fault}}$ crossing the anomaly threshold $\tau_{\text{alarm}}$.
   * **Panel 3:** FDIR Fault Classifier softmax probabilities over time (instantly isolating e.g. "TVC Gimbal Bias" within 3 timesteps).
   * **Panel 4:** TVC Gimbal Deflection & Engine Throttle compensation curves.