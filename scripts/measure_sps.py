"""Gate-1 throughput measurement: full vectorized environment steps/second."""
import time
import torch
from envs.rocket_vec_env import VectorizedRocketEnv

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = 1024
env = VectorizedRocketEnv(num_envs=batch, device=dev, seed=0)
obs, _ = env.reset(probability=0.0)
action = torch.zeros(batch, 3, device=dev)
for _ in range(20):
    env.step(action)
torch.cuda.synchronize()
steps = 300
start = time.perf_counter()
for _ in range(steps):
    env.step(action)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
sps = batch * steps / elapsed
print(f"full env throughput: {sps:,.0f} SPS at B={batch} ({elapsed:.2f} s for {steps} steps)")
assert sps >= 200_000, f"{sps:,.0f} SPS below the 200k gate"
print("Gate 1 (SPS >= 200k): PASS")
