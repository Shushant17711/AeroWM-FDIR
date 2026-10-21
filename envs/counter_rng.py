"""Counter-based pseudo-random numbers (reproducible *per episode*).

A stateful generator (``torch.Generator``) owns a single, strictly ordered
stream of random numbers.  That is exactly what an environment must **not**
use for anything that has to be reproducible per episode: the observation
noise of episode ``k`` would then depend on how many observations the
previous episodes produced, so a benchmark result would change when the
controllers are evaluated in a different order or when one controller is
skipped.

The environment therefore draws its *episode-local* randomness from a
counter-based hash of ``(episode seed, counter)``: the value is a pure
function of the seed and the counter, so it can be recomputed at any time,
in any order, on any device.

The mixer is the SplitMix64 finaliser (three xor-shift/multiply rounds),
which is cheap, vectorises over the batch and produces a well-scrambled
64-bit output per counter.
"""

from __future__ import annotations

import math

import torch

# SplitMix64 constants.  They are written as *signed* int64 values because
# torch has no unsigned integer type; the arithmetic wraps around, which is
# exactly the intended behaviour of the original 64-bit mixer.
_GOLDEN = -7046029254386353131  # 0x9E3779B97F4A7C15
_MIX_A = -2860488556131138987  # 0xBF58476D1CE4E5B9
_MIX_B = -7733592293110705685  # 0x94D049BB133111EB
_SIGN_MASK = 0x7FFFFFFFFFFFFFFF
_SHIFT_30 = 30
_SHIFT_27 = 27
_SHIFT_31 = 31
_SHIFT_11 = 11
_TWO_POW_52 = float(1 << 52)


def splitmix64(x: torch.Tensor) -> torch.Tensor:
    """SplitMix64 finaliser applied element-wise to an int64 tensor."""
    x = x ^ (x >> _SHIFT_30)
    x = x * _MIX_A
    x = x ^ (x >> _SHIFT_27)
    x = x * _MIX_B
    x = x ^ (x >> _SHIFT_31)
    return x


def derive_seed(base: int, index: torch.Tensor, episode: torch.Tensor) -> torch.Tensor:
    """Per-episode seed from a base seed, an environment index and an episode counter.

    The result is a pure function of its arguments, so episode ``k`` of
    environment ``i`` always gets the same seed -- independently of how many
    episodes were run before it.
    """
    base_t = torch.tensor([int(base)], dtype=torch.int64, device=index.device)
    x = base_t + index.to(torch.int64) * _GOLDEN + episode.to(torch.int64) * _MIX_A
    return splitmix64(x)


def hash_uniform(seed: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    """Uniform ``[0, 1)`` from an int64 ``seed`` and an int64 ``counter``."""
    x = splitmix64(seed.to(torch.int64) * _GOLDEN + counter.to(torch.int64))
    bits = torch.bitwise_and(x, _SIGN_MASK)  # 63 usable bits (int64 is signed)
    return (bits >> _SHIFT_11).to(torch.float64) * (1.0 / _TWO_POW_52)


def hash_normal(seed: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    """Standard normal from the same hash (Box-Muller on two counters)."""
    u1 = hash_uniform(seed, counter).clamp_min(1e-12)
    u2 = hash_uniform(seed, counter + 0x5DEECE66D)
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
