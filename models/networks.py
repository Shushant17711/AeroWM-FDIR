"""Building blocks for the latent world model (SimNorm, NormMLP, GRU cell)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SimNorm(nn.Module):
    """Simplicial normalization: softmax over groups of the latent vector.

    Prevents latent collapse/explosion during multi-step imagination without a
    pixel decoder (TD-MPC2).
    """

    def __init__(self, dim: int, groups: int = 32, temperature: float = 1.0) -> None:
        super().__init__()
        if dim % groups != 0:
            raise ValueError("dim must be divisible by groups")
        self.dim = dim
        self.groups = groups
        self.v = dim // groups
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        z = x.reshape(*shape[:-1], self.groups, self.v)
        z = torch.softmax(z / self.temperature, dim=-1)
        return z.reshape(*shape)


class NormMLP(nn.Module):
    """``Linear -> LayerNorm -> Mish`` stack with orthogonal init."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        activation: str = "mish",
        output_activation: bool = False,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        act_cls = nn.Mish if activation == "mish" else nn.SiLU
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        if output_activation:
            layers.append(act_cls())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUCell(nn.Module):
    """``nn.GRUCell`` wrapper with orthogonal initialization."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        for name, param in self.cell.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.cell(x, h)


def gaussian_kl(
    mu_q: torch.Tensor,
    log_std_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_std_p: torch.Tensor,
) -> torch.Tensor:
    """Analytic KL(q || p) between diagonal Gaussians, summed over dims."""
    var_q = torch.exp(2.0 * log_std_q)
    var_p = torch.exp(2.0 * log_std_p)
    kl = log_std_p - log_std_q + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5
    return kl.sum(dim=-1)


def reparameterize(mu: torch.Tensor, log_std: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    std = torch.exp(log_std)
    eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu + std * eps
