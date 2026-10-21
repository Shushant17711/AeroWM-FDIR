r"""FDIR: KL + observation-innovation surprise detector and fault isolation head.

The anomaly signal combines two physically meaningful quantities:

* the latent KL divergence ``D_KL(q(z_t|h_t, o_t) || p(z_t|h_t))``, and
* the *observation innovation* ``o_t - \hat{o}_t`` where ``\hat{o}_t`` is a
  one-step-ahead prediction produced from ``(h_t, prior latent)`` only.

The innovation is normalised per channel by a running nominal standard
deviation, so the score is comparable across flight phases.  The combined score
is tracked by an EMA detector calibrated on nominal flight and alarms above
``mean + k * std``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from configs.schema import RocketPhysicsConfig, WorldModelConfig
from models.fdir_detector import (
    RESIDUAL_DIM,
    FaultParameterHead,
    ResidualExtractor,
    isolation_prior,
)
from models.networks import NormMLP, gaussian_kl


class InnovationNormalizer(nn.Module):
    """Per-channel running mean/std of the observation innovation.

    Only nominal transitions should be folded into these statistics; the
    trainer passes a mask.  Normalising by the nominal innovation scale makes
    ``mean((innov / sigma)^2)`` approximately 1 during healthy flight and much
    larger when the sensors stop matching the prior prediction.
    """

    def __init__(self, channels: Tuple[int, ...], alpha: float = 0.01, eps: float = 1e-4) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.alpha = alpha
        self.eps = eps
        n = len(self.channels)
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def select(self, innovation: torch.Tensor) -> torch.Tensor:
        return innovation[..., list(self.channels)]

    @torch.no_grad()
    def update(self, innovation: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        x = self.select(innovation).detach().float().reshape(-1, len(self.channels))
        if mask is not None:
            flat_mask = mask.detach().bool().reshape(-1)
            x = x[flat_mask]
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        if int(self.count) == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var.clamp_min(self.eps))
        else:
            delta = batch_mean - self.mean
            self.mean.add_(self.alpha * delta)
            self.var.mul_(1.0 - self.alpha).add_(self.alpha * (batch_var + delta * delta))
            self.var.clamp_min_(self.eps)
        self.count += 1

    def normalize(self, innovation: torch.Tensor) -> torch.Tensor:
        """Chi-square-like innovation score (mean ~1 on nominal flight)."""
        x = self.select(innovation).float()
        return (x / torch.sqrt(self.var + self.eps)).pow(2).mean(dim=-1)

    def reset(self) -> None:
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.zero_()


class SurpriseDetector(nn.Module):
    """Exponentially-weighted threshold calibration for the anomaly score."""

    def __init__(self, alpha: float = 0.01, sigmas: float = 4.0, warmup: int = 20) -> None:
        super().__init__()
        self.alpha = alpha
        self.sigmas = sigmas
        self.warmup = warmup
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_surprise", torch.zeros(()))

    @property
    def threshold(self) -> torch.Tensor:
        return self.mean + self.sigmas * torch.sqrt(self.var.clamp_min(1e-8))

    @torch.no_grad()
    def update(self, surprise: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fold nominal samples into the statistics and return batch alarms."""
        value = surprise.detach().float()
        if mask is not None:
            selected = value[mask.detach().bool()]
            if selected.numel() == 0:
                return value > self.threshold, self.threshold
            value_fit = selected
        else:
            value_fit = value
        batch_mean = value_fit.mean()
        delta = batch_mean - self.mean
        self.mean.add_(self.alpha * delta)
        self.var.mul_(1.0 - self.alpha).add_(self.alpha * delta * delta)
        self.count += 1
        self.last_surprise.copy_(batch_mean)
        return surprise > self.threshold, self.threshold

    @torch.no_grad()
    def calibrate(self, surprise: torch.Tensor, steps: int = 100) -> None:
        """Calibrate the alarm threshold on nominal surprise statistics."""
        values = surprise.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        # Robust (median/MAD) calibration: a handful of early-episode
        # transients would otherwise inflate a mean/variance fit and delay
        # every real detection by tens of steps.
        med = values.median()
        mad = (values - med).abs().median().clamp_min(1e-6)
        sigma = 1.4826 * mad
        self.mean.copy_(med)
        self.var.copy_((sigma**2).clamp_min(1e-8))
        self.count.fill_(self.warmup)

    def reset(self) -> None:
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.zero_()


@dataclass
class FDIRSignals:
    logits: torch.Tensor
    context: torch.Tensor
    surprise: torch.Tensor
    kl: torch.Tensor
    innovation: torch.Tensor
    # Exact one-step physics residuals and the estimated fault parameters.
    residual_score: Optional[torch.Tensor] = None
    fault_params: Optional[torch.Tensor] = None
    frozen: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    head_inputs: Optional[torch.Tensor] = None
    bias_estimate: Optional[torch.Tensor] = None
    com_estimate: Optional[torch.Tensor] = None


class FDIRModule(nn.Module):
    """Latent KL + innovation + exact residuals -> surprise, class, context."""

    def reset_state(self) -> None:
        """Clear the latched evidence (call on every episode boundary)."""
        self._evidence_peak = None

    def __init__(
        self,
        cfg: WorldModelConfig,
        physics: Optional[RocketPhysicsConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.detector = SurpriseDetector()
        self.normalizer = InnovationNormalizer(cfg.innovation_channels)
        # Exact actuator / propulsion / rotational residuals (Gate 3).
        self.residuals = ResidualExtractor(physics)
        in_dim = (
            cfg.gru_dim
            + cfg.latent_dim
            + cfg.latent_dim
            + len(cfg.innovation_channels)
            + cfg.action_dim
            + RESIDUAL_DIM
            + 2  # frozen-actuator indicators
        )
        self.head = NormMLP(
            in_dim,
            cfg.hidden_dim,
            cfg.num_fault_classes + cfg.fault_context_dim,
            num_layers=2,
        )
        self.param_head = FaultParameterHead(in_dim, cfg.hidden_dim)
        self._evidence_peak: Optional[torch.Tensor] = None
        # Start the learned classification head at zero so the physics prior is
        # the initial decision rule and training only refines it.
        last = None
        for module in self.head.modules():
            if isinstance(module, torch.nn.Linear):
                last = module
        if last is not None:
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)

    def residual_features(
        self,
        obs_prev: Optional[torch.Tensor],
        obs: Optional[torch.Tensor],
        a_prev: torch.Tensor,
    ) -> tuple:
        """Return ``(smoothed features, frozen indicator, raw features)``."""
        if obs_prev is None or obs is None:
            batch = a_prev.shape[:-1]
            zeros = torch.zeros(*batch, RESIDUAL_DIM, device=a_prev.device, dtype=a_prev.dtype)
            frozen = torch.zeros(*batch, 2, device=a_prev.device, dtype=a_prev.dtype)
            return zeros, frozen, None, None, None, None
        features, out = self.residuals(obs_prev, obs, a_prev)
        return features, out.frozen, out.features, out.bias, out.frozen_instant, out.com_estimate

    @staticmethod
    def kl_divergence(
        mu_post: torch.Tensor,
        log_std_post: torch.Tensor,
        mu_prior: torch.Tensor,
        log_std_prior: torch.Tensor,
    ) -> torch.Tensor:
        return gaussian_kl(mu_post, log_std_post, mu_prior, log_std_prior)

    # Kept for backwards compatibility with earlier call sites/tests.
    @staticmethod
    def surprise(
        mu_post: torch.Tensor,
        log_std_post: torch.Tensor,
        mu_prior: torch.Tensor,
        log_std_prior: torch.Tensor,
    ) -> torch.Tensor:
        return gaussian_kl(mu_post, log_std_post, mu_prior, log_std_prior)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        mu_post: torch.Tensor,
        log_std_post: torch.Tensor,
        mu_prior: torch.Tensor,
        log_std_prior: torch.Tensor,
        obs_innovation: torch.Tensor,
        a_prev: torch.Tensor,
        residual_features: Optional[torch.Tensor] = None,
        frozen: Optional[torch.Tensor] = None,
        raw_residuals: Optional[torch.Tensor] = None,
        bias_estimate: Optional[torch.Tensor] = None,
        com_estimate: Optional[torch.Tensor] = None,
        frozen_instant_input: Optional[torch.Tensor] = None,
        com_input: Optional[torch.Tensor] = None,
    ) -> FDIRSignals:
        kl = self.kl_divergence(mu_post, log_std_post, mu_prior, log_std_prior)
        innovation_score = self.normalizer.normalize(obs_innovation)
        if residual_features is None:
            residual_features = torch.zeros(*kl.shape, RESIDUAL_DIM, device=kl.device, dtype=kl.dtype)
            frozen = torch.zeros(*kl.shape, 2, device=kl.device, dtype=kl.dtype)
        residual_score = self.residuals.surprise(residual_features)
        # Instantaneous frozen-servo evidence.  A stuck axis produces almost no
        # residual until the command has diverged from the frozen angle (tens
        # of steps), but "the servo did not move although the command asked it
        # to" is true on the very first sample.  Folding that into S_t gives a
        # 1-2 step detection even in near-vertical flight.
        frozen_instant = torch.zeros_like(residual_score)
        if frozen_instant_input is not None:
            frozen_instant = frozen_instant_input.max(dim=-1).values
        # CoM evidence is a *physical* quantity (metres of offset), not a
        # sigma score: the per-step rotational residual is far below the
        # noise floor, but half a second of averaging recovers the offset.
        com_score = torch.zeros_like(residual_score)
        if com_input is not None:
            com_score = (com_input.abs().amax(dim=-1) / self.cfg.com_detect_threshold).clamp(0.0, 10.0)
        surprise = (
            self.cfg.surprise_kl_weight * kl
            + self.cfg.innovation_weight * innovation_score
            + self.cfg.residual_weight * residual_score
            + self.cfg.frozen_weight * frozen_instant
            + self.cfg.com_weight * com_score
        )

        head_inputs = torch.cat(
            [
                h,
                z,
                mu_post - mu_prior,
                self.normalizer.select(obs_innovation),
                a_prev,
                residual_features,
                frozen,
            ],
            dim=-1,
        )
        out = self.head(head_inputs)
        # Physics-informed isolation prior: the residual signature is nearly
        # deterministic, so the head only has to refine it.
        # The prior consumes *normalised* residuals: in nominal flight each
        # channel is ~0 +/- 1, so a fault produces an unambiguous signature
        # regardless of the physical units of the channel.
        prior_input = self.residuals.normalize(residual_features).clamp(-30.0, 30.0)
        prior = isolation_prior(prior_input, frozen)
        if com_input is not None:
            # CoM class evidence is a *physical* offset in metres (the sigma
            # score cannot see it: the per-step rotational residual is far
            # below the nominal noise floor).
            com_ev = (com_input.abs().amax(dim=-1) / self.cfg.com_detect_threshold).clamp(0.0, 10.0)
            prior = prior + torch.nn.functional.one_hot(
                torch.full(com_ev.shape, 4, dtype=torch.long, device=com_ev.device),
                num_classes=self.cfg.num_fault_classes,
            ).to(prior.dtype) * (6.0 * com_ev).unsqueeze(-1)
        # Faults are latched: once a signature has been seen it should not be
        # forgotten when the controller masks it later in the episode.  In
        # eval mode the prior evidence is peak-held with a slow decay; during
        # training the instantaneous prior is used (sequences are independent).
        if not self.training:
            if self._evidence_peak is None or self._evidence_peak.shape != prior.shape:
                self._evidence_peak = prior.detach().clone()
            else:
                # Slow evidence average (~250 steps): a fault seen early in the
                # episode stays classified even after the controller masks it.
                self._evidence_peak = 0.996 * self._evidence_peak + 0.004 * prior.detach()
            prior = self._evidence_peak.to(prior.dtype)
        # Bound the learned head so a poorly-calibrated training run cannot
        # override the physics prior (which is the deterministic part).
        head_logits = 3.0 * torch.tanh(out[..., : self.cfg.num_fault_classes] / 3.0)
        logits = head_logits + self.cfg.isolation_prior_weight * prior
        context = torch.tanh(out[..., self.cfg.num_fault_classes :])
        params = self.param_head(head_inputs)
        return FDIRSignals(
            logits,
            context,
            surprise,
            kl,
            innovation_score,
            residual_score,
            params,
            frozen,
            residual_features,
            head_inputs,
            bias_estimate,
            com_estimate,
        )


def detection_latency(
    surprise: torch.Tensor,
    onset: int,
    threshold: float | torch.Tensor,
    horizon: Optional[int] = None,
) -> Optional[int]:
    """First step at/after ``onset`` whose surprise exceeds ``threshold``."""
    thr = float(threshold)
    end = surprise.shape[0] if horizon is None else min(surprise.shape[0], onset + horizon)
    for t in range(onset, end):
        if float(surprise[t]) > thr:
            return t - onset
    return None


def fault_classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == labels).float().mean()
