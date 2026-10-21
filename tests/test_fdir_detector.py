"""Task 4.2: KL surprise detector, free-bits clipping and fault isolation."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from configs.schema import WorldModelConfig
from models.fdir_head import (
    FDIRModule,
    SurpriseDetector,
    detection_latency,
    fault_classification_accuracy,
)
from models.networks import gaussian_kl


def test_surprise_is_non_negative_and_zero_when_identical():
    torch.manual_seed(0)
    mu = torch.randn(8, 16)
    log_std = torch.randn(8, 16) * 0.2
    kl = FDIRModule.surprise(mu, log_std, mu, log_std)
    assert torch.all(kl >= -1e-6)
    assert torch.allclose(kl, torch.zeros(8), atol=1e-6)


def test_fault_head_shapes():
    cfg = WorldModelConfig(latent_dim=64, gru_dim=64, hidden_dim=64, simnorm_groups=8, simnorm_v=8)
    module = FDIRModule(cfg)
    h = torch.randn(4, cfg.gru_dim)
    z = torch.randn(4, cfg.latent_dim)
    mu_q, mu_p = torch.randn(4, cfg.latent_dim), torch.randn(4, cfg.latent_dim)
    ls = torch.zeros(4, cfg.latent_dim)
    innovation = torch.randn(4, 18)
    a_prev = torch.zeros(4, 3)
    signals = module(h, z, mu_q, ls, mu_p, ls, innovation, a_prev)
    logits, context, surprise = signals.logits, signals.context, signals.surprise
    assert logits.shape == (4, 5)
    assert context.shape == (4, cfg.fault_context_dim)
    assert surprise.shape == (4,)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(context.abs() <= 1.0)


def test_free_bits_clips_gradient_below_threshold():
    mu_p = torch.zeros(1, 4)
    log_std_p = torch.zeros(1, 4)
    mu_q = torch.zeros(1, 4, requires_grad=True)
    log_std_q = torch.zeros(1, 4)
    free_bits = 1.0

    kl = gaussian_kl(mu_q, log_std_q, mu_p, log_std_p)
    loss = torch.clamp(kl, min=free_bits).mean()
    loss.backward()
    assert torch.allclose(mu_q.grad, torch.zeros(1, 4))

    mu_q.grad = None
    mu_q2 = torch.full((1, 4), 3.0, requires_grad=True)
    kl2 = gaussian_kl(mu_q2, log_std_q, mu_p, log_std_p)
    loss2 = torch.clamp(kl2, min=free_bits).mean()
    loss2.backward()
    assert mu_q2.grad.abs().sum() > 0


def test_detector_calibration_and_alarm():
    torch.manual_seed(0)
    det = SurpriseDetector(alpha=0.05, sigmas=4.0, warmup=5)
    nominal = torch.full((64,), 1.0) + torch.randn(64) * 0.05
    det.calibrate(nominal)
    alarm, threshold = det.update(nominal)
    assert threshold.item() > 1.0
    assert alarm.float().mean() < 0.05

    spike = torch.full((64,), 50.0)
    alarm, threshold = det.update(spike)
    assert bool(alarm.all())


def test_detection_latency():
    surprise = torch.tensor([0.1, 0.2, 0.3, 9.0, 1.0])
    assert detection_latency(surprise, onset=0, threshold=1.0) == 3
    assert detection_latency(surprise, onset=3, threshold=1.0) == 0
    assert detection_latency(surprise, onset=4, threshold=1.0) is None


def test_classification_accuracy():
    logits = torch.tensor([[2.0, 0, 0, 0, 0], [0, 3.0, 0, 0, 0], [0, 0, 2.0, 0, 0]])
    labels = torch.tensor([0, 0, 2])
    assert fault_classification_accuracy(logits, labels).item() == pytest.approx(2 / 3)


def test_fault_head_gradients():
    cfg = WorldModelConfig(latent_dim=32, gru_dim=32, hidden_dim=32, simnorm_groups=8, simnorm_v=4)
    module = FDIRModule(cfg)
    h = torch.randn(4, 32)
    z = torch.randn(4, 32)
    mu_q = torch.randn(4, 32, requires_grad=True)
    mu_p = torch.randn(4, 32)
    ls = torch.zeros(4, 32)
    innovation = torch.randn(4, 18)
    a_prev = torch.zeros(4, 3)
    signals = module(h, z, mu_q, ls, mu_p, ls, innovation, a_prev)
    logits, context, surprise = signals.logits, signals.context, signals.surprise
    loss = F.cross_entropy(logits, torch.tensor([0, 1, 2, 3])) + context.mean() + surprise.mean() * 1e-3
    loss.backward()
    assert mu_q.grad is not None and torch.isfinite(mu_q.grad).all()
