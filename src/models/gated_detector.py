"""Gated anomaly detector.

This module combines a conditional forecaster and a marginal density model with
a learned per-feature gate.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.forecasters import GaussianLSTMForecaster
from models.gate import UncertaintyGate


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus for positive y (approximately), with stability clamp."""
    return torch.log(torch.expm1(y).clamp_min(1e-6))


class GatedAnomalyDetector(nn.Module):
    """Conditional-marginal mixture detector with per-feature routing."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        marginal_expert: nn.Module,  # must implement surprise(x): (B, D)
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        detach_gate_input: bool = True,
        # Optional regularization to discourage always choosing marginal
        gate_l1: float = 0.0,
        gate_aux_weight: float = 0.0,
        gate_aux_margin: float = 0.0,
        gate_aux_temp: float = 1.0,
        gate_prior_weight: float = 0.0,
        gate_use_residual: bool = False,
        use_loss_standardization: bool = True,
        use_mixture_nll: bool = True,
        # Optional: include gaussian constant term (doesn't change training, but changes score scale)
        include_gaussian_const: bool = False,
        marginal_b_l2: float = 0.0,
    ) -> None:
        super().__init__()

        self.forecaster = GaussianLSTMForecaster(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=output_dim,
        )
        self.marginal_expert = marginal_expert

        num_signals = 2 if gate_use_residual else 1
        self.gate = UncertaintyGate(feature_dim=output_dim, num_signals=num_signals)

        self.detach_gate_input = bool(detach_gate_input)
        self.gate_l1 = float(gate_l1)
        self.gate_aux_weight = float(gate_aux_weight)
        self.gate_aux_margin = float(gate_aux_margin)
        self.gate_aux_temp = float(gate_aux_temp)
        self.gate_prior_weight = float(gate_prior_weight)
        self.gate_use_residual = bool(gate_use_residual)
        self.use_loss_standardization = bool(use_loss_standardization)
        self.use_mixture_nll = bool(use_mixture_nll)
        self.include_gaussian_const = bool(include_gaussian_const)
        self.marginal_b_l2 = float(marginal_b_l2)

        # Scale-matching: learn per-feature affine calibration for marginal loss
        # marg_cal = a * marg + b  (a starts at 1, b starts at 0)
        init = inv_softplus(torch.tensor(1.0))
        self.marginal_a = nn.Parameter(torch.full((output_dim,), float(init)))
        self.marginal_b = nn.Parameter(torch.zeros(output_dim))

        # Optional loss statistics (for standardizing losses in gate auxiliary objective)
        self.register_buffer("forecast_mean", torch.empty(0))
        self.register_buffer("forecast_std", torch.empty(0))
        self.register_buffer("marginal_mean", torch.empty(0))
        self.register_buffer("marginal_std", torch.empty(0))
        self.register_buffer("alpha_prior", torch.empty(0))

        self.has_loss_stats = False
        self.has_alpha_prior = False

        # Optional gate-input statistics (for standardizing gate inputs)
        self.register_buffer("logsig_mean", torch.empty(0))
        self.register_buffer("logsig_std", torch.empty(0))
        self.register_buffer("resid_mean", torch.empty(0))
        self.register_buffer("resid_std", torch.empty(0))

        self.has_gate_input_stats = False
        self.has_resid_stats = False

    def set_loss_stats(
        self,
        forecast_mean: torch.Tensor,
        forecast_std: torch.Tensor,
        marginal_mean: torch.Tensor,
        marginal_std: torch.Tensor,
    ) -> None:
        self.forecast_mean = forecast_mean
        self.forecast_std = forecast_std
        self.marginal_mean = marginal_mean
        self.marginal_std = marginal_std
        self.has_loss_stats = True

    def set_gate_prior(self, alpha_prior: torch.Tensor) -> None:
        self.alpha_prior = alpha_prior
        self.has_alpha_prior = True

    def set_gate_input_stats(
        self,
        logsig_mean: torch.Tensor,
        logsig_std: torch.Tensor,
        resid_mean: Optional[torch.Tensor] = None,
        resid_std: Optional[torch.Tensor] = None,
    ) -> None:
        self.logsig_mean = logsig_mean
        self.logsig_std = logsig_std
        self.has_gate_input_stats = True

        if resid_mean is not None and resid_std is not None:
            self.resid_mean = resid_mean
            self.resid_std = resid_std
            self.has_resid_stats = True

    @staticmethod
    def gaussian_nll_per_dim(
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        log_sigma: torch.Tensor,
        include_const: bool = False,
    ) -> torch.Tensor:
        """
        Returns per-feature Gaussian NLL without reduction: (B, D)

        NLL = 0.5*((x - mu)/sigma)^2 + log_sigma + 0.5*log(2pi) [optional]
        """
        var = sigma.pow(2)
        diff = x - mu
        nll = 0.5 * (diff.pow(2) / var) + log_sigma
        if include_const:
            nll = nll + 0.5 * math.log(2.0 * math.pi)
        return nll

    @staticmethod
    def mixture_nll(
        nll_f: torch.Tensor,
        nll_m: torch.Tensor,
        alpha: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Stable per-feature mixture NLL:
            nll_mix = -log( (1 - a) * p_f + a * p_m )

        Inputs and output are per-feature tensors with shape (B, D).
        """
        a = alpha.clamp(eps, 1.0 - eps)
        log_pf = -nll_f + torch.log1p(-a)
        log_pm = -nll_m + torch.log(a)
        log_mix = torch.logsumexp(torch.stack([log_pf, log_pm], dim=0), dim=0)
        return -log_mix

    def forward(self, x_hist: torch.Tensor, x_target: torch.Tensor) -> dict:
        """
        Args:
            x_hist:   (B, T, input_dim) past window (possibly includes mask concat already)
            x_target: (B, D) next-step ground truth in SAME normalization space as training

        Returns:
            dict of losses/scores/diagnostics. Shapes are preserved exactly.
        """
        mu, sigma, log_sigma = self.forecaster(x_hist)

        # Conditional per-feature surprise
        loss_forecast = self.gaussian_nll_per_dim(
            x=x_target,
            mu=mu,
            sigma=sigma,
            log_sigma=log_sigma,
            include_const=self.include_gaussian_const,
        )  # (B, D)

        # Marginal per-feature surprise (must return (B, D))
        loss_marginal = self.marginal_expert.surprise(x_target)  # (B, D)

        if self.use_mixture_nll:
            # Affine calibration breaks probability semantics; disabled when using mixture NLL.
            loss_marginal_cal = loss_marginal
        else:
            eff_a = F.softplus(self.marginal_a)
            # Calibrate marginal scale per feature to avoid scale-mismatch dominating the gate
            loss_marginal_cal = loss_marginal * eff_a + self.marginal_b

        # Gate input (anti-cheating))
        gate_log_sigma = log_sigma.detach() if self.detach_gate_input else log_sigma
        if self.has_gate_input_stats:
            # Standardize gate inputs using validation-set stats.
            gate_log_sigma = (gate_log_sigma - self.logsig_mean) / (self.logsig_std + 1e-6)

        if self.gate_use_residual:
            residual = (x_target - mu).abs()
            if self.detach_gate_input:
                residual = residual.detach()
            if self.has_resid_stats:
                residual = (residual - self.resid_mean) / (self.resid_std + 1e-6)
            gate_in = (gate_log_sigma, residual)
        else:
            gate_in = gate_log_sigma

        alpha = self.gate(gate_in)  # (B, D)

        # Mixture score (per-feature); default is probability-space mixture NLL.
        if self.use_mixture_nll:
            combined = self.mixture_nll(loss_forecast, loss_marginal_cal, alpha)
        else:
            combined = (1.0 - alpha) * loss_forecast + alpha * loss_marginal_cal  # (B, D)

        # Training objective
        loss = combined.mean()

        if self.gate_l1 > 0.0:
            # Discourage routing everything to marginal by default
            loss = loss + self.gate_l1 * alpha.mean()

        if self.marginal_b_l2 > 0.0:
            loss = loss + self.marginal_b_l2 * self.marginal_b.pow(2).mean()

        gate_aux_loss = loss.new_zeros(())
        if self.gate_aux_weight > 0.0:
            temp = max(self.gate_aux_temp, 1e-6)

            loss_forecast_gate = loss_forecast
            loss_marginal_gate = loss_marginal_cal
            if self.use_loss_standardization and self.has_loss_stats:
                loss_forecast_gate = (loss_forecast - self.forecast_mean) / self.forecast_std
                loss_marginal_gate = (loss_marginal_cal - self.marginal_mean) / self.marginal_std

            gap = loss_forecast_gate - loss_marginal_gate

            gap = gap.clamp(-10.0, 10.0)  # prevents gradient saturation

            alpha_target = torch.sigmoid((gap - self.gate_aux_margin) / temp)
            gate_aux_loss = F.binary_cross_entropy(alpha, alpha_target)
            loss = loss + self.gate_aux_weight * gate_aux_loss

        gate_prior_loss = loss.new_zeros(())
        if self.gate_prior_weight > 0.0 and self.has_alpha_prior:
            alpha_prior = self.alpha_prior
            if alpha_prior.ndim == 1:
                alpha_prior = alpha_prior.unsqueeze(0).expand_as(alpha)
            gate_prior_loss = F.binary_cross_entropy(alpha, alpha_prior)
            loss = loss + self.gate_prior_weight * gate_prior_loss

        return {
            "loss": loss,                            # scalar
            "score_per_feature": combined,           # (B, D)
            "score": combined.mean(dim=1),           # (B,) window score
            "alpha": alpha,                          # (B, D)
            "forecast_loss": loss_forecast,          # (B, D)
            "marginal_loss": loss_marginal,          # (B, D) raw
            "marginal_loss_cal": loss_marginal_cal,  # (B, D) calibrated
            "gate_aux_loss": gate_aux_loss,          # scalar
            "gate_prior_loss": gate_prior_loss,      # scalar
            "pred_mu": mu,                           # (B, D)
            "pred_sigma": sigma,                     # (B, D)
            "pred_log_sigma": log_sigma,             # (B, D)
        }
    