"""Statistics used to stabilise gate training and calibration."""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.forecasters import GaussianLSTMForecaster
from models.gated_detector import GatedAnomalyDetector

if TYPE_CHECKING:
    from models.marginal_flow import MarginalFlowBank


@torch.no_grad()
def compute_loss_stats(
    forecaster: GaussianLSTMForecaster,
    marginal: "MarginalFlowBank",
    loader: DataLoader,
    device: torch.device,
    include_const: bool,
    marginal_a: Optional[torch.Tensor] = None,
    marginal_b: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    forecaster.eval()
    marginal.eval()
    eff_a = None
    eff_b = None
    if marginal_a is not None and marginal_b is not None:
        eff_a = F.softplus(marginal_a)
        eff_b = marginal_b

    sum_f = None
    sumsq_f = None
    sum_m = None
    sumsq_m = None
    sum_gap = None
    sumsq_gap = None
    count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        mu, sigma, log_sigma = forecaster(x)
        loss_f = GatedAnomalyDetector.gaussian_nll_per_dim(
            x=y, mu=mu, sigma=sigma, log_sigma=log_sigma, include_const=include_const
        )
        loss_m = marginal.surprise(y)
        if eff_a is not None and eff_b is not None:
            loss_m_cal = loss_m * eff_a + eff_b
        else:
            loss_m_cal = loss_m
        gap = loss_f - loss_m_cal

        if sum_f is None:
            D = y.size(1)
            sum_f = torch.zeros(D, device=device)
            sumsq_f = torch.zeros(D, device=device)
            sum_m = torch.zeros(D, device=device)
            sumsq_m = torch.zeros(D, device=device)
            sum_gap = torch.zeros(D, device=device)
            sumsq_gap = torch.zeros(D, device=device)

        sum_f += loss_f.sum(dim=0)
        sumsq_f += (loss_f**2).sum(dim=0)
        sum_m += loss_m_cal.sum(dim=0)
        sumsq_m += (loss_m_cal**2).sum(dim=0)
        sum_gap += gap.sum(dim=0)
        sumsq_gap += (gap**2).sum(dim=0)
        count += y.size(0)

    if count == 0:
        raise ValueError("No samples available to compute loss stats.")

    mean_f = sum_f / count
    var_f = sumsq_f / count - mean_f.pow(2)
    std_f = torch.sqrt(var_f.clamp(min=1e-6))

    mean_m = sum_m / count
    var_m = sumsq_m / count - mean_m.pow(2)
    std_m = torch.sqrt(var_m.clamp(min=1e-6))

    mean_gap = sum_gap / count
    var_gap = sumsq_gap / count - mean_gap.pow(2)
    std_gap = torch.sqrt(var_gap.clamp(min=1e-6))

    return {
        "forecast_mean": mean_f,
        "forecast_std": std_f,
        "marginal_mean": mean_m,
        "marginal_std": std_m,
        # The gap is used to build an alpha prior: positive values mean the
        # marginal expert tends to look better than the forecaster on average.
        "gap_mean": mean_gap,
        "gap_std": std_gap,
    }


def compute_alpha_prior(
    gap_mean: torch.Tensor,
    gap_std: torch.Tensor,
    margin: float,
    temp: float,
) -> torch.Tensor:
    denom = gap_std.clamp(min=1e-6)
    gap_z = gap_mean / denom
    temp = max(float(temp), 1e-6)
    # This maps the standardized forecast-vs-marginal gap into a soft prior in
    # [0, 1] that can be used to bias the gate before training.
    return torch.sigmoid((gap_z - float(margin)) / temp)


@torch.no_grad()
def compute_gate_input_stats(
    forecaster: GaussianLSTMForecaster,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    forecaster.eval()
    sum_logsig = None
    sumsq_logsig = None
    sum_resid = None
    sumsq_resid = None
    count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        mu, _, log_sigma = forecaster(x)
        resid = (y - mu).abs()

        if sum_logsig is None:
            D = y.size(1)
            sum_logsig = torch.zeros(D, device=device)
            sumsq_logsig = torch.zeros(D, device=device)
            sum_resid = torch.zeros(D, device=device)
            sumsq_resid = torch.zeros(D, device=device)

        sum_logsig += log_sigma.sum(dim=0)
        sumsq_logsig += (log_sigma**2).sum(dim=0)
        sum_resid += resid.sum(dim=0)
        sumsq_resid += (resid**2).sum(dim=0)
        count += y.size(0)

    if count == 0:
        raise ValueError("No samples available to compute gate input stats.")

    mean_logsig = sum_logsig / count
    var_logsig = sumsq_logsig / count - mean_logsig.pow(2)
    std_logsig = torch.sqrt(var_logsig.clamp(min=1e-6))

    mean_resid = sum_resid / count
    var_resid = sumsq_resid / count - mean_resid.pow(2)
    std_resid = torch.sqrt(var_resid.clamp(min=1e-6))

    return {
        "logsig_mean": mean_logsig,
        "logsig_std": std_logsig,
        "resid_mean": mean_resid,
        "resid_std": std_resid,
    }
