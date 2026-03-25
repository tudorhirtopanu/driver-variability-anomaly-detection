"""Shared training/evaluation loops for the staged optimisation pipeline."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.forecasters import GaussianLSTMForecaster
from models.gated_detector import GatedAnomalyDetector

if TYPE_CHECKING:
    from models.marginal_flow import MarginalFlowBank


def gaussian_nll_mean(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    log_sigma: torch.Tensor,
    include_const: bool,
) -> torch.Tensor:
    nll = GatedAnomalyDetector.gaussian_nll_per_dim(
        x=x, mu=mu, sigma=sigma, log_sigma=log_sigma, include_const=include_const
    )
    return nll.mean()


def train_forecaster_epoch(
    forecaster: GaussianLSTMForecaster,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float,
    desc: str,
    include_const: bool,
) -> float:
    forecaster.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        mu, sigma, log_sigma = forecaster(x)
        loss = gaussian_nll_mean(y, mu, sigma, log_sigma, include_const)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(forecaster.parameters(), clip_grad)
        optimizer.step()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


@torch.no_grad()
def eval_forecaster_epoch(
    forecaster: GaussianLSTMForecaster,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    include_const: bool,
) -> float:
    forecaster.eval()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        mu, sigma, log_sigma = forecaster(x)
        loss = gaussian_nll_mean(y, mu, sigma, log_sigma, include_const)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


@torch.no_grad()
def eval_gated_epoch(
    model: GatedAnomalyDetector,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    total_f = 0.0
    total_m = 0.0
    total_a = 0.0
    total_g = 0.0
    total_p = 0.0
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)

        loss = out["loss"]
        # Keeping the individual components alongside the total loss makes it
        # easier to see whether changes are coming from the forecaster term, the
        # marginal term, the learned alpha, or the gate regularizers.
        f = out["forecast_loss"].mean()
        m = out["marginal_loss_cal"].mean()
        a = out["alpha"].mean()
        g = out["gate_aux_loss"].mean()
        p = out["gate_prior_loss"].mean()

        bs = x.size(0)
        total += loss.item() * bs
        total_f += f.item() * bs
        total_m += m.item() * bs
        total_a += a.item() * bs
        total_g += g.item() * bs
        total_p += p.item() * bs
        n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", alpha=f"{a.item():.2f}")

    return {
        "loss": total / max(1, n),
        "forecast": total_f / max(1, n),
        "marginal": total_m / max(1, n),
        "alpha": total_a / max(1, n),
        "gate_aux": total_g / max(1, n),
        "gate_prior": total_p / max(1, n),
    }


def train_gated_epoch(
    model: GatedAnomalyDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float,
    desc: str,
    experts_eval: bool = False,
) -> Dict[str, float]:
    model.train()
    if experts_eval:
        # In the gate-only stage the experts are frozen, so they stay in eval
        # mode while only the gate parameters receive updates.
        model.forecaster.eval()
        model.marginal_expert.eval()

    total = 0.0
    total_f = 0.0
    total_m = 0.0
    total_a = 0.0
    total_g = 0.0
    total_p = 0.0
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, y)

        loss = out["loss"]
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        f = out["forecast_loss"].mean()
        m = out["marginal_loss_cal"].mean()
        a = out["alpha"].mean()
        g = out["gate_aux_loss"].mean()
        p = out["gate_prior_loss"].mean()

        bs = x.size(0)
        total += loss.item() * bs
        total_f += f.item() * bs
        total_m += m.item() * bs
        total_a += a.item() * bs
        total_g += g.item() * bs
        total_p += p.item() * bs
        n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", alpha=f"{a.item():.2f}")

    return {
        "loss": total / max(1, n),
        "forecast": total_f / max(1, n),
        "marginal": total_m / max(1, n),
        "alpha": total_a / max(1, n),
        "gate_aux": total_g / max(1, n),
        "gate_prior": total_p / max(1, n),
    }


def train_marginal_epoch(
    marginal: "MarginalFlowBank",
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float,
    desc: str,
) -> float:
    marginal.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for y in pbar:
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        surprise = marginal.surprise(y)
        loss = surprise.mean()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(marginal.parameters(), clip_grad)
        optimizer.step()

        bs = y.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


@torch.no_grad()
def eval_marginal_epoch(
    marginal: "MarginalFlowBank",
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> float:
    marginal.eval()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for y in pbar:
        y = y.to(device)
        loss = marginal.surprise(y).mean()
        bs = y.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


def freeze_module(mod: torch.nn.Module, freeze: bool = True) -> None:
    for p in mod.parameters():
        p.requires_grad = not freeze
