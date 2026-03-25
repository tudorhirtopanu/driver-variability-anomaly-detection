from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.forecasters import GaussianLSTMForecaster
from models.gated_detector import GatedAnomalyDetector


class MSELSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.mu_head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.mu_head(last)


def gaussian_nll_per_dim(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    log_sigma: torch.Tensor,
    include_const: bool,
) -> torch.Tensor:
    return GatedAnomalyDetector.gaussian_nll_per_dim(
        x=x,
        mu=mu,
        sigma=sigma,
        log_sigma=log_sigma,
        include_const=include_const,
    )


def train_gaussian_epoch(
    model: GaussianLSTMForecaster,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float,
    include_const: bool,
    desc: str,
) -> float:
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        mu, sigma, log_sigma = model(x)
        loss = gaussian_nll_per_dim(y, mu, sigma, log_sigma, include_const).mean()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


@torch.no_grad()
def eval_gaussian_epoch(
    model: GaussianLSTMForecaster,
    loader: DataLoader,
    device: torch.device,
    include_const: bool,
    desc: str,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        mu, sigma, log_sigma = model(x)
        loss = gaussian_nll_per_dim(y, mu, sigma, log_sigma, include_const).mean()
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


def train_mse_epoch(
    model: MSELSTMForecaster,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float,
    desc: str,
) -> float:
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        mu = model(x)
        loss = F.mse_loss(mu, y)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


@torch.no_grad()
def eval_mse_epoch(
    model: MSELSTMForecaster,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        mu = model(x)
        loss = F.mse_loss(mu, y)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, n)


@torch.no_grad()
def collect_scores_gated(
    model: GatedAnomalyDetector,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    scores = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)
        scores.append(out["score"].detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


@torch.no_grad()
def collect_scores_gaussian(
    model: GaussianLSTMForecaster,
    loader: DataLoader,
    device: torch.device,
    include_const: bool,
) -> np.ndarray:
    model.eval()
    scores = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        mu, sigma, log_sigma = model(x)
        nll = gaussian_nll_per_dim(y, mu, sigma, log_sigma, include_const)
        scores.append(nll.mean(dim=1).detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


@torch.no_grad()
def collect_scores_mse(
    model: MSELSTMForecaster,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    scores = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        mu = model(x)
        mse = (mu - y).pow(2).mean(dim=1)
        scores.append(mse.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def compute_fpr(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    quantiles: List[float],
) -> Dict[str, List[float]]:
    thresholds = [float(np.quantile(val_scores, q)) for q in quantiles]
    fpr_val = [float((val_scores > t).mean()) for t in thresholds]
    fpr_test = [float((test_scores > t).mean()) for t in thresholds]
    return {
        "thresholds": thresholds,
        "fpr_val": fpr_val,
        "fpr_test": fpr_test,
    }
