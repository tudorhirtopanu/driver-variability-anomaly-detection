#!/usr/bin/env python3

"""Nominal FPR evaluation for the fixed-gate HCRL experiment.

Unlike the standard evaluator, this script ignores the gate's learned alpha
output and re-scores each window with one or more user-specified fixed alpha
values.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data.hcrl import IGNORED_FEATURES, META_COLS, list_files_for_drivers, parse_driver_list
from data.preprocessing import (
    align_columns,
    build_windows_from_dfs,
    compute_train_stats,
    load_and_clean_csv,
    load_split_dfs,
    normalize_df,
)
from evaluation.fpr_eval.eval_models import load_gated_checkpoint
from models.gated_detector import GatedAnomalyDetector
from models.forecasters import GaussianLSTMForecaster


class WindowDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


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


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def parse_quantiles(value: str) -> List[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    qs = [float(p) for p in parts]
    for q in qs:
        if q <= 0.0 or q >= 1.0:
            raise ValueError(f"Quantile out of range (0,1): {q}")
    return sorted(qs)


def gaussian_nll_per_dim(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    log_sigma: torch.Tensor,
    include_const: bool,
) -> torch.Tensor:
    nll = GatedAnomalyDetector.gaussian_nll_per_dim(
        x=x,
        mu=mu,
        sigma=sigma,
        log_sigma=log_sigma,
        include_const=include_const,
    )
    return nll


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
def collect_scores_gated_fixed(
    model: GatedAnomalyDetector,
    loader: DataLoader,
    device: torch.device,
    fixed_gate_alpha: float,
    use_mixture_nll: bool,
) -> np.ndarray:
    model.eval()
    scores = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)
        # Replace the learned per-feature alpha with a constant so the only
        # thing changing across runs is the fixed gate setting itself.
        alpha = out["forecast_loss"].new_full(out["forecast_loss"].shape, fixed_gate_alpha)
        if use_mixture_nll:
            combined = GatedAnomalyDetector.mixture_nll(
                out["forecast_loss"],
                out["marginal_loss_cal"],
                alpha,
            )
        else:
            combined = (1.0 - alpha) * out["forecast_loss"] + alpha * out["marginal_loss_cal"]
        scores.append(combined.mean(dim=1).detach().cpu().numpy())
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
    # Validation quantiles define the operating points; the reported test FPR is
    # then measured at those exact thresholds.
    thresholds = [float(np.quantile(val_scores, q)) for q in quantiles]
    fpr_val = [float((val_scores > t).mean()) for t in thresholds]
    fpr_test = [float((test_scores > t).mean()) for t in thresholds]
    return {
        "thresholds": thresholds,
        "fpr_val": fpr_val,
        "fpr_test": fpr_test,
    }

def parse_args():
    p = argparse.ArgumentParser("Evaluate FPR for fixed-gate only")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--train-drivers", required=True)
    p.add_argument("--val-drivers", required=True)
    p.add_argument("--test-drivers", required=True)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--include-gaussian-const", action="store_true", default=False)
    p.add_argument("--gated-ckpt", type=Path, required=True)

    p.add_argument("--gated-hidden-dim", type=int, default=128)
    p.add_argument("--gated-num-layers", type=int, default=2)
    p.add_argument("--gated-dropout", type=float, default=0.1)
    p.add_argument("--gated-use-residual", dest="gated_use_residual", action="store_true")
    p.add_argument("--no-gated-use-residual", dest="gated_use_residual", action="store_false")
    p.set_defaults(gated_use_residual=False)
    p.add_argument("--gated-use-mixture-nll", dest="gated_use_mixture_nll", action="store_true")
    p.add_argument("--no-gated-use-mixture-nll", dest="gated_use_mixture_nll", action="store_false")
    p.set_defaults(gated_use_mixture_nll=True)
    p.add_argument(
        "--fixed-gate-alpha",
        type=float,
        default=None,
        help="Single fixed gate alpha in [0,1] used during evaluation.",
    )
    p.add_argument(
        "--fixed-gate-alphas",
        type=str,
        default="0.3,0.5,0.7",
        help="Comma-separated fixed gate alphas in [0,1] used during evaluation.",
    )

    p.add_argument("--quantiles", type=str, default="0.99")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=get_default_device())
    return p.parse_args()


def parse_alpha_list(value: str):
    if value is None:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return []
    alphas = [float(p) for p in parts]
    for a in alphas:
        if a < 0.0 or a > 1.0:
            raise ValueError(f"fixed gate alpha out of range [0,1]: {a}")
    return alphas


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Args: %s", vars(args))

    alphas = parse_alpha_list(args.fixed_gate_alphas)
    if args.fixed_gate_alpha is not None:
        if not (0.0 <= args.fixed_gate_alpha <= 1.0):
            raise ValueError("--fixed-gate-alpha must be in [0, 1].")
        alphas = [args.fixed_gate_alpha]
    if not alphas:
        raise ValueError("No fixed gate alphas provided.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    pin_memory = device.type == "cuda"

    train_drivers = parse_driver_list(args.train_drivers)
    val_drivers = parse_driver_list(args.val_drivers)
    test_drivers = parse_driver_list(args.test_drivers)

    train_files = list_files_for_drivers(args.data_dir, train_drivers)
    val_files = list_files_for_drivers(args.data_dir, val_drivers)
    test_files = list_files_for_drivers(args.data_dir, test_drivers)

    train_dfs_raw, base_cols = [], None
    for path in train_files:
        df = load_and_clean_csv(path, META_COLS, IGNORED_FEATURES)
        if df.empty:
            continue
        if base_cols is None:
            base_cols = list(df.columns)
        df = align_columns(df, base_cols, path)
        train_dfs_raw.append(df)
    if not train_dfs_raw:
        raise ValueError("No usable training data after cleaning.")

    means, stds, low_std_features, feature_cols = compute_train_stats(train_dfs_raw)
    if low_std_features:
        logging.info("Dropping low-std features: %s", ", ".join(low_std_features))

    train_dfs = [normalize_df(df, feature_cols, means, stds) for df in train_dfs_raw]
    val_dfs, _ = load_split_dfs(
        val_files,
        base_cols=base_cols,
        feature_cols=feature_cols,
        means=means,
        stds=stds,
        split_name="val",
        meta_cols=META_COLS,
        ignored_features=IGNORED_FEATURES,
    )
    test_dfs, _ = load_split_dfs(
        test_files,
        base_cols=base_cols,
        feature_cols=feature_cols,
        means=means,
        stds=stds,
        split_name="test",
        meta_cols=META_COLS,
        ignored_features=IGNORED_FEATURES,
    )

    x_train, y_train, n_train = build_windows_from_dfs(train_dfs, args.lookback)
    x_val, y_val, n_val = build_windows_from_dfs(val_dfs, args.lookback)
    x_test, y_test, n_test = build_windows_from_dfs(test_dfs, args.lookback)

    logging.info("Samples train=%d val=%d test=%d | D=%d", n_train, n_val, n_test, x_train.shape[2])

    train_loader = DataLoader(
        WindowDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        WindowDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        WindowDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    D = x_train.shape[2]

    # Gated model
    if not args.gated_ckpt.exists():
        raise FileNotFoundError(f"Gated checkpoint not found: {args.gated_ckpt}")
    from models.marginal_flow import infer_marginal_type_from_ckpt, make_marginal_expert

    ckpt_obj = torch.load(args.gated_ckpt, map_location="cpu")
    marginal_type, marginal_meta = infer_marginal_type_from_ckpt(ckpt_obj)
    gated = GatedAnomalyDetector(
        input_dim=D,
        output_dim=D,
        marginal_expert=make_marginal_expert(
            D,
            marginal_type=marginal_type,
            hidden_features=int(marginal_meta.get("marginal_hidden", 64)),
            num_bins=int(marginal_meta.get("marginal_bins", 8)),
            tail_bound=float(marginal_meta.get("marginal_tail", 10.0)),
        ),
        hidden_dim=args.gated_hidden_dim,
        num_layers=args.gated_num_layers,
        dropout=args.gated_dropout,
        gate_use_residual=args.gated_use_residual,
        use_mixture_nll=args.gated_use_mixture_nll,
        include_gaussian_const=args.include_gaussian_const,
    ).float().to(device)
    load_gated_checkpoint(gated, args.gated_ckpt, device)
    logging.info("Loaded gated checkpoint: %s", args.gated_ckpt)

    quantiles = parse_quantiles(args.quantiles)
    logging.info("Quantiles: %s", quantiles)

    summary = {
        "quantiles": quantiles,
        "n_val": int(x_val.shape[0]),
        "n_test": int(x_test.shape[0]),
        "fixed_gate_alphas": alphas,
        "models": {"gated": {}},
    }

    results_by_alpha = {}
    for alpha in alphas:
        scores_val = collect_scores_gated_fixed(
            gated,
            val_loader,
            device,
            fixed_gate_alpha=alpha,
            use_mixture_nll=args.gated_use_mixture_nll,
        )
        scores_test = collect_scores_gated_fixed(
            gated,
            test_loader,
            device,
            fixed_gate_alpha=alpha,
            use_mixture_nll=args.gated_use_mixture_nll,
        )
        result = compute_fpr(scores_val, scores_test, quantiles)
        results_by_alpha[alpha] = result
        summary["models"]["gated"][str(alpha)] = result

    (args.outdir / "fpr_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = args.outdir / "fpr_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        f.write("model,fixed_gate_alpha,quantile,threshold,fpr_val,fpr_test,n_val,n_test\n")
        for alpha, result in results_by_alpha.items():
            for i, q in enumerate(quantiles):
                f.write(
                    f"gated,{alpha},{q},{result['thresholds'][i]},{result['fpr_val'][i]},"
                    f"{result['fpr_test'][i]},{summary['n_val']},{summary['n_test']}\n"
                )
    logging.info("Saved %s and %s", args.outdir / "fpr_summary.json", csv_path)


if __name__ == "__main__":
    main()
