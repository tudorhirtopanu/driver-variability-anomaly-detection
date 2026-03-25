"""Argument helpers for nominal false-positive-rate evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch


def get_default_device() -> str:
    """Return the preferred evaluation device for the current machine."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_quantiles(value: str) -> List[float]:
    """Parse and validate a comma-separated quantile list."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    qs = [float(p) for p in parts]
    for q in qs:
        if q <= 0.0 or q >= 1.0:
            raise ValueError(f"Quantile out of range (0,1): {q}")
    return sorted(qs)


def parse_args() -> argparse.Namespace:
    """Build and parse the CLI used by nominal FPR evaluation."""
    p = argparse.ArgumentParser("Evaluate FPR for gated vs Gaussian vs MSE baselines")
    p.add_argument("--dataset", type=str, choices=["hcrl", "sonata", "obd"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--train-drivers", default=None)
    p.add_argument("--val-drivers", default=None)
    p.add_argument("--test-drivers", default=None)
    p.add_argument("--split-json", type=Path, default=None)
    p.add_argument("--split-id", type=int, default=0)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--include-gaussian-const", action="store_true", default=False)

    p.add_argument("--gaussian-epochs", type=int, default=20)
    p.add_argument("--mse-epochs", type=int, default=20)
    p.add_argument("--gaussian-lr", type=float, default=1e-3)
    p.add_argument("--mse-lr", type=float, default=1e-3)
    p.add_argument("--gaussian-patience", type=int, default=0)
    p.add_argument("--mse-patience", type=int, default=0)

    p.add_argument("--gaussian-ckpt", type=Path, default=None)
    p.add_argument("--mse-ckpt", type=Path, default=None)
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

    p.add_argument("--quantiles", type=str, default="0.99")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=get_default_device())
    return p.parse_args()
