"""Argument parsing for the main staged U-GMM training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def get_default_device() -> str:
    """Return the preferred execution device for the current machine."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """Build and parse the CLI used by `src/train.py`."""
    p = argparse.ArgumentParser(
        "Train GatedAnomalyDetector (forecaster + marginal + uncertainty gate)"
    )

    p.add_argument(
        "--dataset",
        type=str,
        choices=["hcrl", "sonata", "obd"],
        required=True,
        help="Which dataset-specific preprocessing module to use.",
    )

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

    # Model shape
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    # Forecaster pretrain
    p.add_argument("--pretrain-forecaster-epochs", type=int, default=20)
    p.add_argument("--forecaster-lr", type=float, default=1e-3)

    # Marginal pretrain
    p.add_argument("--pretrain-marginal-epochs", type=int, default=20)
    p.add_argument("--marginal-lr", type=float, default=5e-4)
    p.add_argument(
        "--marginal-type",
        type=str,
        choices=["flow", "gaussian"],
        default="flow",
        help="Marginal expert type (flow = nflows spline, gaussian = diagonal normal).",
    )
    p.add_argument("--marginal-hidden", type=int, default=64)
    p.add_argument("--marginal-bins", type=int, default=8)
    p.add_argument("--marginal-tail", type=float, default=10.0)
    p.add_argument(
        "--marginal-subsample",
        type=float,
        default=1.0,
        help="Fraction of y_train/y_val used for marginal pretraining (0 < f <= 1).",
    )

    # Gate-only training
    p.add_argument("--gate-epochs", "--epochs", dest="gate_epochs", type=int, default=20)
    p.add_argument("--gate-lr", "--lr", dest="gate_lr", type=float, default=1e-3)
    p.add_argument(
        "--gate-l1",
        type=float,
        default=0.01,
        help="Small penalty to avoid always choosing marginal",
    )
    p.add_argument("--use-mixture-nll", dest="use_mixture_nll", action="store_true")
    p.add_argument("--no-use-mixture-nll", dest="use_mixture_nll", action="store_false")
    p.set_defaults(use_mixture_nll=True)
    p.add_argument(
        "--allow-calibration-with-mixture",
        dest="allow_calibration_with_mixture",
        action="store_true",
    )
    p.set_defaults(allow_calibration_with_mixture=False)
    p.add_argument("--gate-aux-weight", type=float, default=0.001)
    p.add_argument("--gate-aux-margin", type=float, default=0.0)
    p.add_argument("--gate-aux-temp", type=float, default=2.0)
    p.add_argument("--gate-prior-weight", type=float, default=0.0)
    p.add_argument("--gate-prior-margin", type=float, default=0.0)
    p.add_argument("--gate-prior-temp", type=float, default=1.0)
    p.add_argument("--gate-use-residual", action="store_true", default=False)
    p.add_argument("--gate-loss-standardize", dest="gate_loss_standardize", action="store_true")
    p.add_argument("--no-gate-loss-standardize", dest="gate_loss_standardize", action="store_false")
    p.set_defaults(gate_loss_standardize=True)
    p.add_argument("--marginal-b-l2", type=float, default=0.0)
    p.add_argument("--detach-gate-input", dest="detach_gate_input", action="store_true")
    p.add_argument("--no-detach-gate-input", dest="detach_gate_input", action="store_false")
    p.set_defaults(detach_gate_input=True)
    p.add_argument("--train-calibration", dest="train_calibration", action="store_true")
    p.add_argument("--no-train-calibration", dest="train_calibration", action="store_false")
    p.set_defaults(train_calibration=True)

    # Finetune
    p.add_argument("--finetune-epochs", type=int, default=0)
    p.add_argument("--finetune-lr", type=float, default=5e-4)
    p.add_argument("--finetune-expert-lr", type=float, default=1e-4)

    # Misc
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--include-gaussian-const", action="store_true", default=True)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=get_default_device())
    return p.parse_args()
