"""Utilities for saving training metadata and lightweight inspection outputs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.gated_detector import GatedAnomalyDetector


def _serialize_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def build_training_metadata(
    args: argparse.Namespace,
    train_files: list[Path],
    val_files: list[Path],
    test_files: list[Path],
    rows_train: int,
    rows_val: int,
    rows_test: int,
    n_train: int,
    n_val: int,
    n_test: int,
    feature_cols: list[str],
    low_std_features: list[str],
    input_dim: int,
    split_info: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:

    meta: Dict[str, object] = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": _serialize_args(args),
        "data": {
            "dataset": args.dataset,
            "data_dir": str(args.data_dir),
            "train_files": [str(p) for p in train_files],
            "val_files": [str(p) for p in val_files],
            "test_files": [str(p) for p in test_files],
            "rows": {"train": rows_train, "val": rows_val, "test": rows_test},
            "samples": {"train": n_train, "val": n_val, "test": n_test},
            "feature_cols": feature_cols,
            "low_std_features": low_std_features,
        },
        "model": {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lookback": args.lookback,
            "use_mixture_nll": args.use_mixture_nll,
            "gate_use_residual": args.gate_use_residual,
            "include_gaussian_const": args.include_gaussian_const,
            "marginal_type": args.marginal_type,
            "marginal_hidden": args.marginal_hidden,
            "marginal_bins": args.marginal_bins,
            "marginal_tail": args.marginal_tail,
        },
    }
    if split_info is not None:
        meta["split"] = {
            "split_json": str(split_info.get("split_json", "")),
            "split_id": split_info.get("split_id"),
            "split_entry": split_info.get("split_entry"),
        }
    return meta


@torch.no_grad()
def dump_inspection_artifacts(
    model: GatedAnomalyDetector,
    loader: DataLoader,
    device: torch.device,
    outdir: Path,
    prefix: str,
    max_batches: int = 200,
    feature_names: Optional[list[str]] = None,
) -> None:
    """
    Save per-feature alpha/loss summaries and sample arrays for inspection.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    model.eval()

    alpha_list = []
    forecast_list = []
    marginal_list = []
    marginal_cal_list = []
    combined_list = []

    for idx, (x, y) in enumerate(loader):
        if idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)

        alpha_list.append(out["alpha"].detach().cpu())
        forecast_list.append(out["forecast_loss"].detach().cpu())
        marginal_list.append(out["marginal_loss"].detach().cpu())
        marginal_cal_list.append(out["marginal_loss_cal"].detach().cpu())
        combined_list.append(out["score_per_feature"].detach().cpu())

    if not alpha_list:
        logging.warning("No batches collected for inspection artifacts (%s).", prefix)
        return

    alpha_all = torch.cat(alpha_list, dim=0)
    forecast_all = torch.cat(forecast_list, dim=0)
    marginal_all = torch.cat(marginal_list, dim=0)
    marginal_cal_all = torch.cat(marginal_cal_list, dim=0)
    combined_all = torch.cat(combined_list, dim=0)

    D = alpha_all.size(1)
    if feature_names is None or len(feature_names) != D:
        if feature_names is not None:
            logging.warning(
                "Feature name count mismatch for %s; using generic names.", prefix
            )
        feature_names = [f"f{i}" for i in range(D)]

    qs = torch.tensor([0.1, 0.5, 0.9], device=alpha_all.device)
    alpha_q = torch.quantile(alpha_all, qs, dim=0)

    alpha_mean = alpha_all.mean(dim=0)
    forecast_mean = forecast_all.mean(dim=0)
    marginal_mean = marginal_all.mean(dim=0)
    combined_mean = combined_all.mean(dim=0)

    summary = {
        "feature_names": feature_names,
        "mean_alpha": [float(v) for v in alpha_mean.tolist()],
        "alpha_p10": [float(v) for v in alpha_q[0].tolist()],
        "alpha_p50": [float(v) for v in alpha_q[1].tolist()],
        "alpha_p90": [float(v) for v in alpha_q[2].tolist()],
        "mean_forecast_nll": [float(v) for v in forecast_mean.tolist()],
        "mean_marginal_surprise": [float(v) for v in marginal_mean.tolist()],
        "mean_combined_score": [float(v) for v in combined_mean.tolist()],
    }

    (outdir / f"per_feature_summary_{prefix}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    np.save(outdir / f"alpha_samples_{prefix}.npy", alpha_all.numpy())
    np.savez_compressed(
        outdir / f"losses_samples_{prefix}.npz",
        forecast_loss=forecast_all.numpy(),
        marginal_loss=marginal_all.numpy(),
        marginal_loss_cal=marginal_cal_all.numpy(),
        score_per_feature=combined_all.numpy(),
    )

    csv_path = outdir / f"alpha_mean_per_feature_{prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "feature",
                "alpha_mean",
                "alpha_p50",
                "forecast_mean",
                "marginal_mean",
                "combined_mean",
            ]
        )
        for i, name in enumerate(feature_names):
            writer.writerow(
                [
                    name,
                    float(alpha_mean[i].item()),
                    float(alpha_q[1, i].item()),
                    float(forecast_mean[i].item()),
                    float(marginal_mean[i].item()),
                    float(combined_mean[i].item()),
                ]
            )
