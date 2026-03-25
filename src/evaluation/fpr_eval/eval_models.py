"""Model-loading helpers for nominal false-positive-rate evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.forecasters import GaussianLSTMForecaster
from models.gated_detector import GatedAnomalyDetector

from .eval_loops import (
    MSELSTMForecaster,
    eval_gaussian_epoch,
    eval_mse_epoch,
    train_gaussian_epoch,
    train_mse_epoch,
)


def load_gated_checkpoint(
    model: GatedAnomalyDetector,
    ckpt_path: Path,
    device: torch.device,
) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    buffer_keys = [
        "forecast_mean",
        "forecast_std",
        "marginal_mean",
        "marginal_std",
        "logsig_mean",
        "logsig_std",
        "resid_mean",
        "resid_std",
        "alpha_prior",
    ]
    stats = {}
    for key in buffer_keys:
        if key in state:
            stats[key] = state.pop(key)

    # The checkpoint stores learned modules and calibration/statistical buffers
    # together. The buffers are restored through the model helpers below because
    # some of them are registered conditionally depending on how the model was
    # constructed.
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = set(stats.keys())
    extra_missing = [k for k in missing if k not in allowed_missing]
    if extra_missing or unexpected:
        raise RuntimeError(
            f"Gated checkpoint mismatch: missing={extra_missing}, unexpected={unexpected}"
        )

    if {"forecast_mean", "forecast_std", "marginal_mean", "marginal_std"} <= stats.keys():
        model.set_loss_stats(
            stats["forecast_mean"],
            stats["forecast_std"],
            stats["marginal_mean"],
            stats["marginal_std"],
        )
    if {"logsig_mean", "logsig_std"} <= stats.keys():
        if {"resid_mean", "resid_std"} <= stats.keys():
            model.set_gate_input_stats(
                stats["logsig_mean"],
                stats["logsig_std"],
                stats["resid_mean"],
                stats["resid_std"],
            )
        else:
            model.set_gate_input_stats(
                stats["logsig_mean"],
                stats["logsig_std"],
            )
    if "alpha_prior" in stats:
        model.set_gate_prior(stats["alpha_prior"])


def load_raw_train_dfs(data, files):
    base_cols = None
    train_dfs = []
    for path in files:
        df = data.load_and_clean_csv(path, data.META_COLS, data.IGNORED_FEATURES)
        if df.empty:
            logging.warning("No rows after cleaning in %s", path.name)
            continue
        if base_cols is None:
            base_cols = list(df.columns)
        df = data.align_columns(df, base_cols, path)
        train_dfs.append(df)
    if not train_dfs:
        raise ValueError("No usable training data after cleaning.")
    return train_dfs, base_cols


def train_or_load_gaussian(
    args,
    D: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
):
    gaussian = GaussianLSTMForecaster(
        input_size=D,
        hidden_size=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=D,
    ).float().to(device)
    gaussian_ckpt = args.gaussian_ckpt or (args.outdir / "gaussian_lstm.pt")
    if args.gaussian_ckpt is None:
        # FPR evaluation can either reuse an existing baseline or train one
        # into the evaluation directory, which keeps the workflow self-contained.
        logging.info("Training Gaussian baseline for %d epochs...", args.gaussian_epochs)
        opt = torch.optim.Adam(gaussian.parameters(), lr=args.gaussian_lr)
        best_val = float("inf")
        best_state = None
        no_improve = 0
        for e in range(1, args.gaussian_epochs + 1):
            tr = train_gaussian_epoch(
                gaussian,
                train_loader,
                opt,
                device,
                args.clip_grad,
                args.include_gaussian_const,
                f"g-tr {e}",
            )
            va = eval_gaussian_epoch(
                gaussian,
                val_loader,
                device,
                args.include_gaussian_const,
                f"g-va {e}",
            )
            logging.info("Gaussian epoch %d | train=%.6f val=%.6f", e, tr, va)
            if va < best_val:
                best_val = va
                best_state = {k: v.detach().cpu().clone() for k, v in gaussian.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if args.gaussian_patience > 0 and no_improve >= args.gaussian_patience:
                logging.info("Early stopping Gaussian at epoch %d (best val=%.6f).", e, best_val)
                break
        if best_state is not None:
            gaussian.load_state_dict(best_state)
        torch.save(gaussian.state_dict(), gaussian_ckpt)
        logging.info("Saved %s (best val=%.6f)", gaussian_ckpt, best_val)
    else:
        if not gaussian_ckpt.exists():
            raise FileNotFoundError(f"Gaussian checkpoint not found: {gaussian_ckpt}")
        gaussian.load_state_dict(torch.load(gaussian_ckpt, map_location=device))
        logging.info("Loaded Gaussian checkpoint: %s", gaussian_ckpt)

    return gaussian


def train_or_load_mse(
    args,
    D: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
):
    mse = MSELSTMForecaster(
        input_size=D,
        hidden_size=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=D,
    ).float().to(device)
    mse_ckpt = args.mse_ckpt or (args.outdir / "mse_lstm.pt")
    if args.mse_ckpt is None:
        # Matching the Gaussian path above, the MSE baseline is trained on the
        # nominal train/val split when no checkpoint is supplied.
        logging.info("Training MSE baseline for %d epochs...", args.mse_epochs)
        opt = torch.optim.Adam(mse.parameters(), lr=args.mse_lr)
        best_val = float("inf")
        best_state = None
        no_improve = 0
        for e in range(1, args.mse_epochs + 1):
            tr = train_mse_epoch(mse, train_loader, opt, device, args.clip_grad, f"m-tr {e}")
            va = eval_mse_epoch(mse, val_loader, device, f"m-va {e}")
            logging.info("MSE epoch %d | train=%.6f val=%.6f", e, tr, va)
            if va < best_val:
                best_val = va
                best_state = {k: v.detach().cpu().clone() for k, v in mse.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if args.mse_patience > 0 and no_improve >= args.mse_patience:
                logging.info("Early stopping MSE at epoch %d (best val=%.6f).", e, best_val)
                break
        if best_state is not None:
            mse.load_state_dict(best_state)
        torch.save(mse.state_dict(), mse_ckpt)
        logging.info("Saved %s (best val=%.6f)", mse_ckpt, best_val)
    else:
        if not mse_ckpt.exists():
            raise FileNotFoundError(f"MSE checkpoint not found: {mse_ckpt}")
        mse.load_state_dict(torch.load(mse_ckpt, map_location=device))
        logging.info("Loaded MSE checkpoint: %s", mse_ckpt)

    return mse


def build_and_load_gated(args, D: int, device: torch.device):
    from models.marginal_flow import infer_marginal_type_from_ckpt, make_marginal_expert

    if not args.gated_ckpt.exists():
        raise FileNotFoundError(f"Gated checkpoint not found: {args.gated_ckpt}")
    ckpt_obj = torch.load(args.gated_ckpt, map_location="cpu")
    # The gated checkpoint contains enough metadata to rebuild the marginal
    # expert architecture before the state dict is loaded.
    marginal_type, marginal_meta = infer_marginal_type_from_ckpt(ckpt_obj)
    marginal_expert = make_marginal_expert(
        D,
        marginal_type=marginal_type,
        hidden_features=int(marginal_meta.get("marginal_hidden", 64)),
        num_bins=int(marginal_meta.get("marginal_bins", 8)),
        tail_bound=float(marginal_meta.get("marginal_tail", 10.0)),
    )
    gated = GatedAnomalyDetector(
        input_dim=D,
        output_dim=D,
        marginal_expert=marginal_expert,
        hidden_dim=args.gated_hidden_dim,
        num_layers=args.gated_num_layers,
        dropout=args.gated_dropout,
        gate_use_residual=args.gated_use_residual,
        use_mixture_nll=args.gated_use_mixture_nll,
        include_gaussian_const=args.include_gaussian_const,
    ).float().to(device)
    load_gated_checkpoint(gated, args.gated_ckpt, device)
    logging.info("Loaded gated checkpoint: %s", args.gated_ckpt)
    return gated


def write_fpr_outputs(outdir: Path, summary: dict, quantiles):
    # JSON is convenient for scripts; CSV is convenient for quick inspection
    # and plotting outside Python.
    (outdir / "fpr_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    csv_path = outdir / "fpr_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        f.write("model,quantile,threshold,fpr_val,fpr_test,n_val,n_test\n")
        for model_name, result in summary["models"].items():
            for i, q in enumerate(quantiles):
                f.write(
                    f"{model_name},{q},{result['thresholds'][i]},{result['fpr_val'][i]},"
                    f"{result['fpr_test'][i]},{summary['n_val']},{summary['n_test']}\n"
                )
    logging.info("Saved %s and %s", outdir / "fpr_summary.json", csv_path)
