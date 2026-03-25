#!/usr/bin/env python3
"""
Oracle-style evaluation on injected OBD test files across splits.

Per file:
  - compute AUROC, PR-AUC from raw scores
  - compute precision-recall curve and best F1 threshold (oracle per-file)
  - report precision/recall/F1 at best threshold
  - report event-adjusted precision/recall/F1 at best threshold

Per split:
  - mean/std across files, per model

Overall:
  - mean/std across splits, per model
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from data import obd as data
from data.loaders import WindowDataset
from data.preprocessing import align_columns, compute_train_stats, normalize_df
from evaluation.fpr_eval.eval_args import get_default_device
from evaluation.fpr_eval.eval_loops import MSELSTMForecaster, gaussian_nll_per_dim
from evaluation.fpr_eval.eval_models import load_gated_checkpoint
from models.forecasters import GaussianLSTMForecaster
from models.gated_detector import GatedAnomalyDetector
from path_config import get_path
from split_utils import resolve_split_files


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_run_specs(aggregate_path: Path) -> List[Tuple[str, int]]:
    if aggregate_path.is_dir():
        runs: List[Tuple[int, str, int]] = []
        for path in sorted(aggregate_path.iterdir()):
            if not path.is_dir():
                continue
            match = re.fullmatch(r"v(\d+)", path.name)
            if match:
                run_label = path.name
                split_id = int(match.group(1))
                runs.append((int(match.group(1)), run_label, split_id))
        if runs:
            return [(run_label, split_id) for _, run_label, split_id in sorted(runs)]
        raise ValueError(f"No split directories like v1, v2, ... found in {aggregate_path}")

    data = json.loads(aggregate_path.read_text(encoding="utf-8"))
    used = data.get("splits_used", [])
    runs: List[Tuple[str, int]] = []
    for name in used:
        m = re.search(r"fpr_split(\d+)_sub", name)
        if not m:
            raise ValueError(f"Unrecognized split name: {name}")
        split_id = int(m.group(1))
        run_label = f"split_{split_id}"
        runs.append((run_label, split_id))
    return runs


def resolve_repo_checkpoints(
    aggregate_path: Path,
    run_label: str,
    split_id: int,
    sub: str,
) -> Tuple[Path, Path, Path]:
    _ = sub
    fpr_dir = aggregate_path if aggregate_path.is_dir() else aggregate_path.parent
    root = fpr_dir.parent
    version = run_label
    ckpt_dir = root / "checkpoints" / version
    fpr_run_dir = root / "fpr_comparison" / version
    if not ckpt_dir.exists():
        version = f"v{split_id}"
        ckpt_dir = root / "checkpoints" / version
        fpr_run_dir = root / "fpr_comparison" / version
    return (
        ckpt_dir / "gated_model.pt",
        fpr_run_dir / "gaussian_lstm.pt",
        fpr_run_dir / "mse_lstm.pt",
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.replace("Â", ""))


def load_labeled_csv(path: Path, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    if "Time" in df.columns:
        df = df.sort_values(by="Time", kind="mergesort")
    elif "Time(s)" in df.columns:
        df = df.sort_values(by="Time(s)", kind="mergesort")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {path.name}")
    labels = pd.to_numeric(df[label_col], errors="coerce")
    df = df.drop(columns=[label_col], errors="ignore")
    df = df.drop(columns=[c for c in data.META_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in data.IGNORED_FEATURES if c in df.columns], errors="ignore")
    if df.empty:
        return df, labels
    df = df.apply(pd.to_numeric, errors="coerce")
    mask = df.notna().all(axis=1) & labels.notna()
    return df.loc[mask], (labels.loc[mask] > 0).astype(int)


def build_windows_with_labels(df: pd.DataFrame, labels: pd.Series, lookback: int):
    data_arr = df.to_numpy(dtype=np.float32)
    label_arr = labels.to_numpy(dtype=np.int32)
    n_samples = data_arr.shape[0] - lookback
    if n_samples <= 0:
        return None
    windows = [data_arr[i : i + n_samples] for i in range(lookback)]
    x = np.stack(windows, axis=1)
    y = data_arr[lookback:]
    y_labels = label_arr[lookback:]
    return x, y, y_labels


def point_adjustment(score: np.ndarray, label: np.ndarray, threshold: float) -> np.ndarray:
    predict = score > threshold
    actual = label > 0.5

    adjusted_predict = predict.copy()
    anomaly_groups = []
    is_anomaly = False
    start = 0
    for i in range(len(actual)):
        if actual[i] and not is_anomaly:
            is_anomaly = True
            start = i
        elif not actual[i] and is_anomaly:
            is_anomaly = False
            anomaly_groups.append((start, i))
    if is_anomaly:
        anomaly_groups.append((start, len(actual)))

    for start, end in anomaly_groups:
        if np.any(predict[start:end]):
            adjusted_predict[start:end] = True
    return adjusted_predict.astype(int)


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    raw_pred = (y_score > threshold).astype(int)
    adj_pred = point_adjustment(y_score, y_true, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, raw_pred).ravel()
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)

    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_true, adj_pred).ravel()
    adj_prec = tp_a / (tp_a + fp_a + 1e-10)
    adj_rec = tp_a / (tp_a + fn_a + 1e-10)
    adj_f1 = 2 * (adj_prec * adj_rec) / (adj_prec + adj_rec + 1e-10)

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "adj_precision": float(adj_prec),
        "adj_recall": float(adj_rec),
        "adj_f1": float(adj_f1),
    }


def threshold_free_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    if np.unique(y_true).size < 2:
        return {"auroc": 0.5, "pr_auc": 0.0}
    try:
        auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = 0.0
    return {"auroc": float(auroc), "pr_auc": float(pr_auc)}


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Handle degenerate cases
    if np.unique(y_true).size < 2:
        # If all zeros, predict none; if all ones, predict all
        if np.all(y_true == 0):
            threshold = float("inf")
        else:
            threshold = float("-inf")
        metrics = metrics_at_threshold(y_true, y_score, threshold)
        return threshold, metrics

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        threshold = float("inf")
        metrics = metrics_at_threshold(y_true, y_score, threshold)
        return threshold, metrics

    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = int(np.nanargmax(f1))
    threshold = float(thresholds[best_idx])
    # The threshold is chosen independently per file because this script is an
    # oracle analysis of synthetic detectability rather than a deployment-style
    # calibration run.
    metrics = metrics_at_threshold(y_true, y_score, threshold)
    return threshold, metrics


def get_eval_fn(args, model_name: str, feature_cols: List[str], device: torch.device):
    if model_name == "gated":
        from models.marginal_flow import infer_marginal_type_from_ckpt, make_marginal_expert

        ckpt_obj = torch.load(args.gated_ckpt, map_location="cpu")
        marginal_type, marginal_meta = infer_marginal_type_from_ckpt(ckpt_obj)
        marginal_expert = make_marginal_expert(
            len(feature_cols),
            marginal_type=marginal_type,
            hidden_features=int(marginal_meta.get("marginal_hidden", 64)),
            num_bins=int(marginal_meta.get("marginal_bins", 8)),
            tail_bound=float(marginal_meta.get("marginal_tail", 10.0)),
        )
        model = GatedAnomalyDetector(
            input_dim=len(feature_cols),
            output_dim=len(feature_cols),
            marginal_expert=marginal_expert,
            hidden_dim=args.gated_hidden_dim,
            num_layers=args.gated_num_layers,
            dropout=args.gated_dropout,
            gate_use_residual=args.gated_use_residual,
            use_mixture_nll=(args.mode == "mixture"),
            include_gaussian_const=args.include_gaussian_const,
        ).float().to(device)
        load_gated_checkpoint(model, args.gated_ckpt, device)
        model.eval()

        def eval_batch(x, y):
            out = model(x, y)
            return out["score"]

    elif model_name == "gaussian":
        model = GaussianLSTMForecaster(
            input_size=len(feature_cols),
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            output_dim=len(feature_cols),
        ).float().to(device)
        model.load_state_dict(torch.load(args.gaussian_ckpt, map_location=device))
        model.eval()

        def eval_batch(x, y):
            mu, sigma, log_sigma = model(x)
            nll = gaussian_nll_per_dim(y, mu, sigma, log_sigma, args.include_gaussian_const)
            return nll.mean(dim=1)

    elif model_name == "mse":
        model = MSELSTMForecaster(
            input_size=len(feature_cols),
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            output_dim=len(feature_cols),
        ).float().to(device)
        model.load_state_dict(torch.load(args.mse_ckpt, map_location=device))
        model.eval()

        def eval_batch(x, y):
            mu = model(x)
            return (mu - y).pow(2).mean(dim=1)

    return eval_batch


def collect_scores_labels(
    files: List[Path],
    base_cols: List[str],
    feature_cols: List[str],
    means: pd.Series,
    stds: pd.Series,
    eval_fn,
    device: torch.device,
    lookback: int,
    batch_size: int,
    label_col: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    scores_per_file = []
    labels_per_file = []
    names = []
    for path in files:
        df, labels = load_labeled_csv(path, label_col)
        if df.empty:
            continue
        df = align_columns(df, base_cols, path)
        df = normalize_df(df, feature_cols, means, stds)
        windows = build_windows_with_labels(df, labels, lookback)
        if windows is None:
            continue
        x, y, y_labels = windows
        loader = DataLoader(WindowDataset(x, y), batch_size=batch_size, shuffle=False)
        scores_list = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                scores_list.append(eval_fn(xb, yb).cpu().numpy())
        if scores_list:
            scores_per_file.append(np.concatenate(scores_list))
            labels_per_file.append(y_labels)
            names.append(path.name)
    return scores_per_file, labels_per_file, names


@dataclass
class SplitResult:
    split_id: int
    per_file_df: pd.DataFrame
    summary_df: pd.DataFrame


def evaluate_split(args, split_id: int) -> SplitResult:
    split_info = resolve_split_files(args.split_json, split_id, args.data_dir)
    train_files = split_info["train"]
    test_files = split_info["test"]

    # Filter injected files by anomaly log so the evaluation ignores injections
    # that were skipped or failed during generation.
    injected_allow = None
    if args.anomaly_log is not None:
        log_df = pd.read_csv(args.anomaly_log)
        injected_allow = set(log_df.loc[log_df["status"] == "success", "filename"].astype(str))

    injected_files = []
    for path in test_files:
        injected_path = args.injected_dir / path.name
        if not injected_path.exists():
            continue
        if injected_allow is not None and path.name not in injected_allow:
            continue
        injected_files.append(injected_path)

    # Compute normalization stats from the nominal training files listed in the
    # split JSON, then reuse those stats for all injected test files.
    train_dfs_raw = []
    base_cols = None
    for path in train_files:
        df = pd.read_csv(path)
        df = _normalize_columns(df)
        df = df.drop(columns=[c for c in data.META_COLS if c in df.columns], errors="ignore")
        df = df.drop(columns=[c for c in data.IGNORED_FEATURES if c in df.columns], errors="ignore")
        if df.empty:
            continue
        if base_cols is None:
            base_cols = list(df.columns)
        train_dfs_raw.append(align_columns(df, base_cols, path))
    means, stds, _, feature_cols = compute_train_stats(train_dfs_raw)

    device = torch.device(args.device)
    models = ["gated", "gaussian", "mse"]
    eval_fns = {m: get_eval_fn(args, m, feature_cols, device) for m in models}

    rows = []
    for m in models:
        scores_per_file, labels_per_file, names = collect_scores_labels(
            injected_files,
            base_cols,
            feature_cols,
            means,
            stds,
            eval_fns[m],
            device,
            args.lookback,
            args.batch_size,
            args.label_col,
        )
        for scores, labels, fname in zip(scores_per_file, labels_per_file, names):
            tf = threshold_free_metrics(labels, scores)
            tau, metrics = best_f1_threshold(labels, scores)
            rows.append(
                {
                    "split_id": split_id,
                    "model": m,
                    "filename": fname,
                    "best_threshold": tau,
                    "best_f1": metrics["f1"],
                    "precision_best": metrics["precision"],
                    "recall_best": metrics["recall"],
                    "adj_f1_best": metrics["adj_f1"],
                    "adj_precision_best": metrics["adj_precision"],
                    "adj_recall_best": metrics["adj_recall"],
                    "auroc": tf["auroc"],
                    "pr_auc": tf["pr_auc"],
                }
            )

    per_file_df = pd.DataFrame(rows)
    if per_file_df.empty:
        summary_df = pd.DataFrame(
            [{"split_id": split_id, "model": m, "n_files": 0} for m in ["gated", "gaussian", "mse"]]
        )
        return SplitResult(split_id=split_id, per_file_df=per_file_df, summary_df=summary_df)

    metric_cols = [
        "best_f1",
        "precision_best",
        "recall_best",
        "adj_f1_best",
        "adj_precision_best",
        "adj_recall_best",
        "auroc",
        "pr_auc",
    ]
    summary = (
        per_file_df.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten columns
    summary.columns = ["model"] + [f"{c[0]}_{c[1]}" for c in summary.columns[1:]]
    summary.insert(0, "split_id", split_id)
    summary["n_files"] = per_file_df.groupby("model")["filename"].count().values

    return SplitResult(split_id=split_id, per_file_df=per_file_df, summary_df=summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Oracle eval across OBD splits")
    p.add_argument(
        "--aggregate-path",
        type=Path,
        default=Path("results/obd/fpr_comparison"),
    )
    p.add_argument(
        "--split-json",
        type=Path,
        default=Path("fp_analysis/splits/generated/obd/splits_obd.json"),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=get_path("OBD_DATA_DIR", default="path/to/obd_data"),
    )
    p.add_argument(
        "--injected-dir",
        type=Path,
        default=Path("synthetic_data/obd"),
    )
    p.add_argument(
        "--anomaly-log",
        type=Path,
        default=Path("synthetic_data/obd/anomaly_log.csv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/obd/synth_evaluation"),
    )
    p.add_argument("--sub", type=str, default="0.1")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gated-hidden-dim", type=int, default=128)
    p.add_argument("--gated-num-layers", type=int, default=2)
    p.add_argument("--gated-dropout", type=float, default=0.1)
    p.add_argument("--gated-use-residual", action="store_true", default=False)
    p.add_argument("--include-gaussian-const", action="store_true", default=False)
    p.add_argument("--mode", choices=["linear", "mixture"], default="mixture")
    p.add_argument("--label-col", type=str, default="is_anomaly")
    p.add_argument("--device", type=str, default=get_default_device())
    p.add_argument("--first-n", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    run_specs = parse_run_specs(args.aggregate_path)
    if args.first_n is not None:
        run_specs = run_specs[: args.first_n]

    all_split_summaries = []

    for run_label, sid in run_specs:
        log(f"Evaluating {run_label} (split {sid})")
        split_outdir = args.outdir / run_label
        split_outdir.mkdir(parents=True, exist_ok=True)

        # Each run label resolves to one set of trained checkpoints and nominal
        # baseline models for that split.
        args.gated_ckpt, args.gaussian_ckpt, args.mse_ckpt = resolve_repo_checkpoints(
            args.aggregate_path,
            run_label,
            sid,
            args.sub,
        )

        result = evaluate_split(args, sid)
        per_file_path = split_outdir / "per_file_metrics.csv"
        result.per_file_df.to_csv(per_file_path, index=False)

        split_summary_path = split_outdir / "oracle_split_summary.csv"
        result.summary_df.to_csv(split_summary_path, index=False)

        all_split_summaries.append(result.summary_df)

    if not all_split_summaries:
        log("No split summaries produced.")
        return

    all_summaries = pd.concat(all_split_summaries, ignore_index=True)
    all_summaries.to_csv(args.outdir / "oracle_split_summary.csv", index=False)

    # Overall summary across splits (per model)
    metric_cols = [c for c in all_summaries.columns if c.endswith("_mean")]
    overall = (
        all_summaries.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    overall.columns = ["model"] + [f"{c[0]}_{c[1]}" for c in overall.columns[1:]]
    overall.to_csv(args.outdir / "oracle_overall_summary.csv", index=False)
    overall.to_csv(args.outdir / "summary.csv", index=False)

    log(f"Wrote {args.outdir / 'oracle_split_summary.csv'}")
    log(f"Wrote {args.outdir / 'oracle_overall_summary.csv'}")
    log(f"Wrote {args.outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
