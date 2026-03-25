#!/usr/bin/env python3
"""Synthetic HCRL evaluation using point-adjusted event metrics.

Each injected test file is scored window-by-window, then summarised with AUROC
and the best point-adjusted F1 obtained from a precision-recall sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from torch.utils.data import DataLoader

from data.hcrl import IGNORED_FEATURES, META_COLS, list_files_for_drivers, parse_driver_list
from data.loaders import WindowDataset
from data.preprocessing import align_columns, compute_train_stats, normalize_df
from evaluation.fpr_eval.eval_args import get_default_device
from evaluation.fpr_eval.eval_loops import (
    MSELSTMForecaster,
    gaussian_nll_per_dim,
)
from evaluation.fpr_eval.eval_models import load_gated_checkpoint
from models.forecasters import GaussianLSTMForecaster
from models.gated_detector import GatedAnomalyDetector

BatchEvalFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, dict]]

# -------------------------------
# POINT-ADJUSTED METRIC HELPERS
# -------------------------------

def point_adjustment(score: np.ndarray, label: np.ndarray, threshold: float) -> np.ndarray:
    """
    Applies Point-Adjustment Strategy:
    If *any* point in a contiguous anomaly segment is detected, 
    the entire segment is marked as detected (True Positive).
    """
    predict = score > threshold
    actual = label > 0.5
    
    adjusted_predict = predict.copy()
    anomaly_groups = []
    
    # Identify contiguous anomaly regions in ground truth
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
            
    # If raw prediction hits the group, fill the group in adjusted prediction
    for start, end in anomaly_groups:
        if np.any(predict[start:end]):
            adjusted_predict[start:end] = True
            
    return adjusted_predict.astype(int)

def calculate_event_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Computes AUROC (Threshold Independent) and Best Point-Adjusted F1.
    """

    # A short centred rolling mean reduces single-window spikes before the
    # threshold search, which makes the event-level metric less brittle.
    y_score = pd.Series(y_score).rolling(window=5, center=True, min_periods=1).mean().values
    
    # 1. AUROC
    try:
        if np.unique(y_true).size < 2:
            auroc = 0.5 # Undefined if no anomalies
        else:
            auroc = roc_auc_score(y_true, y_score)
    except:
        auroc = 0.5

    # 2. Find Best Threshold via Precision-Recall Curve (Raw)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # F1 = 2*P*R / (P+R)
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls + 1e-10
    f1_scores = numerator / denominator
    
    best_idx = np.argmax(f1_scores)
    # Precision_recall_curve returns thresholds with length = len(precisions)-1
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = thresholds[-1]

    # 3. Apply Point Adjustment using the Best Threshold
    adj_pred = point_adjustment(y_score, y_true, best_threshold)
    
    tn, fp, fn, tp = confusion_matrix(y_true, adj_pred).ravel()
    
    adj_prec = tp / (tp + fp + 1e-10)
    adj_rec = tp / (tp + fn + 1e-10)
    adj_f1 = 2 * (adj_prec * adj_rec) / (adj_prec + adj_rec + 1e-10)

    return {
        "auroc": auroc,
        "best_threshold": best_threshold,
        "adj_f1": adj_f1,
        "adj_precision": adj_prec,
        "adj_recall": adj_rec,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }

# -------------------------------
# Argument Parsing & Loading
# -------------------------------

def parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Synthetic HCRL evaluation")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--train-data-dir", type=Path, default=None)
    p.add_argument("--test-data-dir", type=Path, default=None)
    p.add_argument("--gated-ckpt", type=Path, default=None)
    p.add_argument("--gaussian-ckpt", type=Path, default=None)
    p.add_argument("--mse-ckpt", type=Path, default=None)
    p.add_argument("--models", type=str, default="gated")
    p.add_argument("--gated-score", choices=["model_score", "forecast_only", "marginal_only", "oracle_window"], default="model_score")
    p.add_argument("--label-col", type=str, default="is_anomaly")
    p.add_argument("--train-drivers", required=True)
    p.add_argument("--val-drivers", required=True)
    p.add_argument("--test-drivers", required=True)

    p.add_argument("--quantile", type=float, default=0.999) 
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--m", type=int, default=1)
    
    p.add_argument("--mode", choices=["linear", "mixture"], default="linear")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    
    # Model Params
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gated-hidden-dim", type=int, default=128)
    p.add_argument("--gated-num-layers", type=int, default=2)
    p.add_argument("--gated-dropout", type=float, default=0.1)
    p.add_argument("--gated-use-residual", action="store_true", default=False)
    p.add_argument("--include-gaussian-const", action="store_true", default=False)
    
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=get_default_device())
    return p.parse_args()

def load_feature_csv(path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Time(s)" in df.columns:
        df = df.sort_values(by="Time(s)", kind="mergesort")
    if label_col in df.columns:
        df = df.drop(columns=[label_col], errors="ignore")
    df = df.drop(columns=[c for c in META_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in IGNORED_FEATURES if c in df.columns], errors="ignore")
    if df.empty: return df
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    return df

def load_labeled_csv(path: Path, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Time(s)" in df.columns: df = df.sort_values(by="Time(s)", kind="mergesort")
    labels = pd.to_numeric(df[label_col], errors="coerce")
    df = df.drop(columns=[label_col], errors="ignore")
    df = df.drop(columns=[c for c in META_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in IGNORED_FEATURES if c in df.columns], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    mask = df.notna().all(axis=1) & labels.notna()
    return df.loc[mask], (labels.loc[mask] > 0).astype(int)

def build_windows(df: pd.DataFrame, labels: pd.Series, lookback: int):
    data = df.to_numpy(dtype=np.float32)
    label_arr = labels.to_numpy(dtype=np.int32)
    n_samples = data.shape[0] - lookback
    if n_samples <= 0: return np.empty((0,)), np.empty((0,)), np.empty((0,))
    windows = [data[i : i + n_samples] for i in range(lookback)]
    x = np.stack(windows, axis=1)
    y = data[lookback:]
    y_labels = label_arr[lookback:]
    return x, y, y_labels

# -------------------------------
# MODEL LOADING
# -------------------------------

def get_model_and_score_fns(args, model_name, feature_cols, device):
    if model_name == "gated":
        from models.marginal_flow import infer_marginal_type_from_ckpt, make_marginal_expert

        ckpt_obj = torch.load(args.gated_ckpt, map_location="cpu")
        marginal_type, marginal_meta = infer_marginal_type_from_ckpt(ckpt_obj)
        model = GatedAnomalyDetector(
            input_dim=len(feature_cols), output_dim=len(feature_cols),
            marginal_expert=make_marginal_expert(
                len(feature_cols),
                marginal_type=marginal_type,
                hidden_features=int(marginal_meta.get("marginal_hidden", 64)),
                num_bins=int(marginal_meta.get("marginal_bins", 8)),
                tail_bound=float(marginal_meta.get("marginal_tail", 10.0)),
            ),
            hidden_dim=args.gated_hidden_dim, num_layers=args.gated_num_layers,
            dropout=args.gated_dropout, gate_use_residual=args.gated_use_residual,
            use_mixture_nll=(args.mode=="mixture"), include_gaussian_const=args.include_gaussian_const
        ).float().to(device)
        load_gated_checkpoint(model, args.gated_ckpt, device)
        model.eval()
        
        def eval_batch(x, y):
            out = model(x, y)
            if args.gated_score == "model_score": score = out["score"]
            elif args.gated_score == "forecast_only": score = out["forecast_loss"].mean(1)
            elif args.gated_score == "marginal_only": score = out["marginal_loss_cal"].mean(1)
            elif args.gated_score == "oracle_window": 
                score = torch.minimum(out["forecast_loss"].mean(1), out["marginal_loss_cal"].mean(1))
            return score

    elif model_name == "gaussian":
        model = GaussianLSTMForecaster(
            input_size=len(feature_cols), hidden_size=args.hidden_dim,
            num_layers=args.num_layers, dropout=args.dropout, output_dim=len(feature_cols)
        ).float().to(device)
        model.load_state_dict(torch.load(args.gaussian_ckpt, map_location=device))
        model.eval()
        
        def eval_batch(x, y):
            mu, sigma, log_sigma = model(x)
            nll = gaussian_nll_per_dim(y, mu, sigma, log_sigma, args.include_gaussian_const)
            return nll.mean(dim=1)

    elif model_name == "mse":
        model = MSELSTMForecaster(
            input_size=len(feature_cols), hidden_size=args.hidden_dim,
            num_layers=args.num_layers, dropout=args.dropout, output_dim=len(feature_cols)
        ).float().to(device)
        model.load_state_dict(torch.load(args.mse_ckpt, map_location=device))
        model.eval()
        
        def eval_batch(x, y):
            mu = model(x)
            return (mu - y).pow(2).mean(dim=1)
            
    return eval_batch

# -------------------------------
# MAIN EVALUATION LOOP
# -------------------------------

def run_evaluation(args):
    # 1. Setup Data
    train_drivers = parse_driver_list(args.train_drivers)
    test_drivers = parse_driver_list(args.test_drivers)
    
    train_data_dir = args.train_data_dir or args.data_dir
    test_data_dir = args.test_data_dir or args.data_dir
    
    train_files = list_files_for_drivers(train_data_dir, train_drivers)
    train_dfs_raw = []
    base_cols = None
    for path in train_files:
        df = load_feature_csv(path, args.label_col)
        if not df.empty:
            if base_cols is None: base_cols = list(df.columns)
            train_dfs_raw.append(align_columns(df, base_cols, path))
            
    means, stds, _, feature_cols = compute_train_stats(train_dfs_raw)
    device = torch.device(args.device)
    test_files = list_files_for_drivers(test_data_dir, test_drivers)
    
    results = []
    models = parse_csv_list(args.models)
    
    for model_name in models:
        logging.info(f"--- Evaluating Model: {model_name} ---")
        eval_batch_fn = get_model_and_score_fns(args, model_name, feature_cols, device)
        
        for path in test_files:
            # Each injected file is evaluated independently so per-file metrics
            # can be analysed later without averaging away hard cases.
            df, labels = load_labeled_csv(path, args.label_col)
            if df.empty: continue
            df = align_columns(df, base_cols, path)
            df = normalize_df(df, feature_cols, means, stds)
            
            x_test, y_test, y_labels = build_windows(df, labels, args.lookback)
            if y_labels.size == 0: continue
            
            loader = DataLoader(WindowDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
            
            # Inference
            scores_list = []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    scores_list.append(eval_batch_fn(x, y).cpu().numpy())
            
            y_scores = np.concatenate(scores_list)
            
            # Calculate event-level metrics after point adjustment
            metrics = calculate_event_metrics(y_labels, y_scores)
            
            print(f"File: {path.name:<20} | AUROC: {metrics['auroc']:.4f} | PA-F1: {metrics['adj_f1']:.4f}")
            
            results.append({
                "model": model_name,
                "file": path.name,
                **metrics
            })
            
    # Save Results
    out_df = pd.DataFrame(results)
    if args.outdir.suffix.lower() == ".csv":
        save_path = args.outdir
    else:
        save_path = args.outdir / "event_evaluation_results.csv"
    out_df.to_csv(save_path, index=False)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY (Mean across files)")
    print("="*60)
    summary = out_df.groupby("model")[["auroc", "adj_f1"]].mean()
    print(summary)
    print(f"\nDetailed results saved to: {save_path}")

def main():
    args = parse_args()
    if args.outdir.suffix.lower() == ".csv":
        args.outdir.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    
    run_evaluation(args)

if __name__ == "__main__":
    main()
