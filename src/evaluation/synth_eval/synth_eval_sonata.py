#!/usr/bin/env python3

"""Synthetic Sonata evaluation for the retained stage-2 experiment summaries.

The evaluator reads a directory of trained run outputs, rebuilds the matching
models, scores injected test files, and writes per-run event-metric summaries plus
an overall comparison CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from data import sonata as data
from data.loaders import WindowDataset
from data.preprocessing import align_columns, compute_train_stats, normalize_df
from evaluation.fpr_eval.eval_args import get_default_device
from evaluation.fpr_eval.eval_loops import MSELSTMForecaster, gaussian_nll_per_dim
from evaluation.fpr_eval.eval_models import load_gated_checkpoint
from models.forecasters import GaussianLSTMForecaster
from models.gated_detector import GatedAnomalyDetector
from models.marginal_flow import infer_marginal_type_from_ckpt, make_marginal_expert


DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Paper eval for Sonata synthetic data")
    p.add_argument("--data-dir", type=Path, required=True, help="Nominal training data root.")
    p.add_argument("--injected-dir", type=Path, required=True, help="Synthetic/injected test data root.")
    p.add_argument(
        "--aggregate-path",
        "--stage2-dir",
        dest="aggregate_path",
        type=Path,
        required=True,
        help="Base directory containing per-run model folders.",
    )
    p.add_argument(
        "--outdir",
        "--out-root",
        dest="outdir",
        type=Path,
        default=Path("results/sonata/synth_evaluation"),
        help="Root directory for per-run outputs and the overall summary CSV.",
    )
    p.add_argument(
        "--split-json",
        type=Path,
        default=None,
        help="Optional JSON defining explicit split metadata for runs.",
    )
    p.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Optional comma-separated run identifiers to evaluate.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument(
        "--run-type",
        choices=["all", "base", "precision"],
        default="all",
        help="Filter run folders by type for Temporal3Train-style layouts.",
    )
    p.add_argument("--anomaly-log", type=Path, default=None, help="Optional anomaly log used to filter successful injections.")
    p.add_argument(
        "--use-anomaly-log-filter",
        action="store_true",
        default=False,
        help="If set, only evaluate files marked successful in --anomaly-log.",
    )
    p.add_argument("--label-col", type=str, default="is_anomaly")
    p.add_argument("--models", type=str, default="gated,gaussian,mse")
    p.add_argument("--mode", choices=["linear", "mixture"], default=None)
    p.add_argument(
        "--gated-score",
        choices=["model_score", "forecast_only", "marginal_only", "oracle_window", "max_losses"],
        default="model_score",
    )
    p.add_argument("--gated-use-residual", action="store_true", default=False)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default=get_default_device())
    p.add_argument("--fpr-root", type=Path, default=None)
    p.add_argument("--skip-missing-baselines", action="store_true", default=True)
    p.add_argument("--no-skip-missing-baselines", dest="skip_missing_baselines", action="store_false")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_bool_flag(params: Dict[str, str], key: str, default: bool = False) -> bool:
    if key not in params:
        return default
    val = params.get(key)
    if val is None:
        return True
    if isinstance(val, bool):
        return val
    val = str(val).strip().lower()
    if val in ("", "1", "true", "yes", "y", "t"):
        return True
    if val in ("0", "false", "no", "n", "f"):
        return False
    return default


def _parse_int(params: Dict[str, str], key: str, default: int) -> int:
    val = params.get(key)
    if val is None or val == "":
        return default
    return int(float(val))


def _parse_float(params: Dict[str, str], key: str, default: float) -> float:
    val = params.get(key)
    if val is None or val == "":
        return default
    return float(val)


def _parse_models(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _wanted_runs(value: Optional[str]) -> set[str]:
    if not value:
        return set()
    return {v.strip() for v in value.split(",") if v.strip()}


def default_fpr_root(aggregate_path: Path) -> Path:
    if aggregate_path.name == "checkpoints":
        return aggregate_path.parent / "fpr_comparison"
    return aggregate_path / "fpr_comparison"


def load_splits(path: Path) -> List[Dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "splits" in raw:
        return raw["splits"]
    if isinstance(raw, dict) and "split" in raw:
        return [raw]
    raise ValueError("Unrecognized split JSON format.")


def load_feature_csv(path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Time(s)" in df.columns:
        df = df.sort_values(by="Time(s)", kind="mergesort")
    if label_col in df.columns:
        df = df.drop(columns=[label_col], errors="ignore")
    df = df.drop(columns=[c for c in data.META_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in data.IGNORED_FEATURES if c in df.columns], errors="ignore")
    if df.empty:
        return df
    return df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")


def load_labeled_csv(path: Path, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Time(s)" in df.columns:
        df = df.sort_values(by="Time(s)", kind="mergesort")
    labels = pd.to_numeric(df[label_col], errors="coerce")
    df = df.drop(columns=[label_col], errors="ignore")
    df = df.drop(columns=[c for c in data.META_COLS if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in data.IGNORED_FEATURES if c in df.columns], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    mask = df.notna().all(axis=1) & labels.notna()
    return df.loc[mask], (labels.loc[mask] > 0).astype(int)


def build_windows(df: pd.DataFrame, labels: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_arr = df.to_numpy(dtype=np.float32)
    label_arr = labels.to_numpy(dtype=np.int32)
    n_samples = data_arr.shape[0] - lookback
    if n_samples <= 0:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))
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


def calculate_event_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_score = pd.Series(y_score).rolling(window=5, center=True, min_periods=1).mean().values

    try:
        if np.unique(y_true).size < 2:
            auroc = 0.5
        else:
            auroc = roc_auc_score(y_true, y_score)
    except Exception:
        auroc = 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls + 1e-10
    f1_scores = numerator / denominator

    best_idx = int(np.argmax(f1_scores))
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = thresholds[-1]

    adj_pred = point_adjustment(y_score, y_true, best_threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, adj_pred, labels=[0, 1]).ravel()

    adj_prec = tp / (tp + fp + 1e-10)
    adj_rec = tp / (tp + fn + 1e-10)
    adj_f1 = 2 * (adj_prec * adj_rec) / (adj_prec + adj_rec + 1e-10)

    return {
        "auroc": float(auroc),
        "best_threshold": float(best_threshold),
        "adj_f1": float(adj_f1),
        "adj_precision": float(adj_prec),
        "adj_recall": float(adj_rec),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def load_anomaly_success(anomaly_log: Path) -> Dict[str, set[str]]:
    df = pd.read_csv(anomaly_log)
    if "filename" not in df.columns:
        raise ValueError("anomaly_log missing required column: filename")
    if "driver" not in df.columns:
        raise ValueError("anomaly_log missing required column: driver")

    if "anomaly" in df.columns:
        success = df[~df["anomaly"].astype(str).str.startswith("Injection_")]
    elif "status" in df.columns:
        success = df[df["status"].astype(str).str.lower().eq("success")]
    else:
        raise ValueError("anomaly_log missing required columns: anomaly or status")

    success_by_driver: Dict[str, set[str]] = {}
    for driver, group in success.groupby("driver"):
        success_by_driver[str(driver).upper()] = set(group["filename"].astype(str).tolist())
    return success_by_driver


def driver_from_path(path: Path) -> Optional[str]:
    if path.parent.name in {"A", "B", "C", "D"}:
        return path.parent.name
    try:
        return data.parse_driver_from_filename(path.name)
    except Exception:
        return None


def _load_summary(run_dir: Path) -> Dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_existing_summary(outdir: Path) -> Optional[List[Dict]]:
    summary_csv = outdir / "event_summary.csv"
    if summary_csv.exists():
        with summary_csv.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    results_csv = outdir / "event_evaluation_results.csv"
    if not results_csv.exists():
        return None
    df = pd.read_csv(results_csv)
    if df.empty:
        return None
    summary = df.groupby("model")[["auroc", "adj_f1"]].mean().reset_index()
    summary["n_files"] = df.groupby("model")["file"].count().values
    return summary.to_dict(orient="records")


def _write_summary_csv(outdir: Path, summary_rows: List[Dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "auroc", "adj_f1", "n_files"]
    with (outdir / "event_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_name_from_outdir(outdir: str) -> str:
    return Path(outdir).name


def _to_float(value: str) -> float:
    return float(value)


def build_paper_compare(rows: List[Dict[str, str]], wanted: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
    wanted_set = set(wanted)
    for row in rows:
        run = _run_name_from_outdir(row.get("outdir", ""))
        if wanted_set and run not in wanted_set and row.get("run_id", "") not in wanted_set:
            continue
        model = row.get("model", "")
        grouped.setdefault(run, {})[model] = row

    out: List[Dict[str, object]] = []
    for run in sorted(grouped.keys()):
        gated = grouped[run].get("gated")
        gaussian = grouped[run].get("gaussian")
        mse = grouped[run].get("mse")
        if gated is None or gaussian is None or mse is None:
            continue

        gated_auroc = _to_float(gated["auroc"])
        gaussian_auroc = _to_float(gaussian["auroc"])
        mse_auroc = _to_float(mse["auroc"])
        gated_f1 = _to_float(gated["adj_f1"])
        gaussian_f1 = _to_float(gaussian["adj_f1"])
        mse_f1 = _to_float(mse["adj_f1"])
        out.append(
            {
                "run_id": run,
                "gated_auroc": gated_auroc,
                "gaussian_auroc": gaussian_auroc,
                "mse_auroc": mse_auroc,
                "gated_minus_gaussian_auroc": gated_auroc - gaussian_auroc,
                "gated_minus_mse_auroc": gated_auroc - mse_auroc,
                "gated_adj_f1": gated_f1,
                "gaussian_adj_f1": gaussian_f1,
                "mse_adj_f1": mse_f1,
                "gated_minus_gaussian_adj_f1": gated_f1 - gaussian_f1,
                "gated_minus_mse_adj_f1": gated_f1 - mse_f1,
                "gated_best_auroc": gated_auroc > gaussian_auroc and gated_auroc > mse_auroc,
                "gated_best_adj_f1": gated_f1 > gaussian_f1 and gated_f1 > mse_f1,
            }
        )
    return out


def _resolve_fpr_curve(run_dir: Path, fpr_root: Path) -> Optional[Path]:
    candidates = [
        run_dir / "eval" / "fpr_curve.csv",
        fpr_root / run_dir.name / "fpr_curve.csv",
    ]
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = summary.get("run_id")
        if isinstance(run_id, str) and run_id:
            candidates.append(fpr_root / run_id / "fpr_curve.csv")
    for path in candidates:
        if path.exists():
            return path
    return None


def build_fpr_compare(
    aggregate_path: Path,
    fpr_root: Path,
    wanted: Sequence[str],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    wanted_set = set(wanted)
    run_dirs = sorted(path for path in aggregate_path.iterdir() if path.is_dir())
    for run_dir in run_dirs:
        fpr_path = _resolve_fpr_curve(run_dir, fpr_root)
        if fpr_path is None:
            continue

        run = run_dir.name
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run = summary.get("run_id") or run
        if wanted_set and run not in wanted_set and run_dir.name not in wanted_set:
            continue

        rows = _load_csv(fpr_path)
        by_model_quantile: Dict[Tuple[str, str], float] = {}
        for row in rows:
            by_model_quantile[(row["model"], str(row["quantile"]))] = _to_float(row["fpr_test"])

        quantiles = sorted({str(row["quantile"]) for row in rows}, key=float)
        for quantile in quantiles:
            gated = by_model_quantile.get(("gated", quantile))
            gaussian = by_model_quantile.get(("gaussian", quantile))
            mse = by_model_quantile.get(("mse", quantile))
            if gated is None or gaussian is None or mse is None:
                continue
            out.append(
                {
                    "run_id": run,
                    "quantile": quantile,
                    "gated_fpr": gated,
                    "gaussian_fpr": gaussian,
                    "mse_fpr": mse,
                    "gated_minus_gaussian_fpr": gated - gaussian,
                    "gated_minus_mse_fpr": gated - mse,
                    "gated_best_fpr": gated < gaussian and gated < mse,
                }
            )
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_compare_outputs(
    *,
    aggregate_path: Path,
    fpr_root: Path,
    normal_summary: Path,
    context_summary: Path,
    out_dir: Path,
    wanted_runs: Sequence[str],
) -> None:
    normal_rows = _load_csv(normal_summary)
    context_rows = _load_csv(context_summary)

    normal_cmp = build_paper_compare(normal_rows, wanted_runs)
    context_cmp = build_paper_compare(context_rows, wanted_runs)
    fpr_cmp = build_fpr_compare(aggregate_path, fpr_root, wanted_runs)

    write_csv(out_dir / "compare_fpr.csv", fpr_cmp)
    write_csv(out_dir / "compare_paper_normal.csv", normal_cmp)
    write_csv(out_dir / "compare_paper_contextual.csv", context_cmp)
    logging.info("Wrote comparison CSVs to %s", out_dir)


def _maybe_write_compare_outputs(args: argparse.Namespace) -> None:
    if args.outdir.name in {"normal", "contextual"}:
        compare_root = args.outdir.parent
    else:
        compare_root = args.outdir

    normal_summary = compare_root / "normal" / "stage2_synth_paper_eval_summary.csv"
    context_summary = compare_root / "contextual" / "stage2_synth_paper_eval_summary.csv"
    if not normal_summary.exists() or not context_summary.exists():
        return

    write_compare_outputs(
        aggregate_path=args.aggregate_path,
        fpr_root=args.fpr_root or default_fpr_root(args.aggregate_path),
        normal_summary=normal_summary,
        context_summary=context_summary,
        out_dir=compare_root / "compare",
        wanted_runs=sorted(_wanted_runs(args.runs)),
    )


def load_params(summary: Dict) -> Dict[str, str]:
    params = summary.get("params", {})
    return params if isinstance(params, dict) else {}


def get_run_identifier(run_dir: Path, summary: Optional[Dict] = None) -> str:
    info = summary if summary is not None else _load_summary(run_dir)
    run_id = info.get("run_id")
    if isinstance(run_id, str) and run_id:
        return run_id
    return run_dir.name


def resolve_gated_ckpt(run_dir: Path) -> Path:
    candidates = [
        run_dir / "gated_model.pt",
        run_dir / "checkpoints" / "gated_model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Gated checkpoint not found under {run_dir}")


def resolve_eval_ckpt(run_dir: Path, filename: str) -> Path:
    run_id = get_run_identifier(run_dir)
    candidates = [
        run_dir / "eval" / filename,
        run_dir.parent.parent / "fpr_comparison" / run_dir.name / filename,
        run_dir.parent.parent / "fpr_comparison" / run_id / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{filename} not found for run {run_id}")


def normalize_temporal_driver(name: str) -> str:
    if "_TR_" in name or "_VA_" in name:
        return name.split("_", 1)[0]
    return name


def normalize_driver_list(drivers: Sequence[str]) -> List[str]:
    return [normalize_temporal_driver(str(d).upper()) for d in drivers]


def build_eval_functions(
    run_dir: Path,
    summary: Dict,
    models: Sequence[str],
    device: torch.device,
    forced_mode: Optional[str],
    gated_score: str,
    gated_use_residual_override: bool,
    skip_missing_baselines: bool,
    feature_dim: int,
) -> Dict[str, callable]:
    params = load_params(summary)

    hidden_dim = _parse_int(params, "--hidden-dim", 128)
    num_layers = _parse_int(params, "--num-layers", 2)
    dropout = _parse_float(params, "--dropout", 0.1)
    include_gaussian_const = _parse_bool_flag(params, "--include-gaussian-const", default=False)
    use_mixture_nll = _parse_bool_flag(params, "--use-mixture-nll", default=True)
    mode = forced_mode or ("mixture" if use_mixture_nll else "linear")
    gate_use_residual = gated_use_residual_override or _parse_bool_flag(params, "--gate-use-residual", default=False)
    gate_use_residual = gate_use_residual or _parse_bool_flag(params, "--gated-use-residual", default=False)

    eval_fns: Dict[str, callable] = {}

    for model_name in models:
        if model_name == "gated":
            gated_ckpt = resolve_gated_ckpt(run_dir)
            ckpt_obj = torch.load(gated_ckpt, map_location="cpu")
            marginal_type, marginal_meta = infer_marginal_type_from_ckpt(ckpt_obj)
            model = GatedAnomalyDetector(
                input_dim=feature_dim,
                output_dim=feature_dim,
                marginal_expert=make_marginal_expert(
                    feature_dim,
                    marginal_type=marginal_type,
                    hidden_features=int(marginal_meta.get("marginal_hidden", 64)),
                    num_bins=int(marginal_meta.get("marginal_bins", 8)),
                    tail_bound=float(marginal_meta.get("marginal_tail", 10.0)),
                ),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                gate_use_residual=gate_use_residual,
                use_mixture_nll=(mode == "mixture"),
                include_gaussian_const=include_gaussian_const,
            ).float().to(device)
            load_gated_checkpoint(model, gated_ckpt, device)
            model.eval()

            def eval_batch(x: torch.Tensor, y: torch.Tensor, *, _model=model) -> torch.Tensor:
                out = _model(x, y)
                if gated_score == "model_score":
                    return out["score"]
                if gated_score == "forecast_only":
                    return out["forecast_loss"].mean(1)
                if gated_score == "marginal_only":
                    return out["marginal_loss_cal"].mean(1)
                if gated_score == "max_losses":
                    return torch.maximum(out["forecast_loss"].mean(1), out["marginal_loss_cal"].mean(1))
                return torch.minimum(out["forecast_loss"].mean(1), out["marginal_loss_cal"].mean(1))

            eval_fns["gated"] = eval_batch
            continue

        if model_name == "gaussian":
            try:
                ckpt = resolve_eval_ckpt(run_dir, "gaussian_lstm.pt")
            except FileNotFoundError:
                if skip_missing_baselines:
                    logging.warning("Missing gaussian checkpoint for %s", get_run_identifier(run_dir, summary))
                    continue
                raise
            model = GaussianLSTMForecaster(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=feature_dim,
            ).float().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            def eval_batch(x: torch.Tensor, y: torch.Tensor, *, _model=model) -> torch.Tensor:
                mu, sigma, log_sigma = _model(x)
                nll = gaussian_nll_per_dim(y, mu, sigma, log_sigma, include_gaussian_const)
                return nll.mean(dim=1)

            eval_fns["gaussian"] = eval_batch
            continue

        if model_name == "mse":
            try:
                ckpt = resolve_eval_ckpt(run_dir, "mse_lstm.pt")
            except FileNotFoundError:
                if skip_missing_baselines:
                    logging.warning("Missing mse checkpoint for %s", get_run_identifier(run_dir, summary))
                    continue
                raise
            model = MSELSTMForecaster(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                output_dim=feature_dim,
            ).float().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            def eval_batch(x: torch.Tensor, y: torch.Tensor, *, _model=model) -> torch.Tensor:
                mu = _model(x)
                return (mu - y).pow(2).mean(dim=1)

            eval_fns["mse"] = eval_batch
            continue

        raise ValueError(f"Unknown model: {model_name}")

    return eval_fns


def evaluate_run(
    run_dir: Path,
    summary: Dict,
    args: argparse.Namespace,
    outdir: Path,
    device: torch.device,
    success_by_driver: Optional[Dict[str, set[str]]],
) -> Dict:
    params = load_params(summary)
    run_id = get_run_identifier(run_dir, summary)

    lookback = _parse_int(params, "--lookback", 20) if args.lookback is None else args.lookback
    batch_size = _parse_int(params, "--batch-size", 128) if args.batch_size is None else args.batch_size
    num_workers = _parse_int(params, "--num-workers", 0) if args.num_workers is None else args.num_workers
    seed = _parse_int(params, "--seed", DEFAULT_SEED) if args.seed is None else args.seed

    train_drivers = normalize_driver_list(summary["split"]["train"])
    val_drivers = normalize_driver_list(summary["split"]["val"])
    test_drivers = normalize_driver_list(summary["split"]["test"])

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_files = data.list_files_for_drivers(args.data_dir, train_drivers)
    test_files = data.list_files_for_drivers(args.injected_dir, test_drivers)

    if success_by_driver is not None:
        filtered_files: List[Path] = []
        for path in test_files:
            driver = driver_from_path(path)
            if driver is None:
                continue
            success_files = success_by_driver.get(driver.upper(), set())
            if path.name in success_files:
                filtered_files.append(path)
        test_files = filtered_files

    train_dfs_raw: List[pd.DataFrame] = []
    base_cols: Optional[List[str]] = None
    for path in train_files:
        df = load_feature_csv(path, args.label_col)
        if df.empty:
            continue
        if base_cols is None:
            base_cols = list(df.columns)
        train_dfs_raw.append(align_columns(df, base_cols, path))

    if not train_dfs_raw or base_cols is None:
        raise ValueError(f"No usable training data for run {run_id}.")

    means, stds, _, feature_cols = compute_train_stats(train_dfs_raw)
    # Model hyperparameters are recovered from the saved run summary so the
    # evaluator does not need a separate hand-maintained config table.
    eval_fns = build_eval_functions(
        run_dir=run_dir,
        summary=summary,
        models=_parse_models(args.models),
        device=device,
        forced_mode=args.mode,
        gated_score=args.gated_score,
        gated_use_residual_override=args.gated_use_residual,
        skip_missing_baselines=args.skip_missing_baselines,
        feature_dim=len(feature_cols),
    )

    results: List[Dict[str, object]] = []
    for path in test_files:
        df, labels = load_labeled_csv(path, args.label_col)
        if df.empty:
            continue
        df = align_columns(df, base_cols, path)
        df = normalize_df(df, feature_cols, means, stds)

        x_test, y_test, y_labels = build_windows(df, labels, lookback)
        if y_labels.size == 0:
            continue

        loader = DataLoader(
            WindowDataset(x_test, y_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        for model_name, eval_fn in eval_fns.items():
            scores_list = []
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    scores_list.append(eval_fn(x, y).detach().cpu().numpy())
            y_scores = np.concatenate(scores_list)
            metrics = calculate_event_metrics(y_labels, y_scores)
            results.append(
                {
                    "model": model_name,
                    "file": path.name,
                    "run_id": run_id,
                    "train_drivers": ",".join(train_drivers),
                    "val_drivers": ",".join(val_drivers),
                    "test_drivers": ",".join(test_drivers),
                    **metrics,
                }
            )

    outdir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_csv = outdir / "event_evaluation_results.csv"
    results_df.to_csv(results_csv, index=False)

    summary_rows: List[Dict] = []
    if not results_df.empty:
        summary_df = results_df.groupby("model")[["auroc", "adj_f1"]].mean().reset_index()
        counts = results_df.groupby("model")["file"].count().reset_index().rename(columns={"file": "n_files"})
        summary_df = summary_df.merge(counts, on="model", how="left")
        summary_rows = summary_df.to_dict(orient="records")

    _write_summary_csv(outdir, summary_rows)
    return {
        "run_id": run_id,
        "outdir": str(outdir),
        "train_drivers": ",".join(train_drivers),
        "val_drivers": ",".join(val_drivers),
        "test_drivers": ",".join(test_drivers),
        "summary": summary_rows,
    }


def discover_run_dirs(args: argparse.Namespace) -> List[Tuple[Path, Dict]]:
    run_dirs = sorted([p for p in args.aggregate_path.iterdir() if p.is_dir()])
    summary_pairs: List[Tuple[Path, Dict]] = []
    for run_dir in run_dirs:
        try:
            summary = _load_summary(run_dir)
        except Exception:
            continue
        summary_pairs.append((run_dir, summary))

    if not summary_pairs:
        raise FileNotFoundError(f"No run directories with summary.json found in {args.aggregate_path}")

    if args.run_type == "base":
        filtered = [(p, s) for p, s in summary_pairs if "__v1_base_" in get_run_identifier(p, s)]
        if filtered:
            summary_pairs = filtered
    elif args.run_type == "precision":
        filtered = [(p, s) for p, s in summary_pairs if "__v2_precision_" in get_run_identifier(p, s)]
        if filtered:
            summary_pairs = filtered

    wanted = _wanted_runs(args.runs)
    if wanted:
        summary_pairs = [
            (p, s)
            for p, s in summary_pairs
            if p.name in wanted or get_run_identifier(p, s) in wanted
        ]

    if args.limit is not None:
        summary_pairs = summary_pairs[: args.limit]
    return summary_pairs


def build_split_runs(args: argparse.Namespace) -> List[Tuple[Path, Dict]]:
    lookup: Dict[str, Tuple[Path, Dict]] = {}
    for run_dir, summary in discover_run_dirs(args):
        lookup[run_dir.name] = (run_dir, summary)
        lookup[get_run_identifier(run_dir, summary)] = (run_dir, summary)

    splits = load_splits(args.split_json)
    wanted = _wanted_runs(args.runs)
    resolved: List[Tuple[Path, Dict]] = []
    for idx, entry in enumerate(splits):
        run_key = entry.get("run_id")
        if not run_key:
            raise ValueError("Each split entry must include 'run_id'.")
        if wanted and run_key not in wanted:
            continue
        if run_key not in lookup:
            raise FileNotFoundError(f"Run directory not found for split entry: {run_key}")
        run_dir, summary = lookup[run_key]
        split = entry.get("split") if isinstance(entry.get("split"), dict) else summary.get("split")
        if not isinstance(split, dict):
            raise ValueError(
                f"Missing explicit split metadata for Sonata run '{get_run_identifier(run_dir, summary)}'. "
                "Provide it in summary.json or via --split-json."
            )
        merged = dict(summary)
        merged["split"] = split
        resolved.append((run_dir, merged))
        if args.limit is not None and len(resolved) >= args.limit:
            break
    return resolved


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.aggregate_path.exists():
        raise FileNotFoundError(f"Aggregate path not found: {args.aggregate_path}")

    set_seed(args.seed)
    device = torch.device(args.device)
    success_by_driver: Optional[Dict[str, set[str]]] = None
    if args.use_anomaly_log_filter:
        if args.anomaly_log is None:
            raise ValueError("--use-anomaly-log-filter requires --anomaly-log")
        # The injected Sonata tree may contain failed/skipped injections. When
        # this flag is enabled, only successfully generated files are evaluated.
        success_by_driver = load_anomaly_success(args.anomaly_log)

    run_pairs = build_split_runs(args) if args.split_json is not None else discover_run_dirs(args)
    if not run_pairs:
        logging.warning("No runs found to process.")
        return

    summary_rows: List[Dict] = []
    for run_dir, summary in run_pairs:
        run_id = get_run_identifier(run_dir, summary)
        outdir = args.outdir / run_id
        if args.skip_existing and (outdir / "event_evaluation_results.csv").exists():
            logging.info("Skipping %s (existing results)", run_id)
            existing = _load_existing_summary(outdir)
            if existing:
                for row in existing:
                    summary_rows.append(
                        {
                            "run_id": "",
                            "model": row.get("model"),
                            "auroc": row.get("auroc"),
                            "adj_f1": row.get("adj_f1"),
                            "n_files": row.get("n_files"),
                            "train_drivers": ",".join(normalize_driver_list(summary["split"]["train"])),
                            "val_drivers": ",".join(normalize_driver_list(summary["split"]["val"])),
                            "test_drivers": ",".join(normalize_driver_list(summary["split"]["test"])),
                            "outdir": str(outdir),
                        }
                    )
            continue

        logging.info(
            "==> Evaluating %s | train=%s val=%s test=%s",
            run_id,
            normalize_driver_list(summary["split"]["train"]),
            normalize_driver_list(summary["split"]["val"]),
            normalize_driver_list(summary["split"]["test"]),
        )
        result = evaluate_run(
            run_dir=run_dir,
            summary=summary,
            args=args,
            outdir=outdir,
            device=device,
            success_by_driver=success_by_driver,
        )
        for row in result.get("summary", []):
            summary_rows.append(
                {
                    "run_id": "",
                    "model": row.get("model"),
                    "auroc": row.get("auroc"),
                    "adj_f1": row.get("adj_f1"),
                    "n_files": row.get("n_files"),
                    "train_drivers": result.get("train_drivers"),
                    "val_drivers": result.get("val_drivers"),
                    "test_drivers": result.get("test_drivers"),
                    "outdir": result.get("outdir"),
                }
            )

    if summary_rows:
        summary_csv = args.outdir / "stage2_synth_paper_eval_summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id",
            "model",
            "auroc",
            "adj_f1",
            "n_files",
            "train_drivers",
            "val_drivers",
            "test_drivers",
            "outdir",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        logging.info("Wrote summary CSV: %s", summary_csv)
        _maybe_write_compare_outputs(args)


if __name__ == "__main__":
    main()
