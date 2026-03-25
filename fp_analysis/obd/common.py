"""Shared helpers for nominal-only false-positive analysis on OBD splits."""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _collect_score_batches(
    dataset,
    batch_size: int,
    score_batch_fn: Callable[[object], torch.Tensor],
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    scores = []
    with torch.no_grad():
        for batch in loader:
            scores.append(score_batch_fn(batch).detach().cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=float)
    return np.concatenate(scores, axis=0)


def _collect_feature_batches(
    dataset,
    batch_size: int,
    feature_batch_fn: Callable[[object], torch.Tensor],
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    rows = []
    with torch.no_grad():
        for batch in loader:
            rows.append(feature_batch_fn(batch).detach().cpu().numpy())
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.concatenate(rows, axis=0)


def collect_reconstruction_scores(
    model: torch.nn.Module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    def score_batch(batch) -> torch.Tensor:
        batch = batch.to(device)
        x_hat = model(batch)
        return ((x_hat - batch) ** 2).mean(dim=(1, 2))

    return _collect_score_batches(dataset, batch_size, score_batch)


def collect_reconstruction_feature_mse(
    model: torch.nn.Module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    def feature_batch(batch) -> torch.Tensor:
        batch = batch.to(device)
        x_hat = model(batch)
        return ((batch - x_hat) ** 2).mean(dim=1)

    return _collect_feature_batches(dataset, batch_size, feature_batch)


def collect_forecast_scores(
    model: torch.nn.Module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    def score_batch(batch) -> torch.Tensor:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        return ((y_hat - y) ** 2).mean(dim=1)

    return _collect_score_batches(dataset, batch_size, score_batch)


def collect_forecast_feature_mse(
    model: torch.nn.Module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    def feature_batch(batch) -> torch.Tensor:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        return (y_hat - y) ** 2

    return _collect_feature_batches(dataset, batch_size, feature_batch)


def false_positive_rate(scores: np.ndarray, tau: float) -> float:
    if scores.size == 0:
        return 0.0
    return float((scores > tau).mean())


def fpr_per_file(scores: np.ndarray, file_ids_per_window: np.ndarray, tau: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for file_id in sorted(set(file_ids_per_window.tolist())):
        mask = file_ids_per_window == file_id
        out[file_id] = float((scores[mask] > tau).mean()) if mask.any() else 0.0
    return out


def _safe_mean(values: np.ndarray, indices: Sequence[int]) -> float | None:
    if len(indices) == 0 or values.size == 0:
        return None
    return float(values[list(indices)].mean())


def _safe_sum(values: np.ndarray, indices: Sequence[int]) -> float | None:
    if len(indices) == 0 or values.size == 0:
        return None
    return float(values[list(indices)].sum())


def summarize_nominal_fp_analysis(
    *,
    feature_names: Sequence[str],
    human_controlled_features: Sequence[str],
    ignored_features: Sequence[str],
    test_scores: np.ndarray,
    test_mse_per_win: np.ndarray,
    tau: float,
    label_prefix: str = "",
    error_label: str = "reconstruction",
) -> Dict[str, object]:
    D = len(feature_names)
    fp_mask = test_scores > tau
    nonfp_mask = ~fp_mask

    print(f"\nFalse positive windows (score > tau={tau:.6f}): {fp_mask.sum()} / {len(fp_mask)}")

    fp_mse = test_mse_per_win[fp_mask] if fp_mask.any() else np.empty((0, D), dtype=float)
    nonfp_mse = (
        test_mse_per_win[nonfp_mask] if nonfp_mask.any() else np.empty((0, D), dtype=float)
    )

    per_var_fp = fp_mse.mean(axis=0) if fp_mse.size else np.zeros(D, dtype=float)
    per_var_nonfp = nonfp_mse.mean(axis=0) if nonfp_mse.size else np.zeros(D, dtype=float)

    fp_share = (
        (fp_mse / (fp_mse.sum(axis=1, keepdims=True) + 1e-8)).mean(axis=0)
        if fp_mse.size
        else np.zeros(D, dtype=float)
    )
    nonfp_share = (
        (nonfp_mse / (nonfp_mse.sum(axis=1, keepdims=True) + 1e-8)).mean(axis=0)
        if nonfp_mse.size
        else np.zeros(D, dtype=float)
    )

    ignored_features = list(ignored_features)
    ignored_set = set(ignored_features)
    human_set = set(human_controlled_features)
    human_idx = [
        i for i, name in enumerate(feature_names) if name in human_set and name not in ignored_set
    ]
    system_idx = [
        i
        for i, name in enumerate(feature_names)
        if name not in human_set and name not in ignored_set
    ]
    ignored_idx = [i for i, name in enumerate(feature_names) if name in ignored_set]
    analysis_idx = [i for i, name in enumerate(feature_names) if name not in ignored_set]

    human_fp_mean = _safe_mean(per_var_fp, human_idx) if fp_mse.size else None
    system_fp_mean = _safe_mean(per_var_fp, system_idx) if fp_mse.size else None
    human_nonfp_mean = _safe_mean(per_var_nonfp, human_idx) if nonfp_mse.size else None
    system_nonfp_mean = _safe_mean(per_var_nonfp, system_idx) if nonfp_mse.size else None

    human_fp_share = _safe_sum(fp_share, human_idx) if fp_mse.size else None
    system_fp_share = _safe_sum(fp_share, system_idx) if fp_mse.size else None
    human_nonfp_share = _safe_sum(nonfp_share, human_idx) if nonfp_mse.size else None
    system_nonfp_share = _safe_sum(nonfp_share, system_idx) if nonfp_mse.size else None

    human_inflation = (
        float(human_fp_mean / human_nonfp_mean)
        if human_fp_mean is not None and human_nonfp_mean not in (None, 0.0)
        else None
    )
    system_inflation = (
        float(system_fp_mean / system_nonfp_mean)
        if system_fp_mean is not None and system_nonfp_mean not in (None, 0.0)
        else None
    )

    delta = per_var_fp - per_var_nonfp if fp_mse.size and nonfp_mse.size else np.zeros(D, dtype=float)
    sorted_idx = sorted(analysis_idx, key=lambda idx: float(delta[idx]), reverse=True)
    top_features = []
    if fp_mse.size and nonfp_mse.size:
        print(f"\n{label_prefix}Grouped mean MSE on nominal test windows (FP vs non-FP):")
        if ignored_idx:
            ignored_names = ", ".join(feature_names[idx] for idx in ignored_idx)
            print(f"  Ignored features : {ignored_names}")
        elif ignored_features:
            ignored_names = ", ".join(ignored_features)
            print(f"  Ignored features : {ignored_names} (not present after loading)")
        print(f"  Human  FP     : {human_fp_mean:.4f}" if human_fp_mean is not None else "  Human  FP     : n/a")
        print(
            f"  Human  non-FP : {human_nonfp_mean:.4f}"
            if human_nonfp_mean is not None
            else "  Human  non-FP : n/a"
        )
        print(f"  System FP     : {system_fp_mean:.4f}" if system_fp_mean is not None else "  System FP     : n/a")
        print(
            f"  System non-FP : {system_nonfp_mean:.4f}"
            if system_nonfp_mean is not None
            else "  System non-FP : n/a"
        )
        print(
            f"  Human inflation  : {human_inflation:.4f}"
            if human_inflation is not None
            else "  Human inflation  : n/a"
        )
        print(
            f"  System inflation : {system_inflation:.4f}"
            if system_inflation is not None
            else "  System inflation : n/a"
        )

        print(f"\n{label_prefix}Top features by FP inflation on nominal test windows:")
        for rank, idx in enumerate(sorted_idx[: min(10, len(sorted_idx))], start=1):
            item = {
                "rank": rank,
                "feature": feature_names[int(idx)],
                "delta_mse": float(delta[int(idx)]),
                "fp_mean_mse": float(per_var_fp[int(idx)]),
                "nonfp_mean_mse": float(per_var_nonfp[int(idx)]),
            }
            top_features.append(item)
            print(
                f"  {rank:2d}. {item['feature']:40s} "
                f"delta={item['delta_mse']:.4f} "
                f"fp={item['fp_mean_mse']:.4f} "
                f"nonfp={item['nonfp_mean_mse']:.4f}"
            )

    return {
        "n_test_windows": int(len(test_scores)),
        "n_fp_windows": int(fp_mask.sum()),
        "n_nonfp_windows": int(nonfp_mask.sum()),
        "fp_mean_mse": per_var_fp.tolist() if fp_mse.size else None,
        "nonfp_mean_mse": per_var_nonfp.tolist() if nonfp_mse.size else None,
        "fp_share": fp_share.tolist() if fp_mse.size else None,
        "nonfp_share": nonfp_share.tolist() if nonfp_mse.size else None,
        "feature_names": list(feature_names),
        "analysis_feature_names": [feature_names[idx] for idx in analysis_idx],
        "human_feature_names": [feature_names[idx] for idx in human_idx],
        "system_feature_names": [feature_names[idx] for idx in system_idx],
        "ignored_features": ignored_features,
        "applied_ignored_features": [feature_names[idx] for idx in ignored_idx],
        "human_fp_mean": human_fp_mean,
        "system_fp_mean": system_fp_mean,
        "human_nonfp_mean": human_nonfp_mean,
        "system_nonfp_mean": system_nonfp_mean,
        "human_fp_share": human_fp_share,
        "system_fp_share": system_fp_share,
        "human_nonfp_share": human_nonfp_share,
        "system_nonfp_share": system_nonfp_share,
        "human_inflation": human_inflation,
        "system_inflation": system_inflation,
        "top_features": top_features,
    }


def run_split_evaluation(
    *,
    split_entry: Dict[str, object],
    model_type,
    device: torch.device,
    batch_size: int,
    val_quantile: float,
    human_controlled_features: Sequence[str],
    ignored_features: Sequence[str],
    prepare_split_fn: Callable[[Dict[str, object]], Dict[str, object]],
    train_fn: Callable[[List[np.ndarray], List[np.ndarray], object], Tuple[torch.nn.Module, Dict[str, float]]],
    build_val_dataset_fn: Callable[[List[np.ndarray]], object],
    build_test_dataset_with_ids_fn: Callable[[List[np.ndarray], List[str]], Tuple[object, np.ndarray]],
    score_fn: Callable[[torch.nn.Module, object], np.ndarray],
    feature_mse_fn: Callable[[torch.nn.Module, object], np.ndarray],
    label_prefix: str = "",
    error_label: str = "reconstruction",
) -> Dict[str, object]:
    split = prepare_split_fn(split_entry)
    print("=" * 80)
    print(f"Running split id={split['split_id']} ({model_type})")
    print(f"  train files: {len(split['train_files'])}")
    print(f"  val files  : {len(split['val_files'])}")
    print(f"  test files : {len(split['test_files'])}")

    train_seqs = split["train_seqs"]
    val_seqs = split["val_seqs"]
    test_seqs = split["test_seqs"]
    test_file_ids = split["test_file_ids"]
    feature_names = split["feature_names"]

    model, stats = train_fn(train_seqs, val_seqs, model_type=model_type)
    model.to(device)

    print("Training stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    val_ds = build_val_dataset_fn(val_seqs)
    test_ds, test_file_ids_win = build_test_dataset_with_ids_fn(test_seqs, test_file_ids)

    print(f"Number of features: D={len(feature_names)}")
    print(f"Val windows  : {len(val_ds)}")
    print(f"Test windows : {len(test_ds)}")

    scores_val = score_fn(model, val_ds)
    scores_test = score_fn(model, test_ds)

    tau = float(np.quantile(scores_val, val_quantile))
    fpr_test = false_positive_rate(scores_test, tau)

    print(f"Chosen tau (quantile={val_quantile} of val scores): {tau:.6f}")
    print(f"Nominal test FPR: {fpr_test:.4f}")

    test_fprs = fpr_per_file(scores_test, test_file_ids_win, tau)
    print("\nPer-file FPRs on nominal test:")
    for file_id, value in test_fprs.items():
        print(f"  {file_id}: {value:.4f}")

    test_mse_per_win = feature_mse_fn(model, test_ds)
    summary = summarize_nominal_fp_analysis(
        feature_names=feature_names,
        human_controlled_features=human_controlled_features,
        ignored_features=ignored_features,
        test_scores=scores_test,
        test_mse_per_win=test_mse_per_win,
        tau=tau,
        label_prefix=label_prefix,
        error_label=error_label,
    )

    return {
        "model_type": model_type,
        "split_id": split["split_id"],
        "train_files": split["train_files"],
        "val_files": split["val_files"],
        "test_files": split["test_files"],
        "tau": tau,
        "fpr_test": fpr_test,
        "per_file_fpr": test_fprs,
        **summary,
    }
