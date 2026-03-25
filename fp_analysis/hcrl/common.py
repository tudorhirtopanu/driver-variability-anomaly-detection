"""Shared evaluation helpers for the HCRL false-positive experiments."""

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
    all_mse = []
    with torch.no_grad():
        for batch in loader:
            all_mse.append(feature_batch_fn(batch).detach().cpu().numpy())
    if not all_mse:
        return np.empty((0, 0), dtype=float)
    return np.concatenate(all_mse, axis=0)


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
        sq_err = (x_hat - batch) ** 2
        return sq_err.mean(dim=(1, 2))

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
        se = (batch - x_hat) ** 2
        return se.mean(dim=1)

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
        se = (y_hat - y) ** 2
        return se.mean(dim=1)

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


def fpr_per_driver(
    scores: np.ndarray,
    driver_ids_per_window: np.ndarray,
    tau: float,
) -> Dict[str, float]:
    fprs = {}
    unique_drivers = sorted(set(driver_ids_per_window.tolist()))
    for driver in unique_drivers:
        mask = driver_ids_per_window == driver
        if mask.sum() == 0:
            fprs[driver] = 0.0
        else:
            fprs[driver] = float((scores[mask] > tau).mean())
    return fprs


def _label(prefix: str, text: str) -> str:
    return f"{prefix} {text}" if prefix else text


def summarize_fp_analysis(
    *,
    feature_names: Sequence[str],
    human_controlled_features: Sequence[str],
    seen_scores: np.ndarray,
    unseen_scores: np.ndarray,
    seen_mse_per_win: np.ndarray,
    unseen_mse_per_win: np.ndarray,
    tau: float,
    label_prefix: str = "",
    error_label: str = "reconstruction",
) -> Dict[str, object]:
    D = len(feature_names)

    seen_fp_mask = seen_scores > tau
    unseen_fp_mask = unseen_scores > tau

    print(f"\nFalse positive windows (score > tau={tau:.6f}):")
    print(f"  Seen   test : {seen_fp_mask.sum()} / {len(seen_fp_mask)}")
    print(f"  Unseen test : {unseen_fp_mask.sum()} / {len(unseen_fp_mask)}")

    if seen_fp_mask.any():
        seen_fp_mse = seen_mse_per_win[seen_fp_mask]
        per_var_seen_fp = seen_fp_mse.mean(axis=0)
        seen_fp_share = (
            seen_fp_mse / (seen_fp_mse.sum(axis=1, keepdims=True) + 1e-8)
        ).mean(axis=0)
    else:
        per_var_seen_fp = np.zeros(D, dtype=float)
        seen_fp_share = np.zeros(D, dtype=float)

    seen_nonfp_mask = ~seen_fp_mask
    if seen_nonfp_mask.any():
        seen_nonfp_mse = seen_mse_per_win[seen_nonfp_mask]
        per_var_seen_nonfp = seen_nonfp_mse.mean(axis=0)
        seen_nonfp_share = (
            seen_nonfp_mse / (seen_nonfp_mse.sum(axis=1, keepdims=True) + 1e-8)
        ).mean(axis=0)
    else:
        per_var_seen_nonfp = np.zeros(D, dtype=float)
        seen_nonfp_share = np.zeros(D, dtype=float)

    if unseen_fp_mask.any():
        unseen_fp_mse = unseen_mse_per_win[unseen_fp_mask]
        per_var_unseen_fp = unseen_fp_mse.mean(axis=0)
        unseen_fp_share = (
            unseen_fp_mse / (unseen_fp_mse.sum(axis=1, keepdims=True) + 1e-8)
        ).mean(axis=0)
    else:
        per_var_unseen_fp = np.zeros(D, dtype=float)
        unseen_fp_share = np.zeros(D, dtype=float)

    unseen_nonfp_mask = ~unseen_fp_mask
    if unseen_nonfp_mask.any():
        unseen_nonfp_mse = unseen_mse_per_win[unseen_nonfp_mask]
        per_var_unseen_nonfp = unseen_nonfp_mse.mean(axis=0)
        unseen_nonfp_share = (
            unseen_nonfp_mse / (unseen_nonfp_mse.sum(axis=1, keepdims=True) + 1e-8)
        ).mean(axis=0)
    else:
        per_var_unseen_nonfp = np.zeros(D, dtype=float)
        unseen_nonfp_share = np.zeros(D, dtype=float)

    diff_fp = per_var_unseen_fp - per_var_seen_fp

    human_delta = None
    non_human_delta = None

    human_seen_fp_mean = None
    nonhuman_seen_fp_mean = None
    human_unseen_fp_mean = None
    nonhuman_unseen_fp_mean = None

    human_seen_nonfp_mean = None
    nonhuman_seen_nonfp_mean = None
    human_unseen_nonfp_mean = None
    nonhuman_unseen_nonfp_mean = None

    human_seen_share = None
    nonhuman_seen_share = None
    human_unseen_share = None
    nonhuman_unseen_share = None

    human_seen_nonfp_share = None
    nonhuman_seen_nonfp_share = None
    human_unseen_nonfp_share = None
    nonhuman_unseen_nonfp_share = None

    if human_controlled_features:
        human_idx = [i for i, name in enumerate(feature_names) if name in human_controlled_features]
        non_human_idx = [i for i, name in enumerate(feature_names) if name not in human_controlled_features]

        if human_idx and non_human_idx:
            human_seen_fp_mean = float(per_var_seen_fp[human_idx].mean())
            nonhuman_seen_fp_mean = float(per_var_seen_fp[non_human_idx].mean())
            human_unseen_fp_mean = float(per_var_unseen_fp[human_idx].mean())
            nonhuman_unseen_fp_mean = float(per_var_unseen_fp[non_human_idx].mean())

            human_seen_nonfp_mean = float(per_var_seen_nonfp[human_idx].mean())
            nonhuman_seen_nonfp_mean = float(per_var_seen_nonfp[non_human_idx].mean())
            human_unseen_nonfp_mean = float(per_var_unseen_nonfp[human_idx].mean())
            nonhuman_unseen_nonfp_mean = float(per_var_unseen_nonfp[non_human_idx].mean())

            human_delta = float(diff_fp[human_idx].mean())
            non_human_delta = float(diff_fp[non_human_idx].mean())

            human_seen_share = float(seen_fp_share[human_idx].sum())
            nonhuman_seen_share = float(seen_fp_share[non_human_idx].sum())
            human_unseen_share = float(unseen_fp_share[human_idx].sum())
            nonhuman_unseen_share = float(unseen_fp_share[non_human_idx].sum())

            human_seen_nonfp_share = float(seen_nonfp_share[human_idx].sum())
            nonhuman_seen_nonfp_share = float(seen_nonfp_share[non_human_idx].sum())
            human_unseen_nonfp_share = float(unseen_nonfp_share[human_idx].sum())
            nonhuman_unseen_nonfp_share = float(unseen_nonfp_share[non_human_idx].sum())

            print(f"\n{_label(label_prefix, 'Grouped mean MSE on test windows (FP vs non-FP):')}")
            print("  Seen / Human     FP     : {:.4f}".format(human_seen_fp_mean))
            print("  Seen / Human     non-FP : {:.4f}".format(human_seen_nonfp_mean))
            print("  Seen / Non-human FP     : {:.4f}".format(nonhuman_seen_fp_mean))
            print("  Seen / Non-human non-FP : {:.4f}".format(nonhuman_seen_nonfp_mean))

            print("  Unseen / Human   FP     : {:.4f}".format(human_unseen_fp_mean))
            print("  Unseen / Human   non-FP : {:.4f}".format(human_unseen_nonfp_mean))
            print("  Unseen / Non-human FP   : {:.4f}".format(nonhuman_unseen_fp_mean))
            print("  Unseen / Non-human non-FP : {:.4f}".format(nonhuman_unseen_nonfp_mean))

            print(f"\n{_label(label_prefix, 'Grouped error share on test windows (FP vs non-FP):')}")
            print("  Seen / Human     FP     : {:.4f}".format(human_seen_share))
            print("  Seen / Human     non-FP : {:.4f}".format(human_seen_nonfp_share))
            print("  Seen / Non-human FP     : {:.4f}".format(nonhuman_seen_share))
            print("  Seen / Non-human non-FP : {:.4f}".format(nonhuman_seen_nonfp_share))

            print("  Unseen / Human   FP     : {:.4f}".format(human_unseen_share))
            print("  Unseen / Human   non-FP : {:.4f}".format(human_unseen_nonfp_share))
            print("  Unseen / Non-human FP   : {:.4f}".format(nonhuman_unseen_share))
            print("  Unseen / Non-human non-FP : {:.4f}".format(nonhuman_unseen_nonfp_share))

            print(f"\n{_label(label_prefix, 'Grouped ΔMSE on FPs (Unseen - Seen, test):')}")
            print(f"  Human-controlled features mean ΔMSE    : {human_delta:.4f}")
            print(f"  Non-human features mean ΔMSE          : {non_human_delta:.4f}")

    sorted_idx = np.argsort(diff_fp)[::-1]
    top_k = min(10, len(sorted_idx))
    top_features = []

    print(f"\n{_label(label_prefix, 'Top features by ΔMSE on FPs (Unseen - Seen, test):')}")
    print(f"  (Higher ΔMSE means unseen drivers have larger FP {error_label} error)")
    for rank in range(top_k):
        idx = int(sorted_idx[rank])
        delta = float(diff_fp[idx])
        name = feature_names[idx]
        print(f"  {rank+1:2d}. {name:30s} ΔMSE = {delta: .4f}")
        top_features.append(
            {
                "rank": rank + 1,
                "feature": name,
                "delta_mse": delta,
            }
        )

    return {
        "fp_seen_mean_mse": per_var_seen_fp.tolist(),
        "fp_unseen_mean_mse": per_var_unseen_fp.tolist(),
        "human_seen_fp_mean": human_seen_fp_mean,
        "nonhuman_seen_fp_mean": nonhuman_seen_fp_mean,
        "human_unseen_fp_mean": human_unseen_fp_mean,
        "nonhuman_unseen_fp_mean": nonhuman_unseen_fp_mean,
        "human_seen_nonfp_mean": human_seen_nonfp_mean,
        "nonhuman_seen_nonfp_mean": nonhuman_seen_nonfp_mean,
        "human_unseen_nonfp_mean": human_unseen_nonfp_mean,
        "nonhuman_unseen_nonfp_mean": nonhuman_unseen_nonfp_mean,
        "human_seen_share": human_seen_share,
        "nonhuman_seen_share": nonhuman_seen_share,
        "human_unseen_share": human_unseen_share,
        "nonhuman_unseen_share": nonhuman_unseen_share,
        "human_seen_nonfp_share": human_seen_nonfp_share,
        "nonhuman_seen_nonfp_share": nonhuman_seen_nonfp_share,
        "human_unseen_nonfp_share": human_unseen_nonfp_share,
        "nonhuman_unseen_nonfp_share": nonhuman_unseen_nonfp_share,
        "diff_fp": diff_fp.tolist(),
        "feature_names": list(feature_names),
        "human_delta": float(human_delta) if human_delta is not None else None,
        "non_human_delta": float(non_human_delta) if non_human_delta is not None else None,
        "top_features": top_features,
    }


def run_split_evaluation(
    *,
    train_drivers: List[str],
    unseen_drivers: List[str],
    model_type,
    device: torch.device,
    batch_size: int,
    val_quantile: float,
    human_controlled_features: Sequence[str],
    prepare_split_fn: Callable[..., Dict[str, object]],
    train_fn: Callable[[List[np.ndarray], List[np.ndarray], object], Tuple[torch.nn.Module, Dict[str, float]]],
    build_val_dataset_fn: Callable[[List[np.ndarray]], object],
    build_test_dataset_with_ids_fn: Callable[[List[np.ndarray], List[str]], Tuple[object, np.ndarray]],
    score_fn: Callable[[torch.nn.Module, object], np.ndarray],
    feature_mse_fn: Callable[[torch.nn.Module, object], np.ndarray],
    label_prefix: str = "",
    error_label: str = "reconstruction",
) -> Dict[str, object]:
    print("=" * 80)
    print(f"Running split ({model_type}): train_drivers={train_drivers}, unseen_drivers={unseen_drivers}")

    split = prepare_split_fn(train_drivers=train_drivers, unseen_drivers=unseen_drivers)

    train_seqs = split["train_seqs"]
    val_seqs = split["val_seqs"]
    seen_test_seqs = split["seen_test_seqs"]
    unseen_test_seqs = split["unseen_test_seqs"]
    seen_test_driver_ids = split["seen_test_driver_ids"]
    unseen_test_driver_ids = split["unseen_test_driver_ids"]
    feature_names = split["feature_names"]

    print(f"Number of features: D={len(feature_names)}")
    print(f"Num train seqs       : {len(train_seqs)}")
    print(f"Num val seqs         : {len(val_seqs)}")
    print(f"Num seen TEST seqs   : {len(seen_test_seqs)}")
    print(f"Num unseen TEST seqs : {len(unseen_test_seqs)}")

    model, stats = train_fn(train_seqs, val_seqs, model_type=model_type)
    model.to(device)

    print("Training stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    val_ds = build_val_dataset_fn(val_seqs)
    seen_test_ds, seen_test_driver_ids_win = build_test_dataset_with_ids_fn(
        seen_test_seqs, seen_test_driver_ids
    )
    unseen_test_ds, unseen_test_driver_ids_win = build_test_dataset_with_ids_fn(
        unseen_test_seqs, unseen_test_driver_ids
    )

    print("Windows per condition:")
    print(f"  Val (seen)      : {len(val_ds)}")
    print(f"  Test / Seen     : {len(seen_test_ds)}")
    print(f"  Test / Unseen   : {len(unseen_test_ds)}")

    scores_val = score_fn(model, val_ds)
    scores_seen_test = score_fn(model, seen_test_ds)
    scores_unseen_test = score_fn(model, unseen_test_ds)

    tau = float(np.quantile(scores_val, val_quantile))
    print(f"Chosen tau (quantile={val_quantile} of val scores): {tau:.6f}")

    fpr_seen = false_positive_rate(scores_seen_test, tau)
    fpr_unseen = false_positive_rate(scores_unseen_test, tau)
    print("False Positive Rates (test):")
    print(f"  Seen drivers   : {fpr_seen:.4f}")
    print(f"  Unseen drivers : {fpr_unseen:.4f}")

    print("\nPer-driver FPRs on Test:")
    seen_fprs = fpr_per_driver(scores_seen_test, seen_test_driver_ids_win, tau)
    for driver, value in sorted(seen_fprs.items()):
        print(f"  Seen   driver {driver}: {value:.4f}")
    unseen_fprs = fpr_per_driver(scores_unseen_test, unseen_test_driver_ids_win, tau)
    for driver, value in sorted(unseen_fprs.items()):
        print(f"  Unseen driver {driver}: {value:.4f}")

    seen_mse_per_win = feature_mse_fn(model, seen_test_ds)
    unseen_mse_per_win = feature_mse_fn(model, unseen_test_ds)
    summary = summarize_fp_analysis(
        feature_names=feature_names,
        human_controlled_features=human_controlled_features,
        seen_scores=scores_seen_test,
        unseen_scores=scores_unseen_test,
        seen_mse_per_win=seen_mse_per_win,
        unseen_mse_per_win=unseen_mse_per_win,
        tau=tau,
        label_prefix=label_prefix,
        error_label=error_label,
    )

    return {
        "model_type": model_type,
        "train_drivers": list(train_drivers),
        "unseen_drivers": list(unseen_drivers),
        "tau": float(tau),
        "fpr_seen": float(fpr_seen),
        "fpr_unseen": float(fpr_unseen),
        **summary,
    }
