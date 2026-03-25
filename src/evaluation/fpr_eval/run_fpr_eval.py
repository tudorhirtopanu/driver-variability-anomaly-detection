#!/usr/bin/env python3

"""Nominal false-positive-rate evaluation for gated, Gaussian, and MSE models."""

from __future__ import annotations

import logging

import torch

from data.loaders import make_window_loaders
from data.registry import get_dataset
from .eval_args import parse_args, parse_quantiles
from .eval_loops import (
    collect_scores_gated,
    collect_scores_gaussian,
    collect_scores_mse,
    compute_fpr,
)
from .eval_models import (
    build_and_load_gated,
    load_raw_train_dfs,
    train_or_load_gaussian,
    train_or_load_mse,
    write_fpr_outputs,
)
from split_utils import resolve_split_files


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Args: %s", vars(args))

    data = get_dataset(args.dataset)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    pin_memory = device.type == "cuda"

    if args.split_json is not None:
        split_info = resolve_split_files(args.split_json, args.split_id, args.data_dir)
        train_files = split_info["train"]
        val_files = split_info["val"]
        test_files = split_info["test"]
        logging.info(
            "Using split id %s from %s",
            split_info.get("split_id"),
            args.split_json,
        )
    else:
        if not args.train_drivers or not args.val_drivers or not args.test_drivers:
            raise ValueError(
                "Provide --train-drivers/--val-drivers/--test-drivers or "
                "--split-json/--split-id."
            )
        train_drivers = data.parse_driver_list(args.train_drivers)
        val_drivers = data.parse_driver_list(args.val_drivers)
        test_drivers = data.parse_driver_list(args.test_drivers)

        train_files = data.list_files_for_drivers(args.data_dir, train_drivers)
        val_files = data.list_files_for_drivers(args.data_dir, val_drivers)
        test_files = data.list_files_for_drivers(args.data_dir, test_drivers)

    # Normalization is always anchored to the nominal training data. The same
    # statistics are then applied to validation and test, which mirrors the
    # actual deployment setting for these FPR checks.
    train_dfs_raw, base_cols = load_raw_train_dfs(data, train_files)
    means, stds, low_std_features, feature_cols = data.compute_train_stats(train_dfs_raw)
    if low_std_features:
        logging.info("Dropping low-std features: %s", ", ".join(low_std_features))

    train_dfs = [data.normalize_df(df, feature_cols, means, stds) for df in train_dfs_raw]
    val_dfs, _ = data.load_split_dfs(
        val_files, base_cols, feature_cols, means, stds, split_name="val"
    )
    test_dfs, _ = data.load_split_dfs(
        test_files, base_cols, feature_cols, means, stds, split_name="test"
    )

    x_train, y_train, n_train = data.build_windows_from_dfs(train_dfs, args.lookback)
    x_val, y_val, n_val = data.build_windows_from_dfs(val_dfs, args.lookback)
    x_test, y_test, n_test = data.build_windows_from_dfs(test_dfs, args.lookback)

    logging.info("Samples train=%d val=%d test=%d | D=%d", n_train, n_val, n_test, x_train.shape[2])

    train_loader, val_loader, test_loader = make_window_loaders(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    D = x_train.shape[2]

    gaussian = train_or_load_gaussian(args, D, train_loader, val_loader, device)
    mse = train_or_load_mse(args, D, train_loader, val_loader, device)
    gated = build_and_load_gated(args, D, device)

    quantiles = parse_quantiles(args.quantiles)
    logging.info("Quantiles: %s", quantiles)

    scores = {}
    # Thresholds are calibrated on nominal validation scores and then measured
    # on the held-out nominal test scores.
    scores["gated_val"] = collect_scores_gated(gated, val_loader, device)
    scores["gated_test"] = collect_scores_gated(gated, test_loader, device)
    scores["gaussian_val"] = collect_scores_gaussian(
        gaussian, val_loader, device, args.include_gaussian_const
    )
    scores["gaussian_test"] = collect_scores_gaussian(
        gaussian, test_loader, device, args.include_gaussian_const
    )
    scores["mse_val"] = collect_scores_mse(mse, val_loader, device)
    scores["mse_test"] = collect_scores_mse(mse, test_loader, device)

    summary = {
        "quantiles": quantiles,
        # All three models see the same val/test windows, so the gated counts are
        # used as the canonical sample counts in the summary file.
        "n_val": int(scores["gated_val"].shape[0]),
        "n_test": int(scores["gated_test"].shape[0]),
        "models": {},
    }

    summary["models"]["gated"] = compute_fpr(scores["gated_val"], scores["gated_test"], quantiles)
    summary["models"]["gaussian"] = compute_fpr(scores["gaussian_val"], scores["gaussian_test"], quantiles)
    summary["models"]["mse"] = compute_fpr(scores["mse_val"], scores["mse_test"], quantiles)

    write_fpr_outputs(args.outdir, summary, quantiles)


if __name__ == "__main__":
    main()
