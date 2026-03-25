#!/usr/bin/env python3

"""Main training entrypoint for the gated anomaly detector.

Training happens in stages:
1. pretrain the Gaussian forecaster;
2. pretrain the marginal expert on nominal targets;
3. train the gate with the experts frozen; and
4. optionally finetune all modules together.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from data.registry import get_dataset
from data.loaders import make_target_only_loaders, make_window_loaders
from train_args import parse_args
from training.artifacts import build_training_metadata, dump_inspection_artifacts
from training.gate_stats import (
    compute_alpha_prior,
    compute_gate_input_stats,
    compute_loss_stats,
)
from training.loops import (
    eval_forecaster_epoch,
    eval_gated_epoch,
    eval_marginal_epoch,
    freeze_module,
    train_forecaster_epoch,
    train_gated_epoch,
    train_marginal_epoch,
)

from models.gated_detector import GatedAnomalyDetector
from models.forecasters import GaussianLSTMForecaster
from split_utils import resolve_split_files


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


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Delay importing nflows-backed marginal code so --help works without loading
    # the full training stack.
    from models.marginal_flow import make_marginal_expert

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info("Args: %s", vars(args))

    # Dataset module chosen here. Everything else is the same.
    data = get_dataset(args.dataset)

    if (
        args.use_mixture_nll
        and args.train_calibration
        and not args.allow_calibration_with_mixture
    ):
        logging.warning(
            "Mixture NLL disables calibration by default; forcing train_calibration=False. "
            "Pass --allow-calibration-with-mixture to override."
        )
        args.train_calibration = False

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    pin_memory = device.type == "cuda"

    split_info = None
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

    # Normalization is fit on nominal training data only and then reused for
    # validation/test, which keeps the evaluation pipeline consistent.
    train_dfs_raw, base_cols = load_raw_train_dfs(data, train_files)
    rows_train = sum(len(df) for df in train_dfs_raw)
    means, stds, low_std_features, feature_cols = data.compute_train_stats(train_dfs_raw)
    if low_std_features:
        logging.info("Dropping low-std features: %s", ", ".join(low_std_features))

    train_dfs = [data.normalize_df(df, feature_cols, means, stds) for df in train_dfs_raw]
    val_dfs, rows_val = data.load_split_dfs(
        val_files, base_cols, feature_cols, means, stds, split_name="val"
    )
    test_dfs, rows_test = data.load_split_dfs(
        test_files, base_cols, feature_cols, means, stds, split_name="test"
    )

    x_train, y_train, n_train = data.build_windows_from_dfs(train_dfs, args.lookback)
    x_val, y_val, n_val = data.build_windows_from_dfs(val_dfs, args.lookback)
    x_test, y_test, n_test = data.build_windows_from_dfs(test_dfs, args.lookback)

    logging.info(
        "Samples train=%d val=%d test=%d | D=%d",
        n_train,
        n_val,
        n_test,
        x_train.shape[2],
    )

    training_meta = build_training_metadata(
        args=args,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        rows_train=rows_train,
        rows_val=rows_val,
        rows_test=rows_test,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        feature_cols=feature_cols,
        low_std_features=low_std_features,
        input_dim=x_train.shape[2],
        split_info=split_info,
    )

    # Window loaders are built after normalization so every stage sees the same
    # train/val/test tensors.
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

    # --- Stage 1: pretrain forecaster (Gaussian NLL) ---
    forecaster = GaussianLSTMForecaster(
        input_size=D,
        hidden_size=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=D,
    ).float().to(device)

    if args.pretrain_forecaster_epochs > 0:
        logging.info(
            "Pretraining forecaster for %d epochs...",
            args.pretrain_forecaster_epochs,
        )
        opt_f = torch.optim.Adam(forecaster.parameters(), lr=args.forecaster_lr)
        best_f_val = float("inf")
        best_f_state = None
        forecaster_history = []
        no_improve = 0

        for e in range(1, args.pretrain_forecaster_epochs + 1):
            tr = train_forecaster_epoch(
                forecaster,
                train_loader,
                opt_f,
                device,
                args.clip_grad,
                desc=f"f-tr {e}",
                include_const=args.include_gaussian_const,
            )
            va = eval_forecaster_epoch(
                forecaster,
                val_loader,
                device,
                desc=f"f-va {e}",
                include_const=args.include_gaussian_const,
            )
            logging.info("Forecaster epoch %d | train=%.6f val=%.6f", e, tr, va)
            forecaster_history.append({"epoch": e, "train": tr, "val": va})
            if va < best_f_val - args.early_stop_min_delta:
                best_f_val = va
                best_f_state = {
                    k: v.detach().cpu().clone()
                    for k, v in forecaster.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                logging.info(
                    "Early stopping forecaster at epoch %d (best val=%.6f).",
                    e,
                    best_f_val,
                )
                break

        if best_f_state is not None:
            forecaster.load_state_dict(best_f_state)
        torch.save(forecaster.state_dict(), args.outdir / "forecaster.pt")
        (args.outdir / "forecaster_history.json").write_text(
            json.dumps(forecaster_history, indent=2),
            encoding="utf-8",
        )
        logging.info("Saved forecaster.pt (best val=%.6f)", best_f_val)
    else:
        logging.info("Skipping forecaster pretraining (epochs=0).")
        if args.finetune_epochs == 0:
            logging.warning("Forecaster will remain untrained (pretrain=0, finetune=0).")

    # --- Stage 2: pretrain marginal expert ---
    marginal = make_marginal_expert(
        D,
        marginal_type=args.marginal_type,
        hidden_features=args.marginal_hidden,
        num_bins=args.marginal_bins,
        tail_bound=args.marginal_tail,
    ).float().to(device)

    if args.pretrain_marginal_epochs > 0:
        if not (0.0 < args.marginal_subsample <= 1.0):
            raise ValueError("--marginal-subsample must be in (0, 1].")

        y_train_marg = y_train
        y_val_marg = y_val
        if args.marginal_subsample < 1.0:
            rng = np.random.default_rng(args.seed)

            n_train_sub = max(1, int(len(y_train) * args.marginal_subsample))
            idx_train = rng.choice(len(y_train), size=n_train_sub, replace=False)
            y_train_marg = y_train[idx_train]

            n_val_sub = max(1, int(len(y_val) * args.marginal_subsample))
            idx_val = rng.choice(len(y_val), size=n_val_sub, replace=False)
            y_val_marg = y_val[idx_val]

            logging.info(
                "Marginal subsample: train %d/%d val %d/%d",
                len(y_train_marg),
                len(y_train),
                len(y_val_marg),
                len(y_val),
            )

        logging.info(
            "Pretraining marginal expert for %d epochs...",
            args.pretrain_marginal_epochs,
        )

        marginal_train_loader, marginal_val_loader = make_target_only_loaders(
            y_train=y_train_marg,
            y_val=y_val_marg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        opt_m = torch.optim.Adam(marginal.parameters(), lr=args.marginal_lr)
        best_m_val = float("inf")
        best_m_state = None
        marginal_history = []
        no_improve = 0

        for e in range(1, args.pretrain_marginal_epochs + 1):
            tr = train_marginal_epoch(
                marginal,
                marginal_train_loader,
                opt_m,
                device,
                args.clip_grad,
                desc=f"marg-tr {e}",
            )
            va = eval_marginal_epoch(
                marginal,
                marginal_val_loader,
                device,
                desc=f"marg-va {e}",
            )
            logging.info("Marginal epoch %d | train=%.6f val=%.6f", e, tr, va)
            marginal_history.append({"epoch": e, "train": tr, "val": va})
            if va < best_m_val - args.early_stop_min_delta:
                best_m_val = va
                best_m_state = {
                    k: v.detach().cpu().clone()
                    for k, v in marginal.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                logging.info(
                    "Early stopping marginal at epoch %d (best val=%.6f).",
                    e,
                    best_m_val,
                )
                break

        if best_m_state is not None:
            marginal.load_state_dict(best_m_state)
        torch.save(marginal.state_dict(), args.outdir / "marginal_expert.pt")
        (args.outdir / "marginal_history.json").write_text(
            json.dumps(marginal_history, indent=2),
            encoding="utf-8",
        )
        logging.info("Saved marginal_expert.pt (best val=%.6f)", best_m_val)
    else:
        logging.info("Skipping marginal pretraining (epochs=0).")
        if args.finetune_epochs == 0:
            logging.warning("Marginal expert will remain untrained (pretrain=0, finetune=0).")

    # --- Stage 3: gate-only training (experts frozen) ---
    model = GatedAnomalyDetector(
        input_dim=D,
        output_dim=D,
        marginal_expert=marginal,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        detach_gate_input=args.detach_gate_input,
        gate_l1=args.gate_l1,
        use_mixture_nll=args.use_mixture_nll,
        gate_aux_weight=args.gate_aux_weight,
        gate_aux_margin=args.gate_aux_margin,
        gate_aux_temp=args.gate_aux_temp,
        gate_prior_weight=args.gate_prior_weight,
        gate_use_residual=args.gate_use_residual,
        use_loss_standardization=args.gate_loss_standardize,
        include_gaussian_const=args.include_gaussian_const,
        marginal_b_l2=args.marginal_b_l2,
    ).float().to(device)

    model.forecaster.load_state_dict(forecaster.state_dict())

    logging.info("Computing gate input stats...")
    # The gate can use uncertainty and optionally residual magnitude as inputs;
    # these are standardized from the nominal validation distribution.
    stats_in = compute_gate_input_stats(model.forecaster, val_loader, device)
    if args.gate_use_residual:
        model.set_gate_input_stats(
            stats_in["logsig_mean"],
            stats_in["logsig_std"],
            stats_in["resid_mean"],
            stats_in["resid_std"],
        )
    else:
        model.set_gate_input_stats(
            stats_in["logsig_mean"],
            stats_in["logsig_std"],
        )

    if args.gate_loss_standardize or args.gate_prior_weight > 0.0:
        logging.info("Computing loss stats for gate...")
        marginal_a = None
        marginal_b = None
        if not args.use_mixture_nll:
            marginal_a = model.marginal_a
            marginal_b = model.marginal_b
        stats = compute_loss_stats(
            model.forecaster,
            model.marginal_expert,
            val_loader,
            device,
            include_const=args.include_gaussian_const,
            marginal_a=marginal_a,
            marginal_b=marginal_b,
        )
        if args.gate_loss_standardize:
            model.set_loss_stats(
                stats["forecast_mean"],
                stats["forecast_std"],
                stats["marginal_mean"],
                stats["marginal_std"],
            )
        if args.gate_prior_weight > 0.0:
            # The prior nudges the gate toward the expert that looks better on
            # nominal validation data before any gate gradients are applied.
            alpha_prior = compute_alpha_prior(
                stats["gap_mean"],
                stats["gap_std"],
                margin=args.gate_prior_margin,
                temp=args.gate_prior_temp,
            )
            model.set_gate_prior(alpha_prior)
            model.gate.set_bias_from_prior(alpha_prior)
            logging.info(
                "Gate prior alpha set (mean=%.3f min=%.3f max=%.3f).",
                alpha_prior.mean().item(),
                alpha_prior.min().item(),
                alpha_prior.max().item(),
            )

    logging.info("Running gate output sanity check...")
    sanity_checked = False
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)
        if out["alpha"].shape != y.shape or out["score_per_feature"].shape != y.shape:
            raise ValueError(
                "Sanity check failed: unexpected output shape for gate or scores."
            )
        if not torch.isfinite(out["score_per_feature"]).all():
            raise ValueError(
                "Sanity check failed: non-finite score_per_feature detected."
            )
        sanity_checked = True
        break
    if not sanity_checked:
        raise ValueError("Sanity check failed: val_loader produced no batches.")

    freeze_module(model.forecaster, freeze=True)
    freeze_module(model.marginal_expert, freeze=True)
    freeze_module(model.gate, freeze=False)
    model.marginal_a.requires_grad = args.train_calibration
    model.marginal_b.requires_grad = args.train_calibration

    logging.info(
        "Gate-only training for %d epochs (train_calibration=%s)...",
        args.gate_epochs,
        args.train_calibration,
    )

    gate_params = [p for p in model.parameters() if p.requires_grad]
    if not gate_params:
        raise ValueError("No trainable parameters for gate-only stage.")
    optimizer = torch.optim.Adam(gate_params, lr=args.gate_lr)

    best_gate_val = float("inf")
    best_gate_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    gate_history = []
    no_improve = 0

    if args.gate_epochs > 0:
        for epoch in range(1, args.gate_epochs + 1):
            tr = train_gated_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.clip_grad,
                desc=f"gate {epoch}",
                experts_eval=True,
            )
            va = eval_gated_epoch(model, val_loader, device, desc=f"gate-val {epoch}")
            logging.info(
                "Gate epoch %d/%d | train loss=%.6f (f=%.6f m=%.6f a=%.3f g=%.4f p=%.4f) | "
                "val loss=%.6f (f=%.6f m=%.6f a=%.3f g=%.4f p=%.4f)",
                epoch,
                args.gate_epochs,
                tr["loss"],
                tr["forecast"],
                tr["marginal"],
                tr["alpha"],
                tr["gate_aux"],
                tr["gate_prior"],
                va["loss"],
                va["forecast"],
                va["marginal"],
                va["alpha"],
                va["gate_aux"],
                va["gate_prior"],
            )
            gate_history.append(
                {
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in tr.items()},
                    **{f"val_{k}": v for k, v in va.items()},
                }
            )

            if va["loss"] < best_gate_val - args.early_stop_min_delta:
                best_gate_val = va["loss"]
                best_gate_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                logging.info(
                    "Early stopping gate at epoch %d (best val=%.6f).",
                    epoch,
                    best_gate_val,
                )
                break
    else:
        logging.info("Skipping gate-only training (epochs=0).")

    (args.outdir / "gate_history.json").write_text(
        json.dumps(gate_history, indent=2),
        encoding="utf-8",
    )

    if best_gate_state is not None:
        model.load_state_dict(best_gate_state)

    gate_only_meta = {
        **training_meta,
        "stage": "gate_only",
        "metrics": {"best_gate_val": float(best_gate_val)},
    }
    torch.save(
        {"state_dict": model.state_dict(), "meta": gate_only_meta},
        args.outdir / "gated_model_gate_only.pt",
    )
    logging.info("Saved gated_model_gate_only.pt (best val=%.6f)", best_gate_val)

    dump_inspection_artifacts(
        model,
        val_loader,
        device,
        args.outdir,
        prefix="val_gateonly",
        feature_names=feature_cols,
    )
    dump_inspection_artifacts(
        model,
        test_loader,
        device,
        args.outdir,
        prefix="test_gateonly",
        feature_names=feature_cols,
    )

    # --- Stage 4: optional finetune (unfreeze all, small expert LR) ---
    if args.finetune_epochs > 0:
        logging.info("Finetuning all modules for %d epochs...", args.finetune_epochs)
        freeze_module(model.forecaster, freeze=False)
        freeze_module(model.marginal_expert, freeze=False)
        freeze_module(model.gate, freeze=False)
        model.marginal_a.requires_grad = True
        model.marginal_b.requires_grad = True

        gate_params = list(model.gate.parameters()) + [model.marginal_a, model.marginal_b]
        expert_params = list(model.forecaster.parameters()) + list(model.marginal_expert.parameters())
        optimizer = torch.optim.Adam(
            [
                {"params": gate_params, "lr": args.finetune_lr},
                # A smaller LR on the pretrained experts helps preserve the
                # staged solution while still allowing mild joint adjustment.
                {"params": expert_params, "lr": args.finetune_expert_lr},
            ]
        )

        best_ft_val = float("inf")
        best_ft_state = None
        finetune_history = []
        no_improve = 0

        for epoch in range(1, args.finetune_epochs + 1):
            tr = train_gated_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.clip_grad,
                desc=f"ft {epoch}",
            )
            va = eval_gated_epoch(model, val_loader, device, desc=f"ft-val {epoch}")
            logging.info(
                "Finetune epoch %d/%d | train loss=%.6f (f=%.6f m=%.6f a=%.3f g=%.4f p=%.4f) | "
                "val loss=%.6f (f=%.6f m=%.6f a=%.3f g=%.4f p=%.4f)",
                epoch,
                args.finetune_epochs,
                tr["loss"],
                tr["forecast"],
                tr["marginal"],
                tr["alpha"],
                tr["gate_aux"],
                tr["gate_prior"],
                va["loss"],
                va["forecast"],
                va["marginal"],
                va["alpha"],
                va["gate_aux"],
                va["gate_prior"],
            )
            finetune_history.append(
                {
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in tr.items()},
                    **{f"val_{k}": v for k, v in va.items()},
                }
            )

            if va["loss"] < best_ft_val - args.early_stop_min_delta:
                best_ft_val = va["loss"]
                best_ft_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                logging.info(
                    "Early stopping finetune at epoch %d (best val=%.6f).",
                    epoch,
                    best_ft_val,
                )
                break

        (args.outdir / "finetune_history.json").write_text(
            json.dumps(finetune_history, indent=2),
            encoding="utf-8",
        )

        if best_ft_state is not None:
            model.load_state_dict(best_ft_state)

        final_meta = {
            **training_meta,
            "stage": "finetune",
            "metrics": {
                "best_gate_val": float(best_gate_val),
                "best_finetune_val": float(best_ft_val),
            },
        }
        torch.save(
            {"state_dict": model.state_dict(), "meta": final_meta},
            args.outdir / "gated_model.pt",
        )
        logging.info("Saved gated_model.pt (best val=%.6f)", best_ft_val)
    else:
        final_meta = {
            **training_meta,
            "stage": "no_finetune",
            "metrics": {"best_gate_val": float(best_gate_val)},
        }
        torch.save(
            {"state_dict": model.state_dict(), "meta": final_meta},
            args.outdir / "gated_model.pt",
        )
        logging.info("Saved gated_model.pt (no finetune).")

    dump_inspection_artifacts(
        model,
        val_loader,
        device,
        args.outdir,
        prefix="val_final",
        feature_names=feature_cols,
    )
    dump_inspection_artifacts(
        model,
        test_loader,
        device,
        args.outdir,
        prefix="test_final",
        feature_names=feature_cols,
    )

    te = eval_gated_epoch(model, test_loader, device, desc="test")
    logging.info(
        "Test | loss=%.6f forecast=%.6f marginal=%.6f alpha=%.3f gate_aux=%.4f gate_prior=%.4f",
        te["loss"],
        te["forecast"],
        te["marginal"],
        te["alpha"],
        te["gate_aux"],
        te["gate_prior"],
    )
    (args.outdir / "test_metrics.json").write_text(
        json.dumps(te, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
