# run_experiments.py
"""
python run_experiments.py --model lstm
python run_experiments.py --model transformer
python run_experiments.py --model transformer_forecast
python run_experiments.py --model lstm_forecast   # aka LSTM seq2one
python run_experiments.py --model lstm_vae
python run_experiments.py --model pca
python run_experiments.py --model tcn
python run_experiments.py --model persistence
python run_experiments.py --model usad
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fp_analysis.hcrl.config import (
    SPLITS_DIR,
    RESULTS_DIR,
    MODEL_TYPES,
    NUM_SEEN_OPTIONS,
    NUM_SPLITS_PER_CONFIG,
    RANDOM_SEED,
    set_global_seeds,
)
from fp_analysis.shared.model_factory import ModelType


def load_splits(num_seen: int, path: Path) -> List[Dict[str, List[str]]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["splits"][str(num_seen)]


def save_summaries(summaries, model_type: str, num_seen: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"summaries_{model_type}_seen{num_seen}_splits{len(summaries)}.json"

    # make sure everything is JSON-serialisable
    safe = []
    for s in summaries:
        s_safe = {}
        for k, v in s.items():
            if isinstance(v, np.ndarray):
                s_safe[k] = v.tolist()
            else:
                s_safe[k] = v
        safe.append(s_safe)

    with open(out_path, "w") as f:
        json.dump(safe, f, indent=2)
    print(f"Saved {len(summaries)} summaries to {out_path}")
    return out_path


def aggregate_in_memory(
    summaries,
    num_seen: int,
    num_splits: int,
) -> dict:
    """
    Compute aggregate stats (FPR & human/non-human MSE + shares)
    directly from the in-memory list of per-split summaries.
    """

    if len(summaries) == 0:
        raise ValueError("No summaries to aggregate.")

    feature_names = summaries[0]["feature_names"]

    # -----------------------------
    # 1) FPRs per split
    # -----------------------------
    fpr_seen = np.array([s["fpr_seen"] for s in summaries], dtype=float)
    fpr_unseen = np.array([s["fpr_unseen"] for s in summaries], dtype=float)

    # -----------------------------
    # 2) Group-level FP MSE per split
    # -----------------------------
    human_seen_fp = np.array([s["human_seen_fp_mean"] for s in summaries], dtype=float)
    nonhuman_seen_fp = np.array([s["nonhuman_seen_fp_mean"] for s in summaries], dtype=float)
    human_unseen_fp = np.array([s["human_unseen_fp_mean"] for s in summaries], dtype=float)
    nonhuman_unseen_fp = np.array([s["nonhuman_unseen_fp_mean"] for s in summaries], dtype=float)

    # -----------------------------
    # 3) Group-level NON-FP MSE per split
    # -----------------------------
    human_seen_nonfp = np.array([s["human_seen_nonfp_mean"] for s in summaries], dtype=float)
    nonhuman_seen_nonfp = np.array([s["nonhuman_seen_nonfp_mean"] for s in summaries], dtype=float)
    human_unseen_nonfp = np.array([s["human_unseen_nonfp_mean"] for s in summaries], dtype=float)
    nonhuman_unseen_nonfp = np.array([s["nonhuman_unseen_nonfp_mean"] for s in summaries], dtype=float)

    # -----------------------------
    # 4) Group-level FP error shares per split
    # -----------------------------
    human_seen_share = np.array([s["human_seen_share"] for s in summaries], dtype=float)
    nonhuman_seen_share = np.array([s["nonhuman_seen_share"] for s in summaries], dtype=float)
    human_unseen_share = np.array([s["human_unseen_share"] for s in summaries], dtype=float)
    nonhuman_unseen_share = np.array([s["nonhuman_unseen_share"] for s in summaries], dtype=float)

    # -----------------------------
    # 5) Group-level NON-FP error shares per split
    # -----------------------------
    human_seen_nonfp_share = np.array([s["human_seen_nonfp_share"] for s in summaries], dtype=float)
    nonhuman_seen_nonfp_share = np.array([s["nonhuman_seen_nonfp_share"] for s in summaries], dtype=float)
    human_unseen_nonfp_share = np.array([s["human_unseen_nonfp_share"] for s in summaries], dtype=float)
    nonhuman_unseen_nonfp_share = np.array([s["nonhuman_unseen_nonfp_share"] for s in summaries], dtype=float)

    # -----------------------------
    # 6) Feature-wise FP MSE (averaged across splits) – for plots
    # -----------------------------
    fp_seen_all = np.stack(
        [np.array(s["fp_seen_mean_mse"], dtype=float) for s in summaries],
        axis=0,
    )  # (num_splits, D)
    fp_unseen_all = np.stack(
        [np.array(s["fp_unseen_mean_mse"], dtype=float) for s in summaries],
        axis=0,
    )

    mean_seen_feature_mse = fp_seen_all.mean(axis=0).tolist()
    mean_unseen_feature_mse = fp_unseen_all.mean(axis=0).tolist()

    # -----------------------------
    # 7) Build aggregate dict
    # -----------------------------
    agg = {
        "num_seen": int(num_seen),
        "num_splits": int(num_splits),
        "feature_names": feature_names,

        # FPR aggregates
        "fpr_seen_mean": float(fpr_seen.mean()),
        "fpr_seen_std": float(fpr_seen.std(ddof=0)),
        "fpr_unseen_mean": float(fpr_unseen.mean()),
        "fpr_unseen_std": float(fpr_unseen.std(ddof=0)),

        # FP MSE aggregates (group-level)
        "human_seen_mean_mse": float(human_seen_fp.mean()),
        "nonhuman_seen_mean_mse": float(nonhuman_seen_fp.mean()),
        "human_unseen_mean_mse": float(human_unseen_fp.mean()),
        "nonhuman_unseen_mean_mse": float(nonhuman_unseen_fp.mean()),

        # NON-FP MSE aggregates (group-level)
        "human_seen_nonfp_mean_mse": float(human_seen_nonfp.mean()),
        "nonhuman_seen_nonfp_mean_mse": float(nonhuman_seen_nonfp.mean()),
        "human_unseen_nonfp_mean_mse": float(human_unseen_nonfp.mean()),
        "nonhuman_unseen_nonfp_mean_mse": float(nonhuman_unseen_nonfp.mean()),

        # FP error shares
        "human_seen_share_mean": float(human_seen_share.mean()),
        "nonhuman_seen_share_mean": float(nonhuman_seen_share.mean()),
        "human_unseen_share_mean": float(human_unseen_share.mean()),
        "nonhuman_unseen_share_mean": float(nonhuman_unseen_share.mean()),

        # NON-FP error shares
        "human_seen_nonfp_share_mean": float(human_seen_nonfp_share.mean()),
        "nonhuman_seen_nonfp_share_mean": float(nonhuman_seen_nonfp_share.mean()),
        "human_unseen_nonfp_share_mean": float(human_unseen_nonfp_share.mean()),
        "nonhuman_unseen_nonfp_share_mean": float(nonhuman_unseen_nonfp_share.mean()),

        # Feature-wise FP MSE averages 
        "mean_seen_feature_mse": mean_seen_feature_mse,
        "mean_unseen_feature_mse": mean_unseen_feature_mse,
    }
    return agg



def save_aggregate_to_json(
    aggregate: dict,
    model_type: str,
    num_seen: int,
    num_splits: int,
    out_dir: str = "results",
) -> Path:
    """
    Save aggregate dict to JSON with a nice filename.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = out_path / f"aggregate_{model_type}_seen{num_seen}_splits{num_splits}.json"

    with open(filename, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"Saved aggregate stats to {filename}")
    return filename


if __name__ == "__main__":
    alias_map = {
        "forecaster": "transformer_forecast",
        "lstm_seq2one": "lstm_forecast",
    }
    cli_choices = MODEL_TYPES + [m for m in alias_map if m not in MODEL_TYPES]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=cli_choices,
        required=True,
        help="Which model to run (reconstruction or forecasting)",
    )
    args = parser.parse_args()
    user_choice = args.model.lower()
    model_type_str = alias_map.get(user_choice, user_choice)

    set_global_seeds(RANDOM_SEED)

    from fp_analysis.hcrl.evaluate import run_single_split

    if model_type_str == "lstm":
        model_type = "lstm"
    elif model_type_str == "transformer":
        model_type = "transformer"
    elif model_type_str == "transformer_forecast":
        model_type = "transformer_forecast"
    elif model_type_str == "lstm_forecast":
        model_type = "lstm_forecast"
    elif model_type_str == "lstm_vae":
        model_type = "lstm_vae"
    elif model_type_str == "pca":
        model_type = "pca"
    elif model_type_str == "tcn":
        model_type = "tcn"
    elif model_type_str in ("persistence", "naive"):
        model_type = "persistence"
    elif model_type_str == "usad":
        model_type = "usad"
    else:
        raise ValueError(f"Unknown model type: {model_type_str}")


    splits_path = SPLITS_DIR / "splits_config.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits config not found at {splits_path}. Run scripts/generate_fp_splits.sh hcrl first."
        )

    for num_seen in NUM_SEEN_OPTIONS:
        print(f"\n=== {model_type_str.upper()} | {num_seen}/{10-num_seen} splits "
              f"({NUM_SPLITS_PER_CONFIG} splits each) ===")

        split_cfgs = load_splits(num_seen, splits_path)
        summaries = []
        for split_idx, cfg in enumerate(split_cfgs):
            train_drivers = cfg["train_drivers"]
            unseen_drivers = cfg["unseen_drivers"]
            print(f"\n--- Split {split_idx+1}/{len(split_cfgs)} ---")
            summary = run_single_split(train_drivers, unseen_drivers, model_type=model_type)
            summaries.append(summary)

        # 1) Save per-split summaries
        save_summaries(summaries, model_type=model_type_str, num_seen=num_seen)

        # 2) Compute aggregate metrics in-memory
        aggregate = aggregate_in_memory(
            summaries,
            num_seen=num_seen,
            num_splits=len(summaries),
        )

        # 3) Save aggregate JSON with human/non-human + FPR stats
        save_aggregate_to_json(
            aggregate,
            model_type=model_type_str,
            num_seen=num_seen,
            num_splits=len(summaries),
        )
