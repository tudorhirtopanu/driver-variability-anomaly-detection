"""Run nominal false-positive analysis across the first N OBD file splits."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fp_analysis.obd.config import (
    DEFAULT_NUM_SPLITS,
    HUMAN_CONTROLLED_FEATURES,
    IGNORED_FEATURES,
    MODEL_TYPES,
    RANDOM_SEED,
    RESULTS_ROOT,
    SPLIT_JSON,
    set_global_seeds,
)


def load_splits(path: Path, num_splits: int) -> List[Dict[str, object]]:
    raw = json.loads(path.read_text())
    splits = raw.get("splits")
    if not isinstance(splits, list):
        raise ValueError(f"Unexpected split JSON structure: {path}")
    if num_splits < 1:
        raise ValueError("--num-splits must be >= 1")
    if num_splits > len(splits):
        raise ValueError(
            f"Requested {num_splits} splits, but {path} only contains {len(splits)}."
        )
    return splits[:num_splits]


def result_dir_name(model_type: str) -> str:
    return {
        "pca": "results_pca",
        "persistence": "results_persistence",
        "naive": "results_persistence",
        "lstm_forecast": "results_lstm_forecaster",
        "transformer_forecast": "results_transformer_forecaster",
        "lstm_vae": "results_lstm_vae",
        "lstm": "results_lstm_ae",
        "transformer": "results_transformer_ae",
        "tcn": "results_tcn",
        "usad": "results_usad",
    }[model_type]


def save_summaries(summaries: List[Dict[str, object]], model_type: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summaries_{model_type}_splits{len(summaries)}.json"
    with out_path.open("w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {len(summaries)} summaries to {out_path}")
    return out_path


def _nan_stats(values: List[float | None]) -> tuple[float | None, float | None]:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    if np.isnan(arr).all():
        return None, None
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def _mean_feature_array(summaries: List[Dict[str, object]], key: str) -> List[float] | None:
    rows = [np.array(s[key], dtype=float) for s in summaries if s.get(key) is not None]
    if not rows:
        return None
    return np.stack(rows, axis=0).mean(axis=0).tolist()


def aggregate_in_memory(summaries: List[Dict[str, object]]) -> Dict[str, object]:
    if not summaries:
        raise ValueError("No per-split summaries to aggregate.")

    feature_names = summaries[0]["feature_names"]
    split_ids = [s["split_id"] for s in summaries]
    fpr_test = np.array([s["fpr_test"] for s in summaries], dtype=float)

    human_fp_mean, human_fp_std = _nan_stats([s.get("human_fp_mean") for s in summaries])
    system_fp_mean, system_fp_std = _nan_stats([s.get("system_fp_mean") for s in summaries])
    human_nonfp_mean, human_nonfp_std = _nan_stats([s.get("human_nonfp_mean") for s in summaries])
    system_nonfp_mean, system_nonfp_std = _nan_stats([s.get("system_nonfp_mean") for s in summaries])
    human_inflation_mean, human_inflation_std = _nan_stats(
        [s.get("human_inflation") for s in summaries]
    )
    system_inflation_mean, system_inflation_std = _nan_stats(
        [s.get("system_inflation") for s in summaries]
    )

    return {
        "num_splits": len(summaries),
        "split_ids": split_ids,
        "feature_names": feature_names,
        "analysis_feature_names": summaries[0].get("analysis_feature_names", feature_names),
        "human_feature_names": summaries[0].get("human_feature_names", HUMAN_CONTROLLED_FEATURES),
        "system_feature_names": summaries[0].get("system_feature_names"),
        "ignored_features": summaries[0].get("ignored_features", IGNORED_FEATURES),
        "applied_ignored_features": summaries[0].get("applied_ignored_features", []),
        "fpr_test_mean": float(fpr_test.mean()),
        "fpr_test_std": float(fpr_test.std(ddof=0)),
        "n_test_windows_mean": float(np.mean([s["n_test_windows"] for s in summaries])),
        "n_fp_windows_mean": float(np.mean([s["n_fp_windows"] for s in summaries])),
        "n_nonfp_windows_mean": float(np.mean([s["n_nonfp_windows"] for s in summaries])),
        "valid_fp_splits": int(sum(s["n_fp_windows"] > 0 for s in summaries)),
        "human_fp_mean_mse": human_fp_mean,
        "human_fp_std_mse": human_fp_std,
        "system_fp_mean_mse": system_fp_mean,
        "system_fp_std_mse": system_fp_std,
        "human_nonfp_mean_mse": human_nonfp_mean,
        "human_nonfp_std_mse": human_nonfp_std,
        "system_nonfp_mean_mse": system_nonfp_mean,
        "system_nonfp_std_mse": system_nonfp_std,
        "human_inflation_mean": human_inflation_mean,
        "human_inflation_std": human_inflation_std,
        "system_inflation_mean": system_inflation_mean,
        "system_inflation_std": system_inflation_std,
        "mean_feature_fp_mse": _mean_feature_array(summaries, "fp_mean_mse"),
        "mean_feature_nonfp_mse": _mean_feature_array(summaries, "nonfp_mean_mse"),
    }


def save_aggregate(aggregate: Dict[str, object], model_type: str, num_splits: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"aggregate_{model_type}_splits{num_splits}.json"
    with out_path.open("w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Saved aggregate stats to {out_path}")
    return out_path


if __name__ == "__main__":
    alias_map = {
        "forecaster": "transformer_forecast",
        "lstm_seq2one": "lstm_forecast",
    }
    cli_choices = MODEL_TYPES + [m for m in alias_map if m not in MODEL_TYPES]

    parser = argparse.ArgumentParser(
        description="Run nominal-only false-positive analysis on the first N OBD splits."
    )
    parser.add_argument("--model", required=True, choices=cli_choices)
    parser.add_argument("--num-splits", type=int, default=DEFAULT_NUM_SPLITS)
    parser.add_argument("--split-json", type=Path, default=SPLIT_JSON)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to a model-specific folder under fp_analysis/obd.",
    )
    args = parser.parse_args()

    model_type_str = alias_map.get(args.model.lower(), args.model.lower())
    set_global_seeds(RANDOM_SEED)

    from fp_analysis.obd.evaluate import run_single_split

    splits = load_splits(args.split_json, args.num_splits)
    out_dir = args.out_dir or (RESULTS_ROOT / result_dir_name(model_type_str))

    summaries = []
    for idx, split_entry in enumerate(splits, start=1):
        print(f"\n--- Split {idx}/{len(splits)} ---")
        summaries.append(run_single_split(split_entry, model_type=model_type_str))

    save_summaries(summaries, model_type=model_type_str, out_dir=out_dir)
    aggregate = aggregate_in_memory(summaries)
    save_aggregate(aggregate, model_type=model_type_str, num_splits=len(summaries), out_dir=out_dir)
