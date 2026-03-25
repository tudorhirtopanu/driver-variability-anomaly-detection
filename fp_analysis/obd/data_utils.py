"""OBD data loading and split preparation for nominal false-positive analysis."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fp_analysis.obd.config import DATA_DIR, IGNORED_FEATURES, META_COLS


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.replace("Â", ""))


def load_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_columns(df)

    if "Time" in df.columns:
        df = df.sort_values(by="Time", kind="mergesort")
    elif "Time(s)" in df.columns:
        df = df.sort_values(by="Time(s)", kind="mergesort")

    drop_cols = [c for c in META_COLS if c in df.columns]
    drop_cols += [c for c in IGNORED_FEATURES if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")
    return df.reset_index(drop=True)


def resolve_split_files(split_entry: Dict[str, object], data_dir: Path = DATA_DIR) -> Dict[str, List[Path]]:
    resolved: Dict[str, List[Path]] = {}
    for key in ("train", "val", "test"):
        names = split_entry.get(key, [])
        resolved[key] = [data_dir / name for name in names]
    return resolved


def prepare_split(split_entry: Dict[str, object], data_dir: Path = DATA_DIR) -> Dict[str, object]:
    resolved = resolve_split_files(split_entry, data_dir=data_dir)

    train_raw = []
    val_raw = []
    test_raw = []
    test_file_ids = []
    feature_names = None

    def load_array(path: Path) -> np.ndarray:
        nonlocal feature_names
        df = load_file(path)
        if feature_names is None:
            feature_names = df.columns.tolist()
        else:
            df = df.reindex(columns=feature_names)
        return df.values

    for path in resolved["train"]:
        train_raw.append(load_array(path))
    for path in resolved["val"]:
        val_raw.append(load_array(path))
    for path in resolved["test"]:
        test_raw.append(load_array(path))
        test_file_ids.append(path.name)

    if not train_raw or not val_raw or not test_raw:
        raise ValueError(
            f"Split id={split_entry.get('id')} does not contain non-empty train/val/test file lists."
        )

    train_concat = np.concatenate(train_raw, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_concat)

    def norm(arr: np.ndarray) -> np.ndarray:
        return scaler.transform(arr)

    return {
        "split_id": split_entry.get("id"),
        "train_files": [p.name for p in resolved["train"]],
        "val_files": [p.name for p in resolved["val"]],
        "test_files": [p.name for p in resolved["test"]],
        "train_seqs": [norm(arr) for arr in train_raw],
        "val_seqs": [norm(arr) for arr in val_raw],
        "test_seqs": [norm(arr) for arr in test_raw],
        "test_file_ids": test_file_ids,
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }
