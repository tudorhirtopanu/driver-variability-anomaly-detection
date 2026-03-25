"""Shared preprocessing utilities for time-series CSV datasets."""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

STD_EPS = 1e-8


def load_and_clean_csv(
    path: Path,
    meta_cols: Sequence[str],
    ignored_features: Sequence[str],
    *,
    sort_cols: Sequence[str] = ("Time(s)",),
    preprocess_df: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Load CSV, (optionally) preprocess dataframe, sort by the first available time column,
    drop meta/ignored columns, coerce numeric values, and drop NaNs.

    This matches the per-dataset loaders when called with:
      - HCRL/Sonata: sort_cols=("Time(s)",), preprocess_df=None
      - OBD: sort_cols=("Time", "Time(s)"), preprocess_df=_normalize_columns
    """
    df = pd.read_csv(path)

    if preprocess_df is not None:
        df = preprocess_df(df)

    # Sort by the first time column that exists (stable mergesort).
    for c in sort_cols:
        if c in df.columns:
            df = df.sort_values(by=c, kind="mergesort")
            break

    df = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")
    df = df.drop(columns=[c for c in ignored_features if c in df.columns], errors="ignore")

    if df.empty:
        return df

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")
    return df


def align_columns(df: pd.DataFrame, expected_cols: Sequence[str], path: Path) -> pd.DataFrame:
    """
    Ensure the dataframe has exactly the expected set of columns (no missing/extra),
    then return it re-ordered to match expected_cols.
    """
    expected_set = set(expected_cols)
    found_set = set(df.columns)

    if expected_set != found_set:
        missing = sorted(expected_set - found_set)
        extra = sorted(found_set - expected_set)
        details = []
        if missing:
            details.append(f"missing: {missing}")
        if extra:
            details.append(f"extra: {extra}")
        detail_str = "; ".join(details)
        raise ValueError(f"Column mismatch in {path}: {detail_str}")

    return df.loc[:, list(expected_cols)]


def compute_train_stats(
    train_dfs: Sequence[pd.DataFrame],
) -> Tuple[pd.Series, pd.Series, List[str], List[str]]:
    """
    Compute per-feature mean/std from concatenated training data, then drop features
    with near-zero std (< STD_EPS). Returns:
        means (filtered), stds (filtered), low_std_features, feature_cols
    """
    train_all = pd.concat(train_dfs, ignore_index=True)
    means = train_all.mean()
    stds = train_all.std(ddof=0)

    # Near-constant features destabilize normalization and carry little value.
    low_std_features = sorted(stds[stds < STD_EPS].index.tolist())
    feature_cols = [c for c in train_all.columns if c not in low_std_features]
    if not feature_cols:
        raise ValueError("All features were dropped due to near-zero std.")

    return means.loc[feature_cols], stds.loc[feature_cols], low_std_features, feature_cols


def normalize_df(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    means: pd.Series,
    stds: pd.Series,
) -> pd.DataFrame:
    """Z-score normalize (using training stats) on feature_cols."""
    df = df.loc[:, feature_cols]
    return (df - means) / (stds + STD_EPS)


def load_split_dfs(
    files: Sequence[Path],
    *,
    base_cols: Sequence[str],
    feature_cols: Sequence[str],
    means: pd.Series,
    stds: pd.Series,
    split_name: str,
    meta_cols: Sequence[str],
    ignored_features: Sequence[str],
    sort_cols: Sequence[str] = ("Time(s)",),
    preprocess_df: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Tuple[List[pd.DataFrame], int]:
    """
    Load, clean, align, normalize a list of CSVs. Returns (dfs, total_rows).
    """
    dfs: List[pd.DataFrame] = []
    total_rows = 0

    for path in files:
        df = load_and_clean_csv(
            path,
            meta_cols=meta_cols,
            ignored_features=ignored_features,
            sort_cols=sort_cols,
            preprocess_df=preprocess_df,
        )
        if df.empty:
            logging.warning("No rows after cleaning in %s", path.name)
            continue

        # Validation/test files are forced onto the training schema so column
        # order drift or missing features fail loudly instead of being silently
        # misaligned during normalization.
        df = align_columns(df, base_cols, path)
        df = normalize_df(df, feature_cols, means, stds)

        dfs.append(df)
        total_rows += len(df)

    if not dfs:
        raise ValueError(f"No usable {split_name} data after cleaning.")

    return dfs, total_rows


def build_windows_from_df(
    df: pd.DataFrame,
    lookback: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Build (X, Y) windows for a single file without crossing boundaries.

    Returns:
        X: (N - lookback, lookback, D)
        Y: (N - lookback, D)
    """
    data = df.to_numpy(dtype=np.float32)
    n_rows, _ = data.shape
    n_samples = n_rows - lookback
    if n_samples <= 0:
        return None

    # Build windows without crossing file boundaries by shifting the same array
    # view over the lookback horizon.
    windows = [data[i : i + n_samples] for i in range(lookback)]
    x = np.stack(windows, axis=1)
    y = data[lookback:]
    return x, y


def build_windows_from_dfs(
    dfs: Sequence[pd.DataFrame],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Concatenate per-file windows into a single dataset.
    Returns:
        x_all: (N_total, lookback, D)
        y_all: (N_total, D)
        total_samples: int
    """
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    total_samples = 0

    for df in dfs:
        # Window each file independently so sequences never cross file
        # boundaries. That keeps one trip/session from leaking context into the
        # next one.
        result = build_windows_from_df(df, lookback)
        if result is None:
            continue
        x, y = result
        x_list.append(x)
        y_list.append(y)
        total_samples += x.shape[0]

    if not x_list:
        raise ValueError("No samples available after applying lookback to the data.")

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return x_all, y_all, total_samples
