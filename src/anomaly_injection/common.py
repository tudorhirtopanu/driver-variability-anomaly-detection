"""Shared helpers for the synthetic anomaly injection scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def parse_comma_separated_list(value: str) -> list[str]:
    """Split a comma-separated CLI argument into trimmed items."""
    return [item.strip() for item in value.split(",") if item.strip()]


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, marking non-numeric values as NaN."""
    return pd.to_numeric(series, errors="coerce")


def list_csv_files(directory: Path) -> list[Path]:
    """Return the CSV files directly under a directory in deterministic order."""
    return sorted(path for path in directory.glob("*.csv") if path.is_file())


def mean_abs_change(before: np.ndarray, after: np.ndarray) -> float:
    """Compute the mean absolute element-wise change between two arrays."""
    return float(np.mean(np.abs(after - before)))
