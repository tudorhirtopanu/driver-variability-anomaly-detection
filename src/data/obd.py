from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

META_COLS = ["Time", "Time(s)", "Class", "PathOrder"]
IGNORED_FEATURES: List[str] = []


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize known encoding artifacts (e.g., "Â°C" -> "°C")
    return df.rename(columns=lambda c: c.replace("Â", ""))


def parse_driver_list(value: Optional[str]) -> List[str]:
    # OBD does not use driver-based splits, but the loader exposes the same
    # helper interface as the other datasets.
    if value is None:
        return []
    raw = [v.strip().upper() for v in value.split(",")]
    return [v for v in raw if v]


def list_files_for_drivers(data_dir: Path, drivers: Sequence[str]) -> List[Path]:
    raise ValueError("OBD dataset does not use driver splits. Use --split-json/--split-id.")


def list_csv_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return sorted(data_dir.glob("*.csv"))
