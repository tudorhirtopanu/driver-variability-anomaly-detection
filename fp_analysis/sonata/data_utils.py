# data_utils.py
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import itertools
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fp_analysis.sonata.config import (
    DATA_DIR,
    IGNORED_FEATURES,
    META_COLS,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)


def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove metadata / ignored columns if they exist."""
    cols_to_drop = [c for c in META_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    ignored_to_drop = [c for c in IGNORED_FEATURES if c in df.columns]
    if ignored_to_drop:
        df = df.drop(columns=ignored_to_drop)

    return df


def _load_trip_from_path(path: Path, feature_names: Optional[List[str]]) -> Tuple[np.ndarray, List[str]]:
    """
    Load one trip CSV, drop meta/ignored columns, and ensure consistent column order.
    If feature_names is provided, the CSV must contain those columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df = _drop_columns(df)

    if feature_names is None:
        feature_names = list(df.columns)
    else:
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Trip {path} is missing expected features: {missing}")
        df = df[feature_names]

    data = df.to_numpy(dtype=np.float32)
    return data, feature_names


def _list_trip_paths(driver: str) -> List[Path]:
    driver_dir = DATA_DIR / driver
    if not driver_dir.exists():
        raise FileNotFoundError(f"Driver folder not found: {driver_dir}")
    paths = sorted(driver_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV trips found for driver {driver} in {driver_dir}")
    return paths


def load_driver_trips(
    driver: str,
    feature_names: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """
    Load every trip for a given driver.

    Returns a tuple of:
        trips: list of {"trip_id", "data", "length", "path"}
        feature_names: the shared, ordered feature names
    """
    trips = []
    for path in _list_trip_paths(driver):
        data, feature_names = _load_trip_from_path(path, feature_names)
        trips.append(
            {
                "trip_id": path.stem,
                "data": data,
                "length": data.shape[0],
                "path": path,
            }
        )
    return trips, feature_names


def _choose_trip_allocation(trips: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    """
    Assign full trips to train/val/test to approximate 70/10/20 by total length.
    Exhaustively searches all assignments (<= 3^9 combos) and picks the one
    with the smallest squared error to the target fractions. Empty splits are
    penalised to keep validation/test non-empty.
    """
    if not trips:
        raise ValueError("No trips provided for allocation.")

    lengths = [t["length"] for t in trips]
    total = float(sum(lengths))
    target_fracs = (TRAIN_FRAC, VAL_FRAC, TEST_FRAC)

    best_score = None
    best_assignments = []
    labels = ["train", "val", "test"]

    for assignment in itertools.product(range(3), repeat=len(trips)):
        totals = [0.0, 0.0, 0.0]
        for l, bucket in zip(lengths, assignment):
            totals[bucket] += l

        fracs = [t / total for t in totals]
        error = sum((f - tgt) ** 2 for f, tgt in zip(fracs, target_fracs))

        # Penalise empty splits heavily so we keep hold-out trips.
        penalty = sum(1.0 for t in totals if t == 0.0)
        score = error + penalty

        if best_score is None or score < best_score - 1e-12:
            best_score = score
            best_assignments = [assignment]
        elif abs(score - best_score) <= 1e-12:
            best_assignments.append(assignment)

    choice = random.choice(best_assignments)

    grouped = {k: [] for k in labels}
    for trip, bucket in zip(trips, choice):
        grouped[labels[bucket]].append(trip)

    return grouped


def prepare_split(train_drivers, unseen_drivers):
    """
    Splitting logic for the new multi-trip dataset:
        Seen drivers:
            assign whole trips to train/val/test to approximate 70/10/20 by length
        Unseen drivers:
            keep all trips as test

    Returns:
        train_seqs
        val_seqs
        seen_test_seqs
        unseen_test_seqs
        seen_test_driver_ids
        unseen_test_driver_ids
        feature_names
        scaler_mean
        scaler_scale
    """

    train_seqs_raw: List[np.ndarray] = []
    val_seqs_raw: List[np.ndarray] = []
    seen_test_seqs_raw: List[np.ndarray] = []
    unseen_test_seqs_raw: List[np.ndarray] = []

    seen_test_driver_ids: List[str] = []
    unseen_test_driver_ids: List[str] = []

    feature_names: Optional[List[str]] = None

    # ------- SEEN DRIVERS -------
    for d in train_drivers:
        driver_trips, feature_names = load_driver_trips(d, feature_names)
        grouped = _choose_trip_allocation(driver_trips)

        train_seqs_raw.extend([t["data"] for t in grouped["train"]])
        val_seqs_raw.extend([t["data"] for t in grouped["val"]])
        seen_test_seqs_raw.extend([t["data"] for t in grouped["test"]])
        seen_test_driver_ids.extend([d] * len(grouped["test"]))

    # ------- UNSEEN DRIVERS -------
    for d in unseen_drivers:
        driver_trips, feature_names = load_driver_trips(d, feature_names)
        for trip in driver_trips:
            unseen_test_seqs_raw.append(trip["data"])
            unseen_test_driver_ids.append(d)

    if not train_seqs_raw:
        raise ValueError("No training sequences were created; check trip allocation.")
    if not val_seqs_raw:
        raise ValueError("No validation sequences were created; check trip allocation.")

    # Fit scaler ONLY on training data
    train_concat = np.concatenate(train_seqs_raw, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_concat)

    def norm(a): return scaler.transform(a)

    return {
        "train_seqs": [norm(a) for a in train_seqs_raw],
        "val_seqs": [norm(a) for a in val_seqs_raw],
        "seen_test_seqs": [norm(a) for a in seen_test_seqs_raw],
        "unseen_test_seqs": [norm(a) for a in unseen_test_seqs_raw],
        "seen_test_driver_ids": seen_test_driver_ids,
        "unseen_test_driver_ids": unseen_test_driver_ids,
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }
