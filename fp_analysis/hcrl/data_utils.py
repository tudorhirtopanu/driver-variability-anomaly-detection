# data_utils.py
from typing import List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fp_analysis.hcrl.config import DATA_DIR, IGNORED_FEATURES, META_COLS, TRAIN_FRAC, VAL_FRAC


def _part_sort_key(path: Path):
    """Sort by numeric part suffix when present: trip_A_1_2.csv -> 2."""
    suffix = path.stem.rsplit("_", 1)[-1]
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, suffix)


def load_trip(driver: str, trip: str) -> pd.DataFrame:
    """
    Load all CSV parts for a given driver + trip, drop meta columns,
    and return a concatenated DataFrame with sensor columns only.

    Supports split files named 'trip_<driver>_<trip>_<part>.csv' and also
    handles the single-file naming scheme 'driver_<driver><trip>.csv'.
    """
    part_paths = sorted(DATA_DIR.glob(f"trip_{driver}_{trip}_*.csv"), key=_part_sort_key)

    if not part_paths:
        single_file = DATA_DIR / f"driver_{driver}{trip}.csv"
        if single_file.exists():
            part_paths = [single_file]
        else:
            raise FileNotFoundError(
                f"No CSVs found for driver {driver} trip {trip} under {DATA_DIR}"
            )

    feature_cols = None
    frames: List[pd.DataFrame] = []

    for path in part_paths:
        df_part = pd.read_csv(path)

        # Drop meta + ignored features if present
        drop_cols = [c for c in META_COLS if c in df_part.columns]
        drop_cols += [c for c in IGNORED_FEATURES if c in df_part.columns]
        if drop_cols:
            df_part = df_part.drop(columns=drop_cols)

        df_part = df_part.reset_index(drop=True)

        if feature_cols is None:
            feature_cols = df_part.columns.tolist()
        else:
            # Align later parts to the first part's feature order
            df_part = df_part.reindex(columns=feature_cols)

        frames.append(df_part)

    return pd.concat(frames, axis=0, ignore_index=True)


def prepare_split(train_drivers, unseen_drivers):
    """
    New split logic:
        Seen drivers:
            70% train, 10% val, 20% test (both There + Back)
        Unseen drivers:
            100% test (There + Back)

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

    train_seqs_raw = []
    val_seqs_raw = []
    seen_test_seqs_raw = []
    unseen_test_seqs_raw = []

    seen_test_driver_ids = []
    unseen_test_driver_ids = []

    feature_names = None

    def load_and_clean(driver, trip):
        df = load_trip(driver, trip)
        return df.values

    # ------- SEEN DRIVERS -------
    for d in train_drivers:
        for trip in ["1", "2"]:
            arr = load_and_clean(d, trip)

            if feature_names is None:
                feature_names = load_trip(d, trip).columns.tolist()

            N = arr.shape[0]
            n_train = int(N * TRAIN_FRAC)
            n_val   = int(N * VAL_FRAC)
            train_seqs_raw.append(arr[:n_train])
            val_seqs_raw.append(arr[n_train:n_train+n_val])
            seen_test_seqs_raw.append(arr[n_train+n_val:])

            seen_test_driver_ids.append(d)

    # ------- UNSEEN DRIVERS -------
    for d in unseen_drivers:
        combined = []

        for trip in ["1", "2"]:
            arr = load_and_clean(d, trip)

            if feature_names is None:
                feature_names = load_trip(d, trip).columns.tolist()

            combined.append(arr)

        full = np.concatenate(combined, axis=0)
        unseen_test_seqs_raw.append(full)
        unseen_test_driver_ids.append(d)

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
