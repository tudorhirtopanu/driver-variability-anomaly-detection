"""
Synthetic anomaly injection for Sonata.

This module generates synthetic Sonata trips by applying a small set of
automotive anomalies across the configured driver folders.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from anomaly_injection.common import parse_comma_separated_list
    from path_config import get_path
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from anomaly_injection.common import parse_comma_separated_list
    from path_config import get_path


DEFAULT_INPUT_DIR = get_path("SONATA_DATA_DIR", default="path/to/sonata_data")
DEFAULT_OUTPUT_DIR = "./Sonata_Synthetic_Final"
DEFAULT_DRIVERS = "A,B,C,D"
DEFAULT_SUMMARY_NAME = "anomaly_summary.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Inject Sonata synthetic anomalies")
    # Default paths follow the standard dataset and output layout used here.
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument("--drivers", type=str, default=DEFAULT_DRIVERS)
    p.add_argument("--summary-name", type=str, default=DEFAULT_SUMMARY_NAME)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def parse_driver_list(value: str) -> list[str]:
    return parse_comma_separated_list(value)


def find_active_window(series: pd.Series, window_size: int, min_val: float, min_variance: float):
    """
    Finds a window where the signal is active to ensure the injected
    anomaly is semantically meaningful and not hidden in idle noise.
    """
    valid_starts = []
    series_vals = series.values
    for i in range(len(series) - window_size):
        segment = series_vals[i : i + window_size]
        if np.mean(segment) < min_val:
            continue
        if np.std(segment) < min_variance:
            continue
        valid_starts.append(i)

    return random.choice(valid_starts) if valid_starts else None


def inject_sonata_anomalies(
    df: pd.DataFrame, filename: str, driver_id: str, log_list: list[dict]
) -> pd.DataFrame:
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    pedal = "accelerator_position"
    pressure = "inhale_pressure"
    torque = "engine_torque"

    for col in [pedal, pressure, torque, "car_speed"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    freeze_len = 20
    start_idx = find_active_window(df[pedal], window_size=freeze_len, min_val=18.0, min_variance=1.0)
    if start_idx is not None:
        end_idx = start_idx + 19
        if df.loc[start_idx:end_idx, "is_anomaly"].sum() == 0:
            freeze_val = df.loc[start_idx:end_idx, pedal].max()
            df.loc[start_idx:end_idx, pedal] = freeze_val
            df.loc[start_idx:end_idx, "is_anomaly"] = 1
            log_list.append(
                {
                    "filename": filename,
                    "driver": driver_id,
                    "anomaly": "Active_Freeze",
                    "details": f"Frozen at {freeze_val:.1f} (Segment Max)",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

    drift_len = 40
    start_idx_drift = find_active_window(
        df[pressure], window_size=drift_len, min_val=100.0, min_variance=1.0
    )
    if start_idx_drift is not None:
        end_idx = start_idx_drift + 39
        if df.loc[start_idx_drift:end_idx, "is_anomaly"].sum() == 0:
            drift = np.linspace(0, -60, num=drift_len)
            df.loc[start_idx_drift:end_idx, pressure] += drift
            df.loc[start_idx_drift:end_idx, "is_anomaly"] = 1
            log_list.append(
                {
                    "driver": driver_id,
                    "filename": filename,
                    "type": "Impossible_Drift",
                    "details": "Linear drift -60 units",
                    "start": start_idx_drift,
                    "end": end_idx,
                }
            )

    phys_len = 20
    start_idx_phys = find_active_window(df[torque], window_size=phys_len, min_val=0.8, min_variance=0.1)
    if start_idx_phys is not None:
        end_idx = start_idx_phys + 19
        if df.loc[start_idx_phys:end_idx, "is_anomaly"].sum() == 0:
            segment = df.loc[start_idx_phys:end_idx, torque]
            mean_val = segment.mean()
            inverted = mean_val - (segment - mean_val)
            df.loc[start_idx_phys:end_idx, torque] = inverted
            df.loc[start_idx_phys:end_idx, "is_anomaly"] = 1
            log_list.append(
                {
                    "driver": driver_id,
                    "filename": filename,
                    "type": "Physics_Inversion",
                    "details": "Inverted Torque dynamics",
                    "start": start_idx_phys,
                    "end": end_idx,
                }
            )

    return df


def main() -> None:
    args = parse_args()
    drivers = parse_driver_list(args.drivers)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    injection_log: list[dict] = []
    file_count = 0

    for driver in drivers:
        driver_path = args.input_dir / driver
        if not driver_path.exists():
            print(f"Skipping {driver}: Folder not found.")
            continue

        output_driver_path = args.output_dir / driver
        output_driver_path.mkdir(parents=True, exist_ok=True)

        files = glob.glob(str(driver_path / "*.csv"))
        print(f"Processing Driver {driver}: {len(files)} files.")

        for filepath in files:
            try:
                filename = os.path.basename(filepath)
                df = pd.read_csv(filepath)
                df_injected = inject_sonata_anomalies(df, filename, driver, injection_log)
                df_injected.to_csv(output_driver_path / filename, index=False)
                file_count += 1
            except Exception as exc:
                print(f"Error {driver}/{filename}: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / args.summary_name
    pd.DataFrame(injection_log).to_csv(summary_path, index=False)
    print(f"\nDone. Processed {file_count} files across folders {', '.join(drivers)}.")


if __name__ == "__main__":
    main()
