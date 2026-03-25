"""
Synthetic anomaly injection for HCRL.

This module generates a high-contrast synthetic HCRL dataset by applying a
small set of targeted anomalies to selected drivers.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
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


DEFAULT_INPUT_DIR = get_path("HCRL_DATA_DIR", default="path/to/hcrl_data")
DEFAULT_OUTPUT_DIR = "./HCRL_synth_F_H_synth_1"
DEFAULT_TEST_DRIVERS = "F,H"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Inject HCRL synthetic anomalies")
    # Default paths point to the standard HCRL dataset location and local
    # synthetic output directory used by this project.
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument("--test-drivers", type=str, default=DEFAULT_TEST_DRIVERS)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def parse_driver_list(value: str) -> list[str]:
    return parse_comma_separated_list(value)


def get_driver_id(filename: str) -> str:
    base = os.path.basename(filename)
    for pattern in (r"trip_([A-Za-z])_", r"driver_([A-Za-z])(?:\d|_|\.)"):
        match = re.search(pattern, base)
        if match:
            return match.group(1).upper()
    return "Unknown"


def find_active_window(series: pd.Series, window_size: int, min_val: float, min_variance: float):
    """
    Finds a window where the car is ACTUALLY driving (not idling).
    This ensures we don't inject anomalies into 'stop light' data,
    which would confuse the evaluation.
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

    if valid_starts:
        return random.choice(valid_starts)
    return None


def inject_high_contrast(df: pd.DataFrame, filename: str, log_list: list[dict]) -> pd.DataFrame:
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = 0

    cols = [
        "Accelerator_Pedal_value",
        "Intake_air_pressure",
        "Engine_torque",
        "Vehicle_speed",
    ]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    start_idx = find_active_window(
        df["Accelerator_Pedal_value"], window_size=20, min_val=25, min_variance=2.0
    )
    if start_idx is not None:
        end_idx = start_idx + 20
        if df.loc[start_idx:end_idx, "is_anomaly"].sum() == 0:
            freeze_val = df.iloc[start_idx]["Accelerator_Pedal_value"]
            df.loc[start_idx:end_idx, "Accelerator_Pedal_value"] = freeze_val
            df.loc[start_idx:end_idx, "is_anomaly"] = 1
            log_list.append(
                {
                    "filename": filename,
                    "driver": get_driver_id(filename),
                    "anomaly": "Active_Freeze",
                    "details": f"Frozen at {freeze_val:.1f}% (Active)",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

    drift_len = 40
    start_idx_drift = find_active_window(
        df["Intake_air_pressure"],
        window_size=drift_len,
        min_val=30,
        min_variance=1.0,
    )
    if start_idx_drift is not None:
        end_idx_excl = start_idx_drift + drift_len
        end_idx_incl = end_idx_excl - 1
        if df.loc[start_idx_drift:end_idx_incl, "is_anomaly"].sum() == 0:
            drift = np.linspace(0, -60, num=drift_len)
            original = df.loc[start_idx_drift:end_idx_incl, "Intake_air_pressure"].values
            df.loc[start_idx_drift:end_idx_incl, "Intake_air_pressure"] = original + drift
            df.loc[start_idx_drift:end_idx_incl, "is_anomaly"] = 1
            log_list.append(
                {
                    "filename": filename,
                    "driver": get_driver_id(filename),
                    "anomaly": "Impossible_Drift",
                    "details": "Linear drift -60kpa",
                    "start_idx": start_idx_drift,
                    "end_idx": end_idx_incl,
                }
            )

    start_idx_phys = find_active_window(
        df["Engine_torque"], window_size=20, min_val=40, min_variance=5.0
    )
    if start_idx_phys is not None:
        end_idx = start_idx_phys + 20
        if df.loc[start_idx_phys:end_idx, "is_anomaly"].sum() == 0:
            segment = df.loc[start_idx_phys:end_idx, "Engine_torque"]
            mean_val = segment.mean()
            inverted = mean_val - (segment - mean_val)
            df.loc[start_idx_phys:end_idx, "Engine_torque"] = inverted
            df.loc[start_idx_phys:end_idx, "is_anomaly"] = 1
            log_list.append(
                {
                    "filename": filename,
                    "driver": get_driver_id(filename),
                    "anomaly": "Physics_Inversion",
                    "details": "Torque inverted vs Pedal",
                    "start_idx": start_idx_phys,
                    "end_idx": end_idx,
                }
            )

    return df


def main() -> None:
    args = parse_args()
    test_drivers = parse_driver_list(args.test_drivers)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = glob.glob(str(input_dir / "*.csv"))

    injection_log: list[dict] = []
    print(f"Generating High-Contrast Dataset for Drivers: {test_drivers}...")

    count = 0
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        driver = get_driver_id(filename)
        if driver not in test_drivers:
            continue
        try:
            df = pd.read_csv(filepath)
            df_injected = inject_high_contrast(df, filename, injection_log)
            save_path = output_dir / filename
            df_injected.to_csv(save_path, index=False)
            count += 1
        except Exception as exc:
            print(f"Error {filename}: {exc}")

    if injection_log:
        log_path = output_dir / "anomaly_log.csv"
        pd.DataFrame(injection_log).to_csv(log_path, index=False)
        print(f"\nDone. Processed {count} files.")
        print(f"Log saved to {log_path}")
    else:
        print("\nWarning: No anomalies were injected.")


if __name__ == "__main__":
    main()
