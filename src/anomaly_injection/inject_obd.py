"""Synthetic anomaly injection for OBD."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from anomaly_injection.common import coerce_numeric, list_csv_files, mean_abs_change
    from path_config import get_path
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from anomaly_injection.common import coerce_numeric, list_csv_files, mean_abs_change
    from path_config import get_path


PEDAL_COLS = [
    "Accelerator Pedal Position D [%]",
    "Accelerator Pedal Position E [%]",
    "Absolute Throttle Position [%]",
]
SPEED_COL = "Vehicle Speed Sensor [km/h]"
RPM_COL = "Engine RPM [RPM]"
DEFAULT_INPUT_DIR = get_path("OBD_DATA_DIR", default="path/to/obd_data")


@dataclass
class InjectConfig:
    seed: int
    lookback: int
    length: int
    alpha: float
    pedal_min: float
    pedal_std_max: float
    speed_min: float
    rpm_min: float
    speed_trend_min: float
    rpm_trend_min: float
    effect_speed_min: float
    effect_rpm_min: float


def pick_pedal_col(columns: List[str]) -> Optional[str]:
    for col in PEDAL_COLS:
        if col in columns:
            return col
    return None


def find_candidates(
    pedal: np.ndarray,
    speed: np.ndarray,
    rpm: np.ndarray,
    existing_labels: np.ndarray,
    cfg: InjectConfig,
) -> List[int]:
    n = len(pedal)
    length = cfg.length
    lookback = cfg.lookback
    candidates = []

    if n < lookback + length:
        return candidates

    for start in range(lookback, n - length + 1):
        ctx_slice = slice(start - lookback, start)
        seg_slice = slice(start, start + length)

        ctx_pedal = pedal[ctx_slice]
        ctx_speed = speed[ctx_slice]
        ctx_rpm = rpm[ctx_slice]

        seg_speed = speed[seg_slice]
        seg_rpm = rpm[seg_slice]

        if not (
            np.isfinite(ctx_pedal).all()
            and np.isfinite(ctx_speed).all()
            and np.isfinite(ctx_rpm).all()
            and np.isfinite(seg_speed).all()
            and np.isfinite(seg_rpm).all()
        ):
            continue

        pedal_mean = float(ctx_pedal.mean())
        pedal_std = float(ctx_pedal.std())
        speed_mean = float(ctx_speed.mean())
        rpm_mean = float(ctx_rpm.mean())

        if pedal_mean <= cfg.pedal_min:
            continue
        if pedal_std >= cfg.pedal_std_max:
            continue
        if speed_mean <= cfg.speed_min:
            continue
        if rpm_mean <= cfg.rpm_min:
            continue

        speed_trend = speed[start - 1] - speed[start - cfg.lookback]
        rpm_trend = rpm[start - 1] - rpm[start - cfg.lookback]
        if not (speed_trend > cfg.speed_trend_min or rpm_trend > cfg.rpm_trend_min):
            continue

        seg_speed_trend = seg_speed[-1] - seg_speed[0]
        seg_rpm_trend = seg_rpm[-1] - seg_rpm[0]
        if not (seg_speed_trend > cfg.speed_trend_min or seg_rpm_trend > cfg.rpm_trend_min):
            continue

        if speed[start - 1] <= cfg.speed_min or rpm[start - 1] <= cfg.rpm_min:
            continue

        if existing_labels is not None and np.any(existing_labels[seg_slice] > 0):
            continue

        candidates.append(start)

    return candidates


def apply_injection(
    speed_seg: np.ndarray,
    rpm_seg: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    speed0 = float(speed_seg[0])
    rpm0 = float(rpm_seg[0])
    speed_after = speed0 + alpha * (speed_seg - speed0)
    rpm_after = rpm0 + alpha * (rpm_seg - rpm0)

    effect_speed = mean_abs_change(speed_seg, speed_after)
    effect_rpm = mean_abs_change(rpm_seg, rpm_after)
    return speed_after, rpm_after, effect_speed, effect_rpm


def process_file(path: Path, cfg: InjectConfig, rng: np.random.Generator):
    df = pd.read_csv(path)
    pedal_col = pick_pedal_col(list(df.columns))
    if pedal_col is None or SPEED_COL not in df.columns or RPM_COL not in df.columns:
        return {
            "filename": path.name,
            "start_idx": "",
            "end_idx": "",
            "pedal_col": pedal_col or "",
            "details": "",
            "status": "skipped: missing required columns",
        }, df

    existing_labels = None
    if "is_anomaly" in df.columns:
        existing_labels = coerce_numeric(df["is_anomaly"]).fillna(0).to_numpy(dtype=np.float32)

    pedal = coerce_numeric(df[pedal_col]).to_numpy(dtype=np.float32)
    speed = coerce_numeric(df[SPEED_COL]).to_numpy(dtype=np.float32)
    rpm = coerce_numeric(df[RPM_COL]).to_numpy(dtype=np.float32)

    candidates = find_candidates(pedal, speed, rpm, existing_labels, cfg)
    if not candidates:
        return {
            "filename": path.name,
            "start_idx": "",
            "end_idx": "",
            "pedal_col": pedal_col,
            "details": "",
            "status": "skipped: no valid window meeting constraints",
        }, df

    rng.shuffle(candidates)
    chosen = None
    chosen_stats = None

    for start in candidates:
        seg_slice = slice(start, start + cfg.length)
        speed_seg = speed[seg_slice]
        rpm_seg = rpm[seg_slice]

        speed_after, rpm_after, effect_speed, effect_rpm = apply_injection(
            speed_seg, rpm_seg, cfg.alpha
        )

        if not (effect_speed >= cfg.effect_speed_min and effect_rpm >= cfg.effect_rpm_min):
            continue

        chosen = start
        chosen_stats = (speed_after, rpm_after, effect_speed, effect_rpm)
        break

    if chosen is None:
        return {
            "filename": path.name,
            "start_idx": "",
            "end_idx": "",
            "pedal_col": pedal_col,
            "details": "",
            "status": "skipped: effect size too small for all candidates",
        }, df

    start = chosen
    seg_slice = slice(start, start + cfg.length)
    speed_after, rpm_after, effect_speed, effect_rpm = chosen_stats

    df_out = df.copy()
    speed_idx = df_out.columns.get_loc(SPEED_COL)
    rpm_idx = df_out.columns.get_loc(RPM_COL)
    speed_dtype = df_out[SPEED_COL].dtype
    rpm_dtype = df_out[RPM_COL].dtype
    speed_vals = speed_after
    rpm_vals = rpm_after

    if np.issubdtype(speed_dtype, np.integer):
        speed_vals = np.rint(speed_vals).astype(speed_dtype, copy=False)
    else:
        speed_vals = speed_vals.astype(speed_dtype, copy=False)
    if np.issubdtype(rpm_dtype, np.integer):
        rpm_vals = np.rint(rpm_vals).astype(rpm_dtype, copy=False)
    else:
        rpm_vals = rpm_vals.astype(rpm_dtype, copy=False)

    df_out.iloc[seg_slice, speed_idx] = speed_vals
    df_out.iloc[seg_slice, rpm_idx] = rpm_vals
    df_out["is_anomaly"] = 0
    df_out.loc[seg_slice, "is_anomaly"] = 1

    details = (
        f"alpha={cfg.alpha}, L={cfg.length}, "
        f"effect_speed={effect_speed:.3f}, effect_rpm={effect_rpm:.3f}"
    )
    return {
        "filename": path.name,
        "start_idx": start,
        "end_idx": start + cfg.length - 1,
        "pedal_col": pedal_col,
        "details": details,
        "status": "success",
    }, df_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Inject power-loss under steady pedal anomalies")
    # OBD uses explicit input/output paths so synthetic trip generation stays
    # separate from the nominal dataset root and result directories.
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--length", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--pedal-min", type=float, default=25)
    p.add_argument("--pedal-std-max", type=float, default=0.7)
    p.add_argument("--speed-min", type=float, default=10)
    p.add_argument("--rpm-min", type=float, default=1000)
    p.add_argument("--speed-trend-min", type=float, default=8)
    p.add_argument("--rpm-trend-min", type=float, default=300)
    p.add_argument("--effect-speed-min", type=float, default=1.5)
    p.add_argument("--effect-rpm-min", type=float, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = InjectConfig(
        seed=args.seed,
        lookback=args.lookback,
        length=args.length,
        alpha=args.alpha,
        pedal_min=args.pedal_min,
        pedal_std_max=args.pedal_std_max,
        speed_min=args.speed_min,
        rpm_min=args.rpm_min,
        speed_trend_min=args.speed_trend_min,
        rpm_trend_min=args.rpm_trend_min,
        effect_speed_min=args.effect_speed_min,
        effect_rpm_min=args.effect_rpm_min,
    )

    in_dir = args.input_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    log_rows = []

    files = list_csv_files(in_dir)
    if not files:
        print(f"No CSV files found in {in_dir}")
        return

    for path in files:
        out_path = out_dir / path.name
        log_entry, df_out = process_file(path, cfg, rng)
        df_out.to_csv(out_path, index=False)
        log_rows.append(log_entry)
        print(f"{path.name}: {log_entry['status']}")

    log_df = pd.DataFrame(log_rows)
    log_path = out_dir / "anomaly_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
