import re
from pathlib import Path
from typing import List, Optional, Sequence

META_COLS = ["Time(s)", "Class", "PathOrder"]

IGNORED_FEATURES = [
    "fuel_pressure",
    "flywheel_torque",
    "glow_plug_limit_request",
    "accelerator_position_filtered",
    "flywheel_torque_revised",
    "engine_pressure_maintanance_time",
    "clutch_check",
    # Broken / Default Sensor Values (Constant across all files)
    "logitude_acceleration",  # Typo in dataset (Longitude), constant value
    "latitude_acceleration",  # Constant value
    "brake_sylinder",         # Typo in dataset (Cylinder), constant value
    # Quasi-Constant (Mostly Noise)
    "compressor_activation",  # Constant >99% of the time
    "reduce_block_fuel",      # Constant >98% of the time
    "block_fuel",
    "road_slope",
    "engine_velocity_increase_tcu",
    "target_engine_velocity_lockup",
    "fire_angle_delay_tcu",
    "torque_transform_coeff",
    "engine_torque_limit_tcu",
    "long_fuel_bank",
    "short_fuel_bank",
    "standard_torque_ratio",
    "current_gear",
    "gear_choice",
    "engine_torque_max",
    "mission_oil_temp",
]


def parse_driver_from_filename(fname: str) -> str:
    match = re.search(r"trip_([A-Za-z])_", fname)
    if not match:
        raise ValueError(
            f"Filename '{fname}' does not match expected pattern like trip_A_1_1.csv"
        )
    return match.group(1).upper()


def parse_driver_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    raw = [v.strip().upper() for v in value.split(",")]
    return [v for v in raw if v]


def list_files_for_drivers(data_dir: Path, drivers: Sequence[str]) -> List[Path]:
    """
    Sonata: prefer per-driver subdirectories (data_dir/A/*.csv), else fallback
    to flat directory with driver encoded in filename.
    """
    if not drivers:
        return []
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    drivers_set = {d.upper() for d in drivers}
    counts = {d: 0 for d in drivers_set}
    matched: List[Path] = []

    # Prefer per-driver subdirectories (data_dir/A, data_dir/B, ...)
    for driver in sorted(drivers_set):
        driver_dir = data_dir / driver
        if not driver_dir.exists() or not driver_dir.is_dir():
            continue
        files = sorted(driver_dir.glob("*.csv"))
        if files:
            matched.extend(files)
            counts[driver] += len(files)

    # Fallback: flat directory with driver encoded in filename.
    if not matched:
        for path in sorted(data_dir.glob("*.csv")):
            try:
                driver = parse_driver_from_filename(path.name)
            except ValueError:
                continue
            if driver in drivers_set:
                matched.append(path)
                counts[driver] += 1

    missing = [d for d, c in counts.items() if c == 0]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"No CSV files found for drivers [{missing_str}] in {data_dir}"
        )
    return matched
