import re
from pathlib import Path
from typing import List, Optional, Sequence

META_COLS = ["Time(s)", "Class", "PathOrder"]

IGNORED_FEATURES = [
    "Target_engine_speed_used_in_lock-up_module",
    "Inhibition_of_engine_fuel_cut_off",
    "Torque_scaling_factor(standardization)",
    "Glow_plug_control_request",
    "Engine_soacking_time",
    "Requested_spark_retard_angle_from_TCU",
    "Standard_Torque_Ratio",
    "TCU_requested_engine_RPM_increase",
    "TCU_requests_engine_torque_limit_(ETL)",
    "Filtered_Accelerator_Pedal_value",
    "Engine_coolant_temperature",
    "Engine_coolant_temperature.1",
    "Activation_of_Air_compressor",
    "Fuel_Pressure",
    "Engine_in_fuel_cut_off",
    "Throttle_position_signal",
    "Engine_torque_after_correction",
    "Gear_Selection",
    "Current_spark_timing",
    "Short_Term_Fuel_Trim_Bank1",
    "Long_Term_Fuel_Trim_Bank1",
]


def parse_driver_from_filename(fname: str) -> str:
    patterns = (
        r"trip_([A-Za-z])_",
        r"driver_([A-Za-z])(?:\d|_|\.)",
    )
    for pattern in patterns:
        match = re.search(pattern, fname)
        if match:
            return match.group(1).upper()
    raise ValueError(
        f"Filename '{fname}' does not match expected HCRL patterns like "
        "trip_A_1_1.csv or driver_A1.csv"
    )


def parse_driver_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    raw = [v.strip().upper() for v in value.split(",")]
    return [v for v in raw if v]


def list_files_for_drivers(data_dir: Path, drivers: Sequence[str]) -> List[Path]:
    """
    HCRL: flat directory; driver encoded in filename like trip_A_1_1.csv
    or driver_A1.csv.
    """
    if not drivers:
        return []
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    drivers_set = {d.upper() for d in drivers}
    counts = {d: 0 for d in drivers_set}
    matched: List[Path] = []

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
