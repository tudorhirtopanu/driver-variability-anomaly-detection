"""Generate saved HCRL driver splits for the false-positive study."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from fp_analysis.hcrl.config import (
    ALL_DRIVERS,
    NUM_SEEN_OPTIONS,
    NUM_SPLITS_PER_CONFIG,
    RANDOM_SEED,
    set_global_seeds,
)

def random_split(num_seen: int) -> tuple[List[str], List[str]]:
    if not (0 < num_seen < len(ALL_DRIVERS)):
        raise ValueError("num_seen must be between 1 and 9 for 10 drivers.")
    train_drivers = sorted(random.sample(ALL_DRIVERS, k=num_seen))
    unseen_drivers = sorted([d for d in ALL_DRIVERS if d not in train_drivers])
    return train_drivers, unseen_drivers


def generate_unique_splits(num_seen: int, num_splits: int) -> List[Dict[str, List[str]]]:
    splits = []
    used = set()
    for _ in range(num_splits):
        while True:
            train_drivers, unseen_drivers = random_split(num_seen)
            key = tuple(train_drivers)
            if key not in used:
                used.add(key)
                break
        splits.append(
            {"train_drivers": train_drivers, "unseen_drivers": unseen_drivers}
        )
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the saved HCRL split configuration used by fp_analysis."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fp_analysis/splits/generated/hcrl/splits_config.json"),
        help="Output JSON path for the generated split configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(RANDOM_SEED)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "seed": RANDOM_SEED,
        "num_splits_per_config": NUM_SPLITS_PER_CONFIG,
        "splits": {},
    }

    for num_seen in NUM_SEEN_OPTIONS:
        cfg["splits"][str(num_seen)] = generate_unique_splits(
            num_seen, NUM_SPLITS_PER_CONFIG
        )

    args.out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved splits to {args.out}")


if __name__ == "__main__":
    main()
