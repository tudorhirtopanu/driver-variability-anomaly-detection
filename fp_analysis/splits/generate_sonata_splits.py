"""Generate saved Sonata driver splits for the false-positive study."""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List

from fp_analysis.sonata.config import (
    ALL_DRIVERS,
    NUM_SEEN_OPTIONS,
    NUM_SPLITS_PER_CONFIG,
    RANDOM_SEED,
    set_global_seeds,
)
from fp_analysis.splits.paths import SONATA_SPLITS_JSON


def random_split(num_seen: int) -> tuple[List[str], List[str]]:
    total_drivers = len(ALL_DRIVERS)
    if not (0 < num_seen < total_drivers):
        raise ValueError(
            f"num_seen must be between 1 and {total_drivers - 1} for {total_drivers} drivers."
        )
    train_drivers = sorted(random.sample(ALL_DRIVERS, k=num_seen))
    unseen_drivers = sorted([d for d in ALL_DRIVERS if d not in train_drivers])
    return train_drivers, unseen_drivers


def generate_unique_splits(num_seen: int, num_splits: int) -> List[Dict[str, List[str]]]:
    all_combos = list(combinations(ALL_DRIVERS, num_seen))
    if num_splits > len(all_combos):
        raise ValueError(
            f"Requested {num_splits} splits for {num_seen} seen drivers, "
            f"but only {len(all_combos)} unique combinations exist."
        )

    random.shuffle(all_combos)
    chosen = all_combos[:num_splits]

    splits = []
    for combo in chosen:
        train_drivers = sorted(combo)
        unseen_drivers = sorted([d for d in ALL_DRIVERS if d not in combo])
        splits.append(
            {"train_drivers": train_drivers, "unseen_drivers": unseen_drivers}
        )
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the saved Sonata split configuration used by fp_analysis."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SONATA_SPLITS_JSON,
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
