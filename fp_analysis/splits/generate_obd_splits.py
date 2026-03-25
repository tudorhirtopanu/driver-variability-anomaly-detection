"""Generate saved OBD file-list splits for the false-positive study."""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from fp_analysis.obd.config import DATA_DIR as DEFAULT_DATA_DIR
from fp_analysis.splits.paths import OBD_SPLITS_JSON
from src.data.obd import IGNORED_FEATURES, META_COLS, _normalize_columns, list_csv_files
from src.data.preprocessing import load_and_clean_csv


def count_clean_rows(path: Path) -> int:
    df = load_and_clean_csv(
        path,
        meta_cols=META_COLS,
        ignored_features=IGNORED_FEATURES,
        sort_cols=("Time", "Time(s)"),
        preprocess_df=_normalize_columns,
    )
    return len(df)


def split_score(rows_by_split: Dict[str, int], target_fracs: Dict[str, float]) -> float:
    total = sum(rows_by_split.values())
    if total <= 0:
        return 1e9
    return sum(
        abs(rows_by_split[key] / total - target_fracs[key]) for key in target_fracs
    )


def build_split(
    file_rows: List[Tuple[Path, int]],
    target_fracs: Dict[str, float],
    rng: random.Random,
) -> Tuple[Dict[str, List[Path]], Dict[str, int], Dict[str, float], float]:
    total_rows = sum(r for _, r in file_rows)
    targets = {k: total_rows * target_fracs[k] for k in target_fracs}
    order = list(file_rows)
    rng.shuffle(order)

    splits = {k: [] for k in target_fracs}
    rows = {k: 0 for k in target_fracs}

    for path, n_rows in order:
        deficits = {k: targets[k] - rows[k] for k in splits}
        under = [k for k, value in deficits.items() if value > 0]
        if under:
            key = max(under, key=lambda name: deficits[name])
        else:
            key = min(splits, key=lambda name: rows[name] - targets[name])
        splits[key].append(path)
        rows[key] += n_rows

    if any(len(splits[key]) == 0 for key in splits):
        raise ValueError("Generated split has an empty partition.")

    total = sum(rows.values())
    fracs = {key: rows[key] / total for key in rows}
    score = split_score(rows, target_fracs)
    return splits, rows, fracs, score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OBD train/val/test splits with balanced row counts."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=OBD_SPLITS_JSON)
    parser.add_argument("--num-splits", type=int, default=20)
    parser.add_argument("--max-tries", type=int, default=2000)
    parser.add_argument(
        "--max-abs-diff",
        type=float,
        default=0.02,
        help="Max absolute fraction deviation per split (e.g. 0.02 = 2%%).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    files = list_csv_files(args.data_dir)
    if not files:
        raise ValueError(f"No CSV files found in {args.data_dir}")

    logging.info("Counting rows per file (after cleaning)...")
    file_rows = []
    for path in files:
        rows = count_clean_rows(path)
        file_rows.append((path, rows))

    total_rows = sum(rows for _, rows in file_rows)
    logging.info("Total files=%d total rows=%d", len(file_rows), total_rows)

    target_fracs = {"train": 0.6, "val": 0.2, "test": 0.2}
    rng = random.Random(args.seed)

    entries = []
    seen = set()
    for _ in range(args.max_tries):
        try:
            splits, rows, fracs, score = build_split(file_rows, target_fracs, rng)
        except ValueError:
            continue
        fingerprint = (
            tuple(sorted(path.name for path in splits["train"])),
            tuple(sorted(path.name for path in splits["val"])),
            tuple(sorted(path.name for path in splits["test"])),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        entries.append(
            {
                "train": [path.name for path in splits["train"]],
                "val": [path.name for path in splits["val"]],
                "test": [path.name for path in splits["test"]],
                "rows": rows,
                "fractions": fracs,
                "score": score,
            }
        )

    good = [
        entry
        for entry in entries
        if all(
            abs(entry["fractions"][key] - target_fracs[key]) <= args.max_abs_diff
            for key in target_fracs
        )
    ]
    good.sort(key=lambda entry: entry["score"])

    if len(good) < args.num_splits:
        entries.sort(key=lambda entry: entry["score"])
        for entry in entries:
            if entry in good:
                continue
            good.append(entry)
            if len(good) >= args.num_splits:
                break

    final_splits = []
    for idx, entry in enumerate(good[: args.num_splits]):
        split_entry = dict(entry)
        split_entry["id"] = idx
        final_splits.append(split_entry)

    output = {
        "data_dir": str(args.data_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(file_rows),
        "total_rows": total_rows,
        "target_fractions": target_fracs,
        "max_abs_diff": args.max_abs_diff,
        "file_rows": {path.name: rows for path, rows in file_rows},
        "splits": final_splits,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logging.info("Wrote %d splits to %s", len(final_splits), args.out)


if __name__ == "__main__":
    main()
