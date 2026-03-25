"""
Helpers for loading explicit train/val/test splits from JSON files.

These utilities are used in the OBD workflows and in any experiment where the
split needs to be pinned to a file list instead of inferred from driver IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


class ResolvedSplitFiles(TypedDict):
    train: list[Path]
    val: list[Path]
    test: list[Path]
    split_entry: dict[str, Any]
    split_id: int
    split_json: Path


def load_split_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Split JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "splits" not in data or not data["splits"]:
        raise ValueError(f"Split JSON missing 'splits' entries: {path}")
    return data


def _select_split_entry(data: dict[str, Any], split_id: int) -> dict[str, Any]:
    splits = data.get("splits", [])
    for entry in splits:
        if entry.get("id") == split_id:
            return entry

    # Some compact split files omit explicit ids and rely on list position.
    # Falling back to the array index keeps those files usable while still
    # preferring explicit ids when they are present.
    if 0 <= split_id < len(splits):
        return splits[split_id]
    raise IndexError(
        f"Split id {split_id} not found; available ids: "
        f"{[s.get('id', i) for i, s in enumerate(splits)]}"
    )


def _resolve_path_list(
    entry: dict[str, Any],
    key: str,
    *,
    split_json: Path,
    data_dir: Path,
) -> list[Path]:
    files = entry.get(key)
    if files is None:
        files = entry.get(f"{key}_files")
    if files is None:
        raise ValueError(f"Split entry missing '{key}' list in {split_json}")

    resolved: list[Path] = []
    for name in files:
        path = Path(name)
        if not path.is_absolute():
            path = data_dir / path
        resolved.append(path)
    return resolved


def _validate_paths_exist(paths: list[Path], *, split_json: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return

    missing_str = ", ".join(str(path) for path in missing[:5])
    raise FileNotFoundError(
        f"Missing {len(missing)} files referenced by split JSON "
        f"(showing up to 5): {missing_str}"
    )


def _validate_non_overlapping_partitions(
    *,
    train_files: list[Path],
    val_files: list[Path],
    test_files: list[Path],
) -> None:
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)

    # Overlap usually means the split JSON was assembled incorrectly, so fail
    # early rather than silently leaking files across train/val/test.
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if not overlap:
        return

    overlap_str = ", ".join(str(path) for path in sorted(overlap))
    raise ValueError(f"Split lists overlap: {overlap_str}")


def resolve_split_files(
    split_json: Path, split_id: int, data_dir: Path
) -> ResolvedSplitFiles:
    data = load_split_json(split_json)
    entry = _select_split_entry(data, split_id)

    train_files = _resolve_path_list(entry, "train", split_json=split_json, data_dir=data_dir)
    val_files = _resolve_path_list(entry, "val", split_json=split_json, data_dir=data_dir)
    test_files = _resolve_path_list(entry, "test", split_json=split_json, data_dir=data_dir)

    all_paths = train_files + val_files + test_files
    _validate_paths_exist(all_paths, split_json=split_json)
    _validate_non_overlapping_partitions(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
    )

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
        "split_entry": entry,
        "split_id": entry.get("id", split_id),
        "split_json": split_json,
    }
