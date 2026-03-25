"""Shared path loading for src entrypoints that rely on scripts/local_paths.sh."""

from __future__ import annotations

import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATHS_SH = REPO_ROOT / "scripts" / "local_paths.sh"


def _read_shell_exports(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    exports: dict[str, str] = {}
    pattern = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        exports[key] = value
    return exports


_LOCAL_EXPORTS = _read_shell_exports(LOCAL_PATHS_SH)


def get_path(env_var: str, default: str | Path | None = None) -> Path:
    value = os.environ.get(env_var)
    if value:
        return Path(value)

    if env_var in _LOCAL_EXPORTS:
        return Path(_LOCAL_EXPORTS[env_var])

    if default is None:
        raise RuntimeError(
            f"Path variable {env_var} is not set and {LOCAL_PATHS_SH} does not define it."
        )

    return Path(default)
