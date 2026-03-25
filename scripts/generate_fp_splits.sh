#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  scripts/generate_fp_splits.sh <hcrl|sonata|obd> [extra generator args...]"
  echo
  echo "Notes:"
  echo "  Pass extra Python arguments after --"
  echo
  echo "Examples:"
  echo "  scripts/generate_fp_splits.sh hcrl"
  echo "  scripts/generate_fp_splits.sh sonata"
  echo "  scripts/generate_fp_splits.sh obd -- --data-dir /path/to/obd"
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

DATASET="$1"
shift

if [[ ${1:-} == "--" ]]; then
  shift
fi

case "$DATASET" in
  hcrl)
    MODULE="fp_analysis.splits.generate_hcrl_splits"
    ;;
  sonata)
    MODULE="fp_analysis.splits.generate_sonata_splits"
    ;;
  obd)
    MODULE="fp_analysis.splits.generate_obd_splits"
    ;;
  *)
    echo "Error: dataset must be one of: hcrl, sonata, obd"
    usage
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m "$MODULE" "$@"
