#!/usr/bin/env bash
set -euo pipefail

# Dispatcher for dataset-specific synthetic anomaly injection scripts.
# The injection logic stays in Python modules, while this script gives one
# consistent entrypoint for generating synthetic data across datasets.

usage() {
  echo "Usage:"
  echo "  scripts/run_inject_synth.sh <hcrl|sonata|obd> [extra inject args...]"
  echo
  echo "Notes:"
  echo "  Pass extra Python arguments after --"
  echo
  echo "Examples:"
  echo "  scripts/run_inject_synth.sh hcrl"
  echo "  scripts/run_inject_synth.sh sonata -- --summary-name anomaly_log.csv"
  echo "  scripts/run_inject_synth.sh obd -- --length 40 --alpha 0.1"
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

DATASET="$1"
shift

if [[ "$DATASET" != "hcrl" && "$DATASET" != "sonata" && "$DATASET" != "obd" ]]; then
  echo "Error: dataset must be one of: hcrl, sonata, obd"
  usage
  exit 1
fi

if [[ ${1:-} == "--" ]]; then
  shift
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_PATHS="$ROOT_DIR/scripts/local_paths.sh"
if [[ -f "$LOCAL_PATHS" ]]; then
  # shellcheck source=/dev/null
  source "$LOCAL_PATHS"
fi
PYTHONPATH_DIR="$ROOT_DIR/src"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

run_inject() {
  PYTHONPATH="$PYTHONPATH_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m "$@"
}

case "$DATASET" in
  hcrl)
    : "${HCRL_DATA_DIR:?Set HCRL_DATA_DIR in scripts/local_paths.sh or your shell}"
    INPUT_DIR="$HCRL_DATA_DIR"
    OUTPUT_DIR="${HCRL_SYNTH_OUTPUT_DIR:-$ROOT_DIR/synthetic_data/hcrl/HCRL_synth_F_H}"
    TEST_DRIVERS="${HCRL_SYNTH_TEST_DRIVERS:-F,H}"

    echo "Running HCRL synthetic injection..."
    echo "  input-dir    : $INPUT_DIR"
    echo "  output-dir   : $OUTPUT_DIR"
    echo "  test-drivers : $TEST_DRIVERS"
    echo

    run_inject anomaly_injection.inject_hcrl \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --test-drivers "$TEST_DRIVERS" \
      "$@"
    ;;

  sonata)
    : "${SONATA_DATA_DIR:?Set SONATA_DATA_DIR in scripts/local_paths.sh or your shell}"
    INPUT_DIR="$SONATA_DATA_DIR"
    OUTPUT_DIR="${SONATA_SYNTH_OUTPUT_DIR:-$ROOT_DIR/synthetic_data/sonata}"
    DRIVERS="${SONATA_SYNTH_DRIVERS:-A,B,C,D}"
    SUMMARY_NAME="${SONATA_SYNTH_SUMMARY_NAME:-anomaly_log.csv}"

    echo "Running Sonata synthetic injection..."
    echo "  input-dir     : $INPUT_DIR"
    echo "  output-dir    : $OUTPUT_DIR"
    echo "  drivers       : $DRIVERS"
    echo "  summary-name  : $SUMMARY_NAME"
    echo

    run_inject anomaly_injection.inject_sonata \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --drivers "$DRIVERS" \
      --summary-name "$SUMMARY_NAME" \
      "$@"
    ;;

  obd)
    : "${OBD_DATA_DIR:?Set OBD_DATA_DIR in scripts/local_paths.sh or your shell}"
    INPUT_DIR="$OBD_DATA_DIR"
    OUTPUT_DIR="${OBD_SYNTH_OUTPUT_DIR:-$ROOT_DIR/synthetic_data/obd}"

    echo "Running OBD synthetic injection..."
    echo "  input-dir   : $INPUT_DIR"
    echo "  output-dir  : $OUTPUT_DIR"
    echo

    run_inject anomaly_injection.inject_obd \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      "$@"
    ;;
esac
