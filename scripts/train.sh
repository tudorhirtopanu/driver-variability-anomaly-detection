#!/usr/bin/env bash
set -euo pipefail

# Dataset-aware wrapper around `python -m train`.
# Main training defaults are grouped here so the run configuration is explicit.

usage() {
  echo "Usage:"
  echo "  scripts/train.sh <hcrl|sonata|obd> [extra train args...]"
  echo
  echo "Notes:"
  echo "  Pass extra Python arguments after --"
  echo
  echo "Examples:"
  echo "  scripts/train.sh hcrl"
  echo "  scripts/train.sh sonata -- --seed 123 --batch-size 64"
  echo "  scripts/train.sh obd -- --data-dir /path/to/obd --split-json /path/to/splits_obd.json"
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

run_train() {
  PYTHONPATH="$PYTHONPATH_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m train "$@"
}

SEED=42
LOOKBACK=20
NUM_LAYERS=2
DROPOUT=0.1
PRETRAIN_FORECASTER_EPOCHS=50
GATE_EPOCHS=50
EARLY_STOP_MIN_DELTA=0.001
GATE_AUX_WEIGHT=0.001
GATE_AUX_TEMP=2.0

case "$DATASET" in
  hcrl)
    : "${HCRL_DATA_DIR:?Set HCRL_DATA_DIR in scripts/local_paths.sh or your shell}"
    DATA_DIR="$HCRL_DATA_DIR"
    RUN_ID="${HCRL_RUN_ID:-gated_baseline_v5}"
    OUTDIR="${HCRL_OUTDIR:-$ROOT_DIR/results/hcrl/checkpoints/$RUN_ID}"
    TRAIN_DRIVERS="${HCRL_TRAIN_DRIVERS:-B,D,F,H,E,C}"
    VAL_DRIVERS="${HCRL_VAL_DRIVERS:-A,I}"
    TEST_DRIVERS="${HCRL_TEST_DRIVERS:-G,J}"

    BATCH_SIZE=128
    NUM_WORKERS=8
    HIDDEN_DIM=256
    PRETRAIN_MARGINAL_EPOCHS=20
    GATE_LR=5e-4
    EARLY_STOP_PATIENCE=10
    GATE_AUX_MARGIN=1.0
    GATE_L1=0.01

    echo "Running HCRL training..."
    echo "  data-dir      : $DATA_DIR"
    echo "  outdir        : $OUTDIR"
    echo "  train drivers : $TRAIN_DRIVERS"
    echo "  val drivers   : $VAL_DRIVERS"
    echo "  test drivers  : $TEST_DRIVERS"
    echo

    run_train \
      --dataset "$DATASET" \
      --data-dir "$DATA_DIR" \
      --outdir "$OUTDIR" \
      --train-drivers "$TRAIN_DRIVERS" \
      --val-drivers "$VAL_DRIVERS" \
      --test-drivers "$TEST_DRIVERS" \
      --lookback "$LOOKBACK" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --hidden-dim "$HIDDEN_DIM" \
      --num-layers "$NUM_LAYERS" \
      --dropout "$DROPOUT" \
      --pretrain-forecaster-epochs "$PRETRAIN_FORECASTER_EPOCHS" \
      --pretrain-marginal-epochs "$PRETRAIN_MARGINAL_EPOCHS" \
      --gate-epochs "$GATE_EPOCHS" \
      --gate-lr "$GATE_LR" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
      --use-mixture-nll \
      --include-gaussian-const \
      --gate-aux-weight "$GATE_AUX_WEIGHT" \
      --gate-aux-temp "$GATE_AUX_TEMP" \
      --gate-aux-margin "$GATE_AUX_MARGIN" \
      --gate-l1 "$GATE_L1" \
      --seed "$SEED" \
      "$@"
    ;;

  sonata)
    : "${SONATA_DATA_DIR:?Set SONATA_DATA_DIR in scripts/local_paths.sh or your shell}"
    DATA_DIR="$SONATA_DATA_DIR"
    RUN_ID="${SONATA_RUN_ID:-v1_test_C}"
    OUTDIR="${SONATA_OUTDIR:-$ROOT_DIR/results/sonata/checkpoints/$RUN_ID}"
    case "$RUN_ID" in
      v1_test_C)
        DEFAULT_TRAIN_DRIVERS="A,D"
        DEFAULT_VAL_DRIVERS="B"
        DEFAULT_TEST_DRIVERS="C"
        ;;
      v2_test_D)
        DEFAULT_TRAIN_DRIVERS="B,C"
        DEFAULT_VAL_DRIVERS="A"
        DEFAULT_TEST_DRIVERS="D"
        ;;
      v3_test_A)
        DEFAULT_TRAIN_DRIVERS="B,C"
        DEFAULT_VAL_DRIVERS="D"
        DEFAULT_TEST_DRIVERS="A"
        ;;
      v4_test_B)
        DEFAULT_TRAIN_DRIVERS="C,D"
        DEFAULT_VAL_DRIVERS="A"
        DEFAULT_TEST_DRIVERS="B"
        ;;
      *)
        DEFAULT_TRAIN_DRIVERS="${SONATA_TRAIN_DRIVERS:-}"
        DEFAULT_VAL_DRIVERS="${SONATA_VAL_DRIVERS:-}"
        DEFAULT_TEST_DRIVERS="${SONATA_TEST_DRIVERS:-}"
        if [[ -z "$DEFAULT_TRAIN_DRIVERS" || -z "$DEFAULT_VAL_DRIVERS" || -z "$DEFAULT_TEST_DRIVERS" ]]; then
          echo "Error: unsupported Sonata run id '$RUN_ID'"
          echo "Supported run ids: v1_test_C, v2_test_D, v3_test_A, v4_test_B"
          echo "Or set SONATA_TRAIN_DRIVERS, SONATA_VAL_DRIVERS, and SONATA_TEST_DRIVERS explicitly."
          exit 1
        fi
        ;;
    esac
    TRAIN_DRIVERS="${SONATA_TRAIN_DRIVERS:-$DEFAULT_TRAIN_DRIVERS}"
    VAL_DRIVERS="${SONATA_VAL_DRIVERS:-$DEFAULT_VAL_DRIVERS}"
    TEST_DRIVERS="${SONATA_TEST_DRIVERS:-$DEFAULT_TEST_DRIVERS}"

    BATCH_SIZE=64
    NUM_WORKERS=0
    HIDDEN_DIM=192
    PRETRAIN_MARGINAL_EPOCHS=40
    GATE_EPOCHS=100
    GATE_LR=7e-4
    EARLY_STOP_PATIENCE=10
    GATE_AUX_WEIGHT=0.05
    GATE_AUX_TEMP=1.0
    GATE_AUX_MARGIN=0.0
    GATE_L1=0.0001
    MARGINAL_B_L2=1e-5

    echo "Running Sonata training..."
    echo "  data-dir      : $DATA_DIR"
    echo "  run id        : $RUN_ID"
    echo "  outdir        : $OUTDIR"
    echo "  train drivers : $TRAIN_DRIVERS"
    echo "  val drivers   : $VAL_DRIVERS"
    echo "  test drivers  : $TEST_DRIVERS"
    echo

    run_train \
      --dataset "$DATASET" \
      --data-dir "$DATA_DIR" \
      --outdir "$OUTDIR" \
      --train-drivers "$TRAIN_DRIVERS" \
      --val-drivers "$VAL_DRIVERS" \
      --test-drivers "$TEST_DRIVERS" \
      --lookback "$LOOKBACK" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --hidden-dim "$HIDDEN_DIM" \
      --num-layers "$NUM_LAYERS" \
      --dropout "$DROPOUT" \
      --pretrain-forecaster-epochs "$PRETRAIN_FORECASTER_EPOCHS" \
      --pretrain-marginal-epochs "$PRETRAIN_MARGINAL_EPOCHS" \
      --gate-epochs "$GATE_EPOCHS" \
      --gate-lr "$GATE_LR" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
      --use-mixture-nll \
      --include-gaussian-const \
      --allow-calibration-with-mixture \
      --gate-aux-weight "$GATE_AUX_WEIGHT" \
      --gate-aux-temp "$GATE_AUX_TEMP" \
      --gate-aux-margin "$GATE_AUX_MARGIN" \
      --gate-l1 "$GATE_L1" \
      --marginal-b-l2 "$MARGINAL_B_L2" \
      --seed "$SEED" \
      "$@"
    ;;

  obd)
    : "${OBD_DATA_DIR:?Set OBD_DATA_DIR in scripts/local_paths.sh or your shell}"
    DATA_DIR="$OBD_DATA_DIR"
    RUN_VERSION="${OBD_RUN_VERSION:-v1}"
    OUTDIR="${OBD_OUTDIR:-$ROOT_DIR/results/obd/checkpoints/$RUN_VERSION}"
    SPLIT_JSON="${OBD_SPLIT_JSON:-$ROOT_DIR/fp_analysis/splits/generated/obd/splits_obd.json}"
    SPLIT_ID="${OBD_SPLIT_ID:-0}"

    BATCH_SIZE=256
    NUM_WORKERS=4
    HIDDEN_DIM=128
    PRETRAIN_MARGINAL_EPOCHS=5
    GATE_LR=3e-4
    EARLY_STOP_PATIENCE=5
    GATE_AUX_MARGIN=0.25
    GATE_L1=0.005
    MARGINAL_TYPE=flow
    MARGINAL_HIDDEN=16
    MARGINAL_BINS=4
    MARGINAL_TAIL=5.0
    MARGINAL_SUBSAMPLE=0.1

    echo "Running OBD training..."
    echo "  data-dir : $DATA_DIR"
    echo "  version  : $RUN_VERSION"
    echo "  outdir   : $OUTDIR"
    echo "  split    : $SPLIT_JSON (id=$SPLIT_ID)"
    echo

    run_train \
      --dataset "$DATASET" \
      --data-dir "$DATA_DIR" \
      --outdir "$OUTDIR" \
      --split-json "$SPLIT_JSON" \
      --split-id "$SPLIT_ID" \
      --lookback "$LOOKBACK" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --hidden-dim "$HIDDEN_DIM" \
      --num-layers "$NUM_LAYERS" \
      --dropout "$DROPOUT" \
      --pretrain-forecaster-epochs "$PRETRAIN_FORECASTER_EPOCHS" \
      --pretrain-marginal-epochs "$PRETRAIN_MARGINAL_EPOCHS" \
      --gate-epochs "$GATE_EPOCHS" \
      --gate-lr "$GATE_LR" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
      --use-mixture-nll \
      --include-gaussian-const \
      --gate-aux-weight "$GATE_AUX_WEIGHT" \
      --gate-aux-temp "$GATE_AUX_TEMP" \
      --gate-aux-margin "$GATE_AUX_MARGIN" \
      --gate-l1 "$GATE_L1" \
      --marginal-type "$MARGINAL_TYPE" \
      --marginal-hidden "$MARGINAL_HIDDEN" \
      --marginal-bins "$MARGINAL_BINS" \
      --marginal-tail "$MARGINAL_TAIL" \
      --marginal-subsample "$MARGINAL_SUBSAMPLE" \
      --seed "$SEED" \
      "$@"
    ;;
esac
