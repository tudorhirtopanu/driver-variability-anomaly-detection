#!/usr/bin/env bash
set -euo pipefail

# Dataset-aware wrapper around nominal FPR evaluation.
# This keeps the saved-checkpoint paths and per-dataset defaults together
# rather than scattering them across ad-hoc shell commands.

usage() {
  echo "Usage:"
  echo "  scripts/run_fpr_eval.sh <hcrl|sonata|obd> [extra eval args...]"
  echo
  echo "Notes:"
  echo "  Pass extra Python arguments after --"
  echo
  echo "Examples:"
  echo "  scripts/run_fpr_eval.sh hcrl"
  echo "  scripts/run_fpr_eval.sh sonata -- --gated-ckpt /path/to/gated_model.pt"
  echo "  scripts/run_fpr_eval.sh obd -- --data-dir /path/to/obd --split-json /path/to/splits_obd.json"
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

run_eval() {
  PYTHONPATH="$PYTHONPATH_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m evaluation.fpr_eval.run_fpr_eval "$@"
}

LOOKBACK=20
NUM_LAYERS=2
DROPOUT=0.1
GATED_NUM_LAYERS=2
GATED_DROPOUT=0.1
GAUSSIAN_EPOCHS=50
GAUSSIAN_PATIENCE=10
MSE_EPOCHS=50
MSE_PATIENCE=10
QUANTILES="0.95,0.99,0.999"
SEED=42

case "$DATASET" in
  hcrl)
    : "${HCRL_DATA_DIR:?Set HCRL_DATA_DIR in scripts/local_paths.sh or your shell}"
    DATA_DIR="$HCRL_DATA_DIR"
    FPR_RUN_ID="${HCRL_FPR_RUN_ID:-fpr_comparison_v5}"
    GATED_RUN_ID="${HCRL_GATED_RUN_ID:-gated_baseline_v5}"
    OUTDIR="${HCRL_FPR_OUTDIR:-$ROOT_DIR/results/hcrl/fpr_comparison/$FPR_RUN_ID}"
    TRAIN_DRIVERS="${HCRL_TRAIN_DRIVERS:-B,D,F,H,E,C}"
    VAL_DRIVERS="${HCRL_VAL_DRIVERS:-A,I}"
    TEST_DRIVERS="${HCRL_TEST_DRIVERS:-G,J}"
    GATED_CKPT="${HCRL_GATED_CKPT:-$ROOT_DIR/results/hcrl/checkpoints/$GATED_RUN_ID/gated_model.pt}"
    GAUSSIAN_CKPT="${HCRL_GAUSSIAN_CKPT:-$ROOT_DIR/results/hcrl/fpr_comparison/$FPR_RUN_ID/gaussian_lstm.pt}"
    MSE_CKPT="${HCRL_MSE_CKPT:-$ROOT_DIR/results/hcrl/fpr_comparison/$FPR_RUN_ID/mse_lstm.pt}"

    BATCH_SIZE=128
    NUM_WORKERS=8
    HIDDEN_DIM=256
    GATED_HIDDEN_DIM=256
    QUANTILES="0.95,0.99,0.999"

    echo "Running HCRL FPR evaluation..."
    echo "  data-dir      : $DATA_DIR"
    echo "  run id        : $FPR_RUN_ID"
    echo "  outdir        : $OUTDIR"
    echo "  train drivers : $TRAIN_DRIVERS"
    echo "  val drivers   : $VAL_DRIVERS"
    echo "  test drivers  : $TEST_DRIVERS"
    echo "  ckpt          : $GATED_CKPT"
    echo

    run_eval \
      --dataset "$DATASET" \
      --data-dir "$DATA_DIR" \
      --outdir "$OUTDIR" \
      --train-drivers "$TRAIN_DRIVERS" \
      --val-drivers "$VAL_DRIVERS" \
      --test-drivers "$TEST_DRIVERS" \
      --gated-ckpt "$GATED_CKPT" \
      --gaussian-ckpt "$GAUSSIAN_CKPT" \
      --mse-ckpt "$MSE_CKPT" \
      --lookback "$LOOKBACK" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --hidden-dim "$HIDDEN_DIM" \
      --num-layers "$NUM_LAYERS" \
      --dropout "$DROPOUT" \
      --gated-hidden-dim "$GATED_HIDDEN_DIM" \
      --gated-num-layers "$GATED_NUM_LAYERS" \
      --gated-dropout "$GATED_DROPOUT" \
      --gated-use-mixture-nll \
      --include-gaussian-const \
      --gaussian-epochs "$GAUSSIAN_EPOCHS" \
      --gaussian-patience "$GAUSSIAN_PATIENCE" \
      --mse-epochs "$MSE_EPOCHS" \
      --mse-patience "$MSE_PATIENCE" \
      --quantiles "$QUANTILES" \
      --seed "$SEED" \
      "$@"
    ;;

  sonata)
    : "${SONATA_DATA_DIR:?Set SONATA_DATA_DIR in scripts/local_paths.sh or your shell}"
    DATA_DIR="$SONATA_DATA_DIR"
    RUN_ID="${SONATA_RUN_ID:-v1_test_C}"
    OUTDIR="${SONATA_FPR_OUTDIR:-$ROOT_DIR/results/sonata/fpr_comparison/$RUN_ID}"
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
    GATED_CKPT="${SONATA_GATED_CKPT:-$ROOT_DIR/results/sonata/checkpoints/$RUN_ID/gated_model.pt}"
    GAUSSIAN_CKPT="${SONATA_GAUSSIAN_CKPT:-$ROOT_DIR/results/sonata/fpr_comparison/$RUN_ID/gaussian_lstm.pt}"
    MSE_CKPT="${SONATA_MSE_CKPT:-$ROOT_DIR/results/sonata/fpr_comparison/$RUN_ID/mse_lstm.pt}"

    BATCH_SIZE=64
    NUM_WORKERS=0
    HIDDEN_DIM=192
    GATED_HIDDEN_DIM=192
    QUANTILES="0.99,0.999"

    echo "Running Sonata FPR evaluation..."
    echo "  data-dir      : $DATA_DIR"
    echo "  run id        : $RUN_ID"
    echo "  outdir        : $OUTDIR"
    echo "  train drivers : $TRAIN_DRIVERS"
    echo "  val drivers   : $VAL_DRIVERS"
    echo "  test drivers  : $TEST_DRIVERS"
    echo "  ckpt          : $GATED_CKPT"
    echo

    run_eval \
      --dataset "$DATASET" \
      --data-dir "$DATA_DIR" \
      --outdir "$OUTDIR" \
      --train-drivers "$TRAIN_DRIVERS" \
      --val-drivers "$VAL_DRIVERS" \
      --test-drivers "$TEST_DRIVERS" \
      --gated-ckpt "$GATED_CKPT" \
      --gaussian-ckpt "$GAUSSIAN_CKPT" \
      --mse-ckpt "$MSE_CKPT" \
      --lookback "$LOOKBACK" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --hidden-dim "$HIDDEN_DIM" \
      --num-layers "$NUM_LAYERS" \
      --dropout "$DROPOUT" \
      --gated-hidden-dim "$GATED_HIDDEN_DIM" \
      --gated-num-layers "$GATED_NUM_LAYERS" \
      --gated-dropout "$GATED_DROPOUT" \
      --gated-use-mixture-nll \
      --include-gaussian-const \
      --gaussian-epochs "$GAUSSIAN_EPOCHS" \
      --gaussian-patience "$GAUSSIAN_PATIENCE" \
      --mse-epochs "$MSE_EPOCHS" \
      --mse-patience "$MSE_PATIENCE" \
      --quantiles "$QUANTILES" \
      --seed "$SEED" \
      "$@"
    ;;

  obd)
    : "${OBD_DATA_DIR:?Set OBD_DATA_DIR in scripts/local_paths.sh or your shell}"
    DATA_DIR="$OBD_DATA_DIR"
    RUN_VERSION="${OBD_RUN_VERSION:-v1}"
    OUTDIR="${OBD_FPR_OUTDIR:-$ROOT_DIR/results/obd/fpr_comparison/$RUN_VERSION}"
    SPLIT_JSON="${OBD_SPLIT_JSON:-$ROOT_DIR/fp_analysis/splits/generated/obd/splits_obd.json}"
    SPLIT_ID="${OBD_SPLIT_ID:-0}"
    GATED_CKPT="${OBD_GATED_CKPT:-$ROOT_DIR/results/obd/checkpoints/$RUN_VERSION/gated_model.pt}"
    GAUSSIAN_CKPT="${OBD_GAUSSIAN_CKPT:-$ROOT_DIR/results/obd/fpr_comparison/$RUN_VERSION/gaussian_lstm.pt}"
    MSE_CKPT="${OBD_MSE_CKPT:-$ROOT_DIR/results/obd/fpr_comparison/$RUN_VERSION/mse_lstm.pt}"

    BATCH_SIZE=128
    NUM_WORKERS=8
    HIDDEN_DIM=128
    GATED_HIDDEN_DIM=128
    QUANTILES="0.95,0.99,0.999"

    echo "Running OBD FPR evaluation..."
    echo "  data-dir : $DATA_DIR"
    echo "  version  : $RUN_VERSION"
    echo "  outdir   : $OUTDIR"
    echo "  split    : $SPLIT_JSON (id=$SPLIT_ID)"
    echo "  ckpt     : $GATED_CKPT"
    echo

    run_eval \
      --dataset "$DATASET" \
      --data-dir "$DATA_DIR" \
      --outdir "$OUTDIR" \
      --split-json "$SPLIT_JSON" \
      --split-id "$SPLIT_ID" \
      --gated-ckpt "$GATED_CKPT" \
      --gaussian-ckpt "$GAUSSIAN_CKPT" \
      --mse-ckpt "$MSE_CKPT" \
      --lookback "$LOOKBACK" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --hidden-dim "$HIDDEN_DIM" \
      --num-layers "$NUM_LAYERS" \
      --dropout "$DROPOUT" \
      --gated-hidden-dim "$GATED_HIDDEN_DIM" \
      --gated-num-layers "$GATED_NUM_LAYERS" \
      --gated-dropout "$GATED_DROPOUT" \
      --gated-use-mixture-nll \
      --include-gaussian-const \
      --gaussian-epochs "$GAUSSIAN_EPOCHS" \
      --gaussian-patience "$GAUSSIAN_PATIENCE" \
      --mse-epochs "$MSE_EPOCHS" \
      --mse-patience "$MSE_PATIENCE" \
      --quantiles "$QUANTILES" \
      --seed "$SEED" \
      "$@"
    ;;
esac
