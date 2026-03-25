#!/usr/bin/env bash
set -euo pipefail

# Fixed-gate nominal evaluation for the separate HCRL fixed-gate experiment.

usage() {
  echo "Usage:"
  echo "  scripts/run_fpr_eval_fixed_gate.sh [extra eval args...]"
  echo
  echo "Notes:"
  echo "  HCRL fixed-gate experiment only"
  echo "  Pass extra Python arguments after --"
  echo
  echo "Examples:"
  echo "  scripts/run_fpr_eval_fixed_gate.sh"
  echo "  scripts/run_fpr_eval_fixed_gate.sh -- --fixed-gate-alphas 0.2,0.4,0.6"
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
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
    "$PYTHON_BIN" -m evaluation.fixed_gate.fpr_eval "$@"
}

: "${HCRL_DATA_DIR:?Set HCRL_DATA_DIR in scripts/local_paths.sh or your shell}"
DATA_DIR="$HCRL_DATA_DIR"
FPR_RUN_ID="${HCRL_FIXED_GATE_FPR_RUN_ID:-fpr_comparison_v5_fixed_gate_multi}"
GATED_RUN_ID="${HCRL_GATED_RUN_ID:-gated_baseline_v5}"
OUTDIR="${HCRL_FIXED_GATE_FPR_OUTDIR:-$ROOT_DIR/results/hcrl/fixed_gate_comparison/$FPR_RUN_ID}"
GATED_CKPT="${HCRL_GATED_CKPT:-$ROOT_DIR/results/hcrl/checkpoints/$GATED_RUN_ID/gated_model.pt}"

TRAIN_DRIVERS="${HCRL_TRAIN_DRIVERS:-B,D,F,H,E,C}"
VAL_DRIVERS="${HCRL_VAL_DRIVERS:-A,I}"
TEST_DRIVERS="${HCRL_TEST_DRIVERS:-G,J}"

LOOKBACK=20
BATCH_SIZE=128
NUM_WORKERS=8
GATED_HIDDEN_DIM=256
GATED_NUM_LAYERS=2
GATED_DROPOUT=0.1
FIXED_GATE_ALPHAS="${HCRL_FIXED_GATE_ALPHAS:-0.3,0.5,0.7}"
QUANTILES="0.95,0.97,0.99,0.999"
SEED=42

echo "Running HCRL fixed-gate FPR evaluation..."
echo "  data-dir      : $DATA_DIR"
echo "  run id        : $FPR_RUN_ID"
echo "  outdir        : $OUTDIR"
echo "  train drivers : $TRAIN_DRIVERS"
echo "  val drivers   : $VAL_DRIVERS"
echo "  test drivers  : $TEST_DRIVERS"
echo "  ckpt          : $GATED_CKPT"
echo

run_eval \
  --data-dir "$DATA_DIR" \
  --outdir "$OUTDIR" \
  --train-drivers "$TRAIN_DRIVERS" \
  --val-drivers "$VAL_DRIVERS" \
  --test-drivers "$TEST_DRIVERS" \
  --gated-ckpt "$GATED_CKPT" \
  --lookback "$LOOKBACK" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --gated-hidden-dim "$GATED_HIDDEN_DIM" \
  --gated-num-layers "$GATED_NUM_LAYERS" \
  --gated-dropout "$GATED_DROPOUT" \
  --gated-use-mixture-nll \
  --include-gaussian-const \
  --fixed-gate-alphas "$FIXED_GATE_ALPHAS" \
  --quantiles "$QUANTILES" \
  --seed "$SEED" \
  "$@"
