#!/usr/bin/env bash
set -euo pipefail

# Fixed-gate synthetic evaluation stays separate from the main synth dispatcher
# because it belongs to the dedicated fixed-gate experiment family.

usage() {
  echo "Usage:"
  echo "  scripts/run_synth_eval_fixed_gate.sh [extra eval args...]"
  echo
  echo "Notes:"
  echo "  HCRL fixed-gate experiment only"
  echo "  Pass extra Python arguments after --"
  echo
  echo "Examples:"
  echo "  scripts/run_synth_eval_fixed_gate.sh"
  echo "  scripts/run_synth_eval_fixed_gate.sh -- --fixed-gate-alphas 0.25,0.5,0.75"
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
    "$PYTHON_BIN" -m evaluation.fixed_gate.synth_eval "$@"
}

RUN_ID="${HCRL_FIXED_GATE_SYNTH_RUN_ID:-v5_fixed_gate_multi}"
GATED_RUN_ID="${HCRL_GATED_RUN_ID:-gated_baseline_v5}"
: "${HCRL_DATA_DIR:?Set HCRL_DATA_DIR in scripts/local_paths.sh or your shell}"
TRAIN_DATA_DIR="$HCRL_DATA_DIR"
TEST_DATA_DIR="${HCRL_SYNTH_DATA_DIR:-$ROOT_DIR/synthetic_data/hcrl/HCRL_synth_G_J}"
OUTDIR="${HCRL_FIXED_GATE_SYNTH_OUTDIR:-$ROOT_DIR/results/hcrl/fixed_gate_synth_evaluation/$RUN_ID}"
GATED_CKPT="${HCRL_GATED_CKPT:-$ROOT_DIR/results/hcrl/checkpoints/$GATED_RUN_ID/gated_model.pt}"

MODELS="gated"
TRAIN_DRIVERS="${HCRL_TRAIN_DRIVERS:-B,D,F,H,E,C}"
VAL_DRIVERS="${HCRL_VAL_DRIVERS:-A,I}"
TEST_DRIVERS="${HCRL_TEST_DRIVERS:-G,J}"

LOOKBACK=20
BATCH_SIZE=128
NUM_WORKERS=8
HIDDEN_DIM=256
NUM_LAYERS=2
DROPOUT=0.1
GATED_HIDDEN_DIM=256
GATED_NUM_LAYERS=2
GATED_DROPOUT=0.1
FIXED_GATE_ALPHAS="${HCRL_FIXED_GATE_ALPHAS:-0.3,0.5,0.7}"
SEED=42

echo "Running HCRL fixed-gate synthetic evaluation..."
echo "  train data : $TRAIN_DATA_DIR"
echo "  test data  : $TEST_DATA_DIR"
echo "  run id     : $RUN_ID"
echo "  outdir     : $OUTDIR"
echo "  ckpt       : $GATED_CKPT"
echo

run_eval \
  --models "$MODELS" \
  --gated-score model_score \
  --mode mixture \
  --train-data-dir "$TRAIN_DATA_DIR" \
  --test-data-dir "$TEST_DATA_DIR" \
  --outdir "$OUTDIR" \
  --gated-ckpt "$GATED_CKPT" \
  --train-drivers "$TRAIN_DRIVERS" \
  --val-drivers "$VAL_DRIVERS" \
  --test-drivers "$TEST_DRIVERS" \
  --lookback "$LOOKBACK" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --gated-hidden-dim "$GATED_HIDDEN_DIM" \
  --gated-num-layers "$GATED_NUM_LAYERS" \
  --gated-dropout "$GATED_DROPOUT" \
  --include-gaussian-const \
  --fixed-gate-alphas "$FIXED_GATE_ALPHAS" \
  --seed "$SEED" \
  "$@"
