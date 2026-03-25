#!/usr/bin/env bash
set -euo pipefail

# Dispatcher for dataset-specific synthetic evaluation modules.
# The Python implementations stay separate because HCRL, Sonata, and OBD do
# not share the same synthetic-evaluation workflow cleanly.

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

TARGET="${SYNTH_EVAL_TARGET:-}"
SCRIPT_LABEL="${SYNTH_EVAL_LABEL:-scripts/run_synth_eval.sh}"

run_module() {
  local module="$1"
  shift
  PYTHONPATH="$PYTHONPATH_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m "$module" "$@"
}

usage_command() {
  local target="$1"
  if [[ "$SCRIPT_LABEL" == "scripts/run_synth_eval.sh" ]]; then
    printf "%s %s" "$SCRIPT_LABEL" "$target"
  else
    printf "%s" "$SCRIPT_LABEL"
  fi
}

print_general_usage() {
  cat <<EOF
Usage:
  scripts/run_synth_eval.sh <hcrl|sonata|obd> [extra eval args...]

Notes:
  Pass extra Python arguments after --

Examples:
  scripts/run_synth_eval.sh hcrl
  scripts/run_synth_eval.sh sonata -- --run-type base
  scripts/run_synth_eval.sh obd -- --first-n 1
EOF
}

print_target_usage() {
  local target="$1"
  local cmd
  cmd="$(usage_command "$target")"
  case "$target" in
    hcrl)
      cat <<EOF
Usage:
  $cmd [extra eval args...]

Notes:
  Pass extra Python arguments after --

Examples:
  $cmd
  $cmd -- --models gated,gaussian
EOF
      ;;
    sonata)
      cat <<EOF
Usage:
  $cmd [extra eval args...]

Notes:
  Pass extra Python arguments after --

Examples:
  $cmd
  $cmd -- --run-type base
EOF
      ;;
    obd)
      cat <<EOF
Usage:
  $cmd [extra eval args...]

Notes:
  Pass extra Python arguments after --

Examples:
  $cmd
  $cmd -- --first-n 1
EOF
      ;;
  esac
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    hcrl|sonata|obd)
      if [[ -z "$TARGET" ]]; then
        TARGET="$1"
        shift
      fi
      ;;
  esac
fi

if [[ -z "$TARGET" ]]; then
  if [[ $# -eq 0 || ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    print_general_usage
    exit 0
  fi
  echo "Error: target must be one of: hcrl, sonata, obd"
  print_general_usage
  exit 1
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  print_target_usage "$TARGET"
  exit 0
fi

if [[ ${1:-} == "--" ]]; then
  shift
fi

case "$TARGET" in
  hcrl)
    GATED_RUN_ID="${HCRL_GATED_RUN_ID:-gated_baseline_v5}"
    FPR_RUN_ID="${HCRL_FPR_RUN_ID:-fpr_comparison_v5}"
    SYNTH_RUN_ID="${HCRL_SYNTH_RUN_ID:-v5}"
    : "${HCRL_DATA_DIR:?Set HCRL_DATA_DIR in scripts/local_paths.sh or your shell}"
    TRAIN_DATA_DIR="$HCRL_DATA_DIR"
    TEST_DATA_DIR="${HCRL_SYNTH_DATA_DIR:-$ROOT_DIR/synthetic_data/hcrl/HCRL_synth_G_J}"
    OUTDIR="${HCRL_SYNTH_OUTDIR:-$ROOT_DIR/results/hcrl/synth_evaluation/$SYNTH_RUN_ID.csv}"
    GATED_CKPT="${HCRL_GATED_CKPT:-$ROOT_DIR/results/hcrl/checkpoints/$GATED_RUN_ID/gated_model.pt}"
    GAUSSIAN_CKPT="${HCRL_GAUSSIAN_CKPT:-$ROOT_DIR/results/hcrl/fpr_comparison/$FPR_RUN_ID/gaussian_lstm.pt}"
    MSE_CKPT="${HCRL_MSE_CKPT:-$ROOT_DIR/results/hcrl/fpr_comparison/$FPR_RUN_ID/mse_lstm.pt}"

    MODELS="${HCRL_SYNTH_MODELS:-gated,gaussian,mse}"
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
    SEED=42

    echo "Running HCRL synthetic evaluation..."
    echo "  train data : $TRAIN_DATA_DIR"
    echo "  test data  : $TEST_DATA_DIR"
    echo "  output     : $OUTDIR"
    echo "  gated ckpt : $GATED_CKPT"
    echo

    run_module evaluation.synth_eval.synth_eval_hcrl \
      --models "$MODELS" \
      --gated-score model_score \
      --mode mixture \
      --train-data-dir "$TRAIN_DATA_DIR" \
      --test-data-dir "$TEST_DATA_DIR" \
      --outdir "$OUTDIR" \
      --gated-ckpt "$GATED_CKPT" \
      --gaussian-ckpt "$GAUSSIAN_CKPT" \
      --mse-ckpt "$MSE_CKPT" \
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
      --seed "$SEED" \
      "$@"
    ;;

  sonata)
    : "${SONATA_DATA_DIR:?Set SONATA_DATA_DIR in scripts/local_paths.sh or your shell}"
    DATA_DIR="$SONATA_DATA_DIR"
    SYNTH_NORMAL_DIR="${SONATA_SYNTH_NORMAL_DIR:-${SONATA_SYNTH_DATA_DIR:-$ROOT_DIR/synthetic_data/sonata}}"
    SYNTH_CONTEXT_DIR="${SONATA_SYNTH_CONTEXT_DIR:-}"
    ANOMALY_LOG="${SONATA_SYNTH_LOG:-$ROOT_DIR/synthetic_data/sonata/anomaly_log.csv}"
    AGGREGATE_PATH="${SONATA_AGGREGATE_PATH:-$ROOT_DIR/results/sonata/checkpoints}"
    FPR_ROOT="${SONATA_FPR_ROOT:-$ROOT_DIR/results/sonata/fpr_comparison}"
    OUTDIR="${SONATA_SYNTH_OUTDIR:-$ROOT_DIR/results/sonata/synth_evaluation}"
    RUN_TYPE="${SONATA_RUN_TYPE:-all}"
    RUNS="${SONATA_SYNTH_RUNS:-}"
    SKIP_EXISTING="${SONATA_SYNTH_SKIP_EXISTING:-0}"
    FILTER_SUCCESS="${SONATA_SYNTH_FILTER_SUCCESS:-1}"
    LOOKBACK=20
    BATCH_SIZE=64
    NUM_WORKERS=0
    SEED=42

    COMMON_ARGS=(
      --data-dir "$DATA_DIR"
      --aggregate-path "$AGGREGATE_PATH"
      --run-type "$RUN_TYPE"
      --lookback "$LOOKBACK"
      --batch-size "$BATCH_SIZE"
      --num-workers "$NUM_WORKERS"
      --seed "$SEED"
    )

    if [[ -n "$RUNS" ]]; then
      COMMON_ARGS+=(--runs "$RUNS")
    fi

    if [[ "$SKIP_EXISTING" == "1" ]]; then
      COMMON_ARGS+=(--skip-existing)
    else
      COMMON_ARGS+=(--no-skip-existing)
    fi

    FILTER_ARGS=()
    if [[ "$FILTER_SUCCESS" == "1" && -f "$ANOMALY_LOG" ]]; then
      FILTER_ARGS+=(--anomaly-log "$ANOMALY_LOG" --use-anomaly-log-filter)
    fi

    echo "Running Sonata synthetic evaluation..."
    echo "  data-dir      : $DATA_DIR"
    echo "  aggregate     : $AGGREGATE_PATH"
    echo "  normal synth  : $SYNTH_NORMAL_DIR"
    if [[ -n "$SYNTH_CONTEXT_DIR" ]]; then
      echo "  context synth : $SYNTH_CONTEXT_DIR"
    fi
    if [[ ${#FILTER_ARGS[@]} -gt 0 ]]; then
      echo "  anomaly-log   : $ANOMALY_LOG"
    fi
    echo "  outdir        : $OUTDIR"
    echo

    NORMAL_ARGS=("${COMMON_ARGS[@]}")
    if [[ ${#FILTER_ARGS[@]} -gt 0 ]]; then
      NORMAL_ARGS+=("${FILTER_ARGS[@]}")
    fi
    NORMAL_ARGS+=(--injected-dir "$SYNTH_NORMAL_DIR" --outdir "$OUTDIR/normal" --fpr-root "$FPR_ROOT")

    run_module evaluation.synth_eval.synth_eval_sonata "${NORMAL_ARGS[@]}" "$@"

    if [[ -n "$SYNTH_CONTEXT_DIR" && -d "$SYNTH_CONTEXT_DIR" ]]; then
      CONTEXT_ARGS=("${COMMON_ARGS[@]}")
      if [[ ${#FILTER_ARGS[@]} -gt 0 ]]; then
        CONTEXT_ARGS+=("${FILTER_ARGS[@]}")
      fi
      CONTEXT_ARGS+=(--injected-dir "$SYNTH_CONTEXT_DIR" --outdir "$OUTDIR/contextual" --fpr-root "$FPR_ROOT")
      run_module evaluation.synth_eval.synth_eval_sonata "${CONTEXT_ARGS[@]}" "$@"
    fi
    ;;

  obd)
    : "${OBD_DATA_DIR:?Set OBD_DATA_DIR in scripts/local_paths.sh or your shell}"
    DATA_DIR="$OBD_DATA_DIR"
    AGGREGATE_PATH="${OBD_SYNTH_AGGREGATE_PATH:-$ROOT_DIR/results/obd/fpr_comparison}"
    SPLIT_JSON="${OBD_SPLIT_JSON:-$ROOT_DIR/fp_analysis/splits/generated/obd/splits_obd.json}"
    INJECTED_DIR="${OBD_SYNTH_INJECTED_DIR:-$ROOT_DIR/synthetic_data/obd}"
    ANOMALY_LOG="${OBD_SYNTH_LOG:-$ROOT_DIR/synthetic_data/obd/anomaly_log.csv}"
    OUTDIR="${OBD_SYNTH_OUTDIR:-$ROOT_DIR/results/obd/synth_evaluation}"
    SUB="${OBD_SUB:-0.1}"
    FIRST_N="${OBD_SYNTH_FIRST_N:-}"

    LOOKBACK=20
    BATCH_SIZE=128
    HIDDEN_DIM=128
    NUM_LAYERS=2
    DROPOUT=0.1
    GATED_HIDDEN_DIM=128
    GATED_NUM_LAYERS=2
    GATED_DROPOUT=0.1

    OBD_ARGS=(
      --aggregate-path "$AGGREGATE_PATH"
      --split-json "$SPLIT_JSON"
      --data-dir "$DATA_DIR"
      --injected-dir "$INJECTED_DIR"
      --anomaly-log "$ANOMALY_LOG"
      --outdir "$OUTDIR"
      --sub "$SUB"
      --lookback "$LOOKBACK"
      --batch-size "$BATCH_SIZE"
      --hidden-dim "$HIDDEN_DIM"
      --num-layers "$NUM_LAYERS"
      --dropout "$DROPOUT"
      --gated-hidden-dim "$GATED_HIDDEN_DIM"
      --gated-num-layers "$GATED_NUM_LAYERS"
      --gated-dropout "$GATED_DROPOUT"
      --mode mixture
      --include-gaussian-const
    )
    if [[ -n "$FIRST_N" ]]; then
      OBD_ARGS+=(--first-n "$FIRST_N")
    fi

    echo "Running OBD synthetic evaluation..."
    echo "  data-dir    : $DATA_DIR"
    echo "  aggregate   : $AGGREGATE_PATH"
    echo "  injected    : $INJECTED_DIR"
    echo "  anomaly-log : $ANOMALY_LOG"
    echo "  outdir      : $OUTDIR"
    echo

    run_module evaluation.synth_eval.synth_eval_obd "${OBD_ARGS[@]}" "$@"
    ;;

esac
