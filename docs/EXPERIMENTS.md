# Experiments

This file is a short runbook for the main experiment entry points in this repository.

Run all commands from the repository root. The shell wrappers prefer `.venv/bin/python` when it exists.

## Dataset Paths

The wrappers load dataset paths from `scripts/local_paths.sh`. You can also override them in your shell before running commands:

```bash
export HCRL_DATA_DIR="/path/to/hcrl"
export SONATA_DATA_DIR="/path/to/sonata"
export OBD_DATA_DIR="/path/to/obd"
```

The false-positive analysis bundle can also be pointed at explicit dataset roots:

```bash
export HCRL_FP_DATA_DIR="$HCRL_DATA_DIR"
export SONATA_FP_DATA_DIR="$SONATA_DATA_DIR"
export OBD_FP_DATA_DIR="$OBD_DATA_DIR"
export OBD_FP_SPLIT_JSON="fp_analysis/splits/generated/obd/splits_obd.json"
```

## Train The Main U-GMM Models

Use the dataset-aware training wrapper:

```bash
bash scripts/train.sh hcrl
bash scripts/train.sh sonata
bash scripts/train.sh obd
```

These runs write checkpoints under `results/<dataset>/checkpoints/` unless you override the output directory with variables such as `HCRL_OUTDIR`, `SONATA_OUTDIR`, or `OBD_OUTDIR`.

You can pass extra training arguments after `--`:

```bash
bash scripts/train.sh sonata -- --seed 123
```

## Evaluate The Main U-GMM Models

Nominal false-positive-rate evaluation:

```bash
bash scripts/run_fpr_eval.sh hcrl
bash scripts/run_fpr_eval.sh sonata
bash scripts/run_fpr_eval.sh obd
```

These commands compare the gated model against the Gaussian and MSE baselines and write outputs under `results/<dataset>/fpr_comparison/`.

Synthetic anomaly workflow:

```bash
bash scripts/run_inject_synth.sh hcrl
bash scripts/run_inject_synth.sh sonata
bash scripts/run_inject_synth.sh obd

bash scripts/run_synth_eval.sh hcrl
bash scripts/run_synth_eval.sh sonata
bash scripts/run_synth_eval.sh obd
```

`run_inject_synth.sh` creates or refreshes synthetic test data under `synthetic_data/`. `run_synth_eval.sh` evaluates the trained models on those injected files and writes results under `results/<dataset>/synth_evaluation/`.

HCRL also has a fixed-gate ablation:

```bash
bash scripts/run_fpr_eval_fixed_gate.sh
bash scripts/run_synth_eval_fixed_gate.sh
```

## Run The False-Positive Study

The false-positive study lives under `fp_analysis/` and is separate from the main U-GMM pipeline.

First generate the split files:

```bash
bash scripts/generate_fp_splits.sh hcrl
bash scripts/generate_fp_splits.sh sonata
bash scripts/generate_fp_splits.sh obd
```

Then run the per-dataset experiment drivers:

```bash
.venv/bin/python fp_analysis/hcrl/run_experiments.py --model pca
.venv/bin/python fp_analysis/sonata/run_experiments.py --model pca
.venv/bin/python fp_analysis/obd/run_experiments.py --model pca
```

Available model names include:

```text
lstm
transformer
transformer_forecast
lstm_forecast
lstm_vae
pca
tcn
persistence
usad
```

The HCRL and Sonata runners iterate over the saved driver splits in `fp_analysis/splits/generated/`. The OBD runner uses the generated file-list split JSON and supports extra flags such as `--num-splits` and `--out-dir`.

## Quick Checks

Use the following commands to print usage information about the scripts:

```bash
bash scripts/train.sh --help
bash scripts/run_fpr_eval.sh --help
bash scripts/run_synth_eval.sh --help
bash scripts/run_inject_synth.sh --help
bash scripts/generate_fp_splits.sh --help
.venv/bin/python fp_analysis/hcrl/run_experiments.py --help
.venv/bin/python fp_analysis/sonata/run_experiments.py --help
.venv/bin/python fp_analysis/obd/run_experiments.py --help
```
