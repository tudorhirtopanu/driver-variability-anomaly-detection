# Driver Variability Anomaly Detection

This is a dissertation codebase for multivariate time-series anomaly detection on the HCRL, Sonata, and OBD datasets.

It contains:

- `src/`: main U-GMM training and evaluation code
- `scripts/`: shell entrypoints for training, nominal evaluation, synthetic injection, and synthetic evaluation
- `fp_analysis/`: separate false-positive study code
- `results/`: retained checkpoints and evaluation outputs
- `synthetic_data/`: retained injected datasets and anomaly logs

## Basic Usage

Set dataset paths in [`scripts/local_paths.sh`](scripts/local_paths.sh), create a virtual environment, install [`requirements.txt`](requirements.txt), then run the wrapper scripts from the repository root.

Main entrypoints:

```bash
bash scripts/train.sh hcrl
bash scripts/run_fpr_eval.sh sonata
bash scripts/run_inject_synth.sh obd
bash scripts/run_synth_eval.sh obd
bash scripts/generate_fp_splits.sh hcrl
```

For false-positive study runs:

```bash
.venv/bin/python fp_analysis/hcrl/run_experiments.py --model pca
.venv/bin/python fp_analysis/sonata/run_experiments.py --model pca
.venv/bin/python fp_analysis/obd/run_experiments.py --model pca
```

## Docs

- [`docs/experiments.md`](docs/EXPERIMENTS.md): brief run instructions
- [`ARTIFACTS.md`](ARTIFACTS.md): what saved outputs mean

## Datasets

This repository uses the following datasets:

- **HCRL dataset** — [Driving Dataset (HCRL)](https://sites.google.com/hksecurity.net/hcrl/Datasets/driving-dataset). Kwak, B. I., Woo, J., and Kim, H. K. *Know Your Master: Driver Profiling-based Anti-theft Method*. PST 2016.
- **Sonata dataset** — [This Car is Mine!: Driver Pattern Dataset Extracted from CAN-bus](https://ieee-dataport.org/open-access/car-mine-driver-pattern-dataset-extracted-can-bus). Park, K. H., Kwak, B. I., and Kim, H. K. IEEE Dataport, 2020.
- **OBD dataset** — [Automotive OBD-II Dataset](https://radar.kit.edu/radar/en/dataset/bCtGxdTklQlfQcAq). Weber, M. *Automotive OBD-II Dataset*. RADAR4KIT, 2023.
