# Artifacts

This repository keeps a representative subset of experiment outputs so the main results can be inspected without rerunning the full pipeline.

## Main U-GMM Artifacts

The `results/` tree contains retained outputs from the main U-GMM training and evaluation workflow.

### Checkpoints

Saved under:

- `results/hcrl/checkpoints/`
- `results/sonata/checkpoints/`
- `results/obd/checkpoints/`

Typical files:

- `gated_model.pt`: trained gated detector checkpoint
- `alpha_mean_per_feature_test_final.csv`: mean gate weight per feature on the retained test split
- `per_feature_summary_test_final.json`: per-feature score summary on the test split
- Sonata also keeps `args.json` and `summary.json` for the run configuration and final summary

These artifacts show the trained model state and the learned feature-level gate behaviour for each retained run.

### Nominal False-Positive Evaluation

Saved under:

- `results/hcrl/fpr_comparison/`
- `results/sonata/fpr_comparison/`
- `results/obd/fpr_comparison/`
- `results/hcrl/fixed_gate_comparison/`

Typical files:

- `fpr_curve.csv`: false-positive rate measured at the configured score quantiles
- `fpr_summary.json`: compact summary of the nominal evaluation
- `gaussian_lstm.pt`: trained Gaussian baseline forecaster used for comparison
- `mse_lstm.pt`: trained MSE baseline forecaster used for comparison

These artifacts show how the gated model compares with the simpler forecasting baselines on nominal-only data. The fixed-gate HCRL folders store the same type of summary for fixed mixing weights instead of the learned gate.

### Synthetic Evaluation

Saved under:

- `results/hcrl/synth_evaluation/`
- `results/sonata/synth_evaluation/`
- `results/obd/synth_evaluation/`

Representative files:

- HCRL: `v1.csv` ... `v5.csv`
  - columns include `model`, `file`, `auroc`, `best_threshold`, `adj_f1`, `adj_precision`, `adj_recall`, and confusion-count columns such as `tp`, `fp`, `fn`, `tn`
- Sonata:
  - `per_file_metrics.csv`: per-file synthetic anomaly metrics
  - `split_summary.csv`: per-run or per-split summary
  - `overall_summary.csv`: cross-run summary with mean and standard deviation
- OBD:
  - `summary.csv`: overall summary across retained OBD synthetic-eval runs
  - `v*/per_file_metrics.csv`: per-file oracle-style synthetic metrics for each split
  - `v*/oracle_split_summary.csv`: per-split aggregate summary

These artifacts show how well each model detects injected anomalies, including AUROC, PR-AUC, best-threshold F1, and event-adjusted metrics.

### Retained Summary Tables

Two additional retained summary tables are worth calling out:

- `results/sonata/fpr_comparison/fpr_gain_summary.csv`
  - columns: `quantile`, `model`, `mean_fpr`, `std_fpr`, `gain_vs_gated`
  - shows how much nominal FPR changes relative to the gated model
- `results/obd/synth_evaluation/summary.csv`
  - stores cross-split aggregate synthetic-detection metrics for the retained OBD runs

## Synthetic Datasets

The `synthetic_data/` tree contains injected test data that can be reused for synthetic evaluation.

### HCRL

Saved under folders such as:

- `synthetic_data/hcrl/HCRL_synth_C_I/`
- `synthetic_data/hcrl/HCRL_synth_F_H/`
- `synthetic_data/hcrl/HCRL_synth_G_I/`
- `synthetic_data/hcrl/HCRL_synth_G_J/`

Each folder contains:

- injected trip CSVs
- `anomaly_log.csv`

The folder name identifies which driver pair was used for the synthetic test set.

### Sonata

Saved under:

- `synthetic_data/sonata/A/`
- `synthetic_data/sonata/B/`
- `synthetic_data/sonata/C/`
- `synthetic_data/sonata/D/`
- `synthetic_data/sonata/anomaly_log.csv`

These are injected versions of the retained Sonata trips, grouped by driver.

### OBD

Saved under:

- `synthetic_data/obd/`
- `synthetic_data/obd/anomaly_log.csv`

This folder contains injected OBD CSV files and the anomaly log used during synthetic evaluation.

## False-Positive Study Artifacts

The standalone false-positive study under `fp_analysis/` keeps generated split files and aggregated result bundles.

### Generated Splits

Saved under:

- `fp_analysis/splits/generated/hcrl/splits_config.json`
- `fp_analysis/splits/generated/sonata/splits_config.json`
- `fp_analysis/splits/generated/obd/splits_obd.json`

These files define the fixed train/validation/test partitions used by the false-positive study.

- HCRL and Sonata split files store seen-driver vs unseen-driver configurations
- OBD split files store explicit train/val/test file lists, along with row-count and fraction metadata

### Per-Split Summaries

Saved under:

- `fp_analysis/hcrl/results/summaries_*.json`
- `fp_analysis/sonata/results/summaries_*.json`
- `fp_analysis/obd/results/summaries_*.json`

These are lists of per-split results. Typical contents include:

- nominal false-positive rate
- feature-wise mean error on false-positive windows
- grouped error statistics for human-controlled vs non-human or system features
- inflation or delta values comparing FP windows with non-FP windows

They are the most detailed retained outputs from the false-positive study.

### Aggregate Summaries

Saved under:

- `fp_analysis/hcrl/results/aggregate_*.json`
- `fp_analysis/sonata/results/aggregate_*.json`
- `fp_analysis/obd/results/aggregate_*.json`

These aggregate the per-split summaries into cross-split means and standard deviations.

Representative contents:

- HCRL and Sonata:
  - `fpr_seen_mean`, `fpr_unseen_mean`
  - grouped MSE and share summaries for human-controlled vs non-human features
  - mean feature-wise FP MSE across splits
- OBD:
  - `fpr_test_mean`
  - grouped human vs system FP and non-FP MSE
  - inflation metrics
  - average feature-wise FP and non-FP MSE

These aggregate files are the easiest place to inspect the headline false-positive study outcomes without reading every split summary.

## What Is Retained Versus Omitted

The repository keeps the artifacts that are most useful for inspection and reproduction:

- trained checkpoints for the main gated model
- trained baseline checkpoints used for nominal comparison
- compact JSON and CSV summaries for nominal and synthetic evaluation
- reusable synthetic datasets and anomaly logs
- generated split files and summary bundles for the false-positive study

It does not keep every intermediate training artifact from every run. Full reruns may generate additional logs, histories, or temporary files that are not checked in here.
