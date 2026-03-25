"""Configuration for the standalone Sonata false-positive analysis bundle."""

from pathlib import Path
import os
import torch
import random
import numpy as np

from fp_analysis.path_config import get_path
from fp_analysis.splits.paths import SONATA_SPLITS_DIR

ROOT = Path(__file__).resolve().parent
DATA_DIR = get_path("SONATA_FP_DATA_DIR", fallback_var="SONATA_DATA_DIR")

# Drivers are inferred from the folders inside data/ (A, B, C, D) when the
# local data bundle is present.
ALL_DRIVERS = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()]) if DATA_DIR.exists() else []

# Split ratios for the seen-driver trip segments
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TEST_FRAC = 0.20

META_COLS = ["Time(s)", "Class", "PathOrder"]

# Cross-driver leave-one-out: 3 seen, 1 unseen (all 4 permutations)
NUM_SEEN_OPTIONS = [3]
NUM_SPLITS_PER_CONFIG = 4

# Directory for saved splits & results
SPLITS_DIR = SONATA_SPLITS_DIR
RESULTS_DIR = ROOT / "results"

# Allowed model types
MODEL_TYPES = [
    "lstm",
    "transformer",
    "transformer_forecast",
    "lstm_forecast",
    "lstm_vae",
    "pca",
    "tcn",
    "persistence",
    "naive",  # alias for persistence
    "usad",
]

# Windowing / training
WINDOW_SIZE = 60
STRIDE = 1
BATCH_SIZE = 512
EPOCHS = 200
LR = 1e-3

# Sequence model hyperparameters
HIDDEN_DIM = 96
LATENT_DIM = 32
NUM_LAYERS = 1
VAE_BETA = 1e-3

# PCA baseline
PCA_COMPONENTS = 16

# TCN Hyperparameters
TCN_CHANNELS = [64, 64]
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

# USAD (KDD'20) Hyperparameters
USAD_HIDDEN_DIM = 128
USAD_LATENT_DIM = 32
USAD_ALPHA = 0.5
USAD_BETA = 0.5

PATIENCE = 5
MIN_DELTA = 1e-5

VAL_QUANTILE = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 26

# Ignored features
IGNORED_FEATURES = [
# Strictly Constant (Zero Variance)
    'fuel_pressure',
    'flywheel_torque',
    'glow_plug_limit_request',
    'accelerator_position_filtered',
    'flywheel_torque_revised',
    'engine_pressure_maintanance_time',
    'clutch_check',

    # Broken / Default Sensor Values (Constant across all files)
    'logitude_acceleration',   # Typo in dataset (Longitude), constant value
    'latitude_acceleration',   # Constant value
    'brake_sylinder',          # Typo in dataset (Cylinder), constant value

    # Quasi-Constant (Mostly Noise)
    'compressor_activation',   # Constant >99% of the time
    'reduce_block_fuel',       # Constant >98% of the time
    'block_fuel',

    "road_slope",
    "engine_velocity_increase_tcu",
    "target_engine_velocity_lockup",

    "fire_angle_delay_tcu",
    "torque_transform_coeff",
    "engine_torque_limit_tcu",

    "long_fuel_bank",
    "short_fuel_bank",
    "standard_torque_ratio",

    "current_gear",
    "gear_choice",

    "engine_torque_max",
    "mission_oil_temp"
]

# Human-controlled features
HUMAN_CONTROLLED_FEATURES = [
    "accelerator_position",
    "brake_switch",
    "steering_wheel_angle",
    "steering_wheel_acceleration"
]

def set_global_seeds(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------
# Transformer-specific hyperparameters
# --------------------------------------------------------------------
# Transformer-specific hyperparameters
D_MODEL = 96            # internal transformer dimension
NHEAD = 4               # number of attention heads
NUM_TRANS_LAYERS = 3    # number of encoder layers
DIM_FEEDFORWARD = 192   # FFN hidden dim
DROPOUT = 0.1
