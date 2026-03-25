"""Configuration for the standalone HCRL false-positive analysis bundle."""

from pathlib import Path
import os
import torch
import random
import numpy as np

from fp_analysis.path_config import get_path
from fp_analysis.splits.paths import HCRL_SPLITS_DIR

ROOT = Path(__file__).resolve().parent

# Location of the raw trip CSVs
DATA_DIR = get_path("HCRL_FP_DATA_DIR", fallback_var="HCRL_DATA_DIR")

ALL_DRIVERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
TRIPS = ["1", "2"]   # "1" = there, "2" = back

# Split ratios for the seen-driver trip segments
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TEST_FRAC = 0.20

META_COLS = ["Time(s)", "Class", "PathOrder"]

# Which driver splits to use and how many
NUM_SEEN_OPTIONS = [3, 5, 7]
NUM_SPLITS_PER_CONFIG = 10

# Directory for saved splits & results
SPLITS_DIR = HCRL_SPLITS_DIR
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
WINDOW_SIZE = 30
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

PATIENCE = 20
#MIN_DELTA = 1e-5
MIN_DELTA = 1e-6

#VAL_QUANTILE = 0.975
VAL_QUANTILE = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 88

# Ignored features
IGNORED_FEATURES = [
    "Target_engine_speed_used_in_lock-up_module",
    "Inhibition_of_engine_fuel_cut_off",
    "Torque_scaling_factor(standardization)",
    "Glow_plug_control_request",
    "Engine_soacking_time",
    "Requested_spark_retard_angle_from_TCU",
    "Standard_Torque_Ratio",
    "TCU_requested_engine_RPM_increase",
    "TCU_requests_engine_torque_limit_(ETL)",
    "Filtered_Accelerator_Pedal_value",
    "Engine_coolant_temperature",
    "Engine_coolant_temperature.1",
    "Activation_of_Air_compressor"
]

# Human-controlled features
HUMAN_CONTROLLED_FEATURES = [
    "Accelerator_Pedal_value",
    "Indication_of_brake_switch_ON/OFF",
    "Master_cylinder_pressure",
    "Steering_wheel_angle",
    "Steering_wheel_speed",
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
