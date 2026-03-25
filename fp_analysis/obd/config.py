"""Configuration for the standalone OBD false-positive analysis bundle."""

from pathlib import Path
import os
import random

import numpy as np
import torch

from fp_analysis.path_config import get_path
from fp_analysis.splits.paths import OBD_SPLITS_JSON


ROOT = Path(__file__).resolve().parent
DATA_DIR = get_path("OBD_FP_DATA_DIR", fallback_var="OBD_DATA_DIR")
SPLIT_JSON = Path(
    os.environ.get(
        "OBD_FP_SPLIT_JSON",
        str(OBD_SPLITS_JSON),
    )
)
RESULTS_ROOT = Path(
    os.environ.get(
        "OBD_FP_RESULTS_ROOT",
        str(ROOT),
    )
)

META_COLS = ["Time", "Time(s)", "Class", "PathOrder"]
IGNORED_FEATURES = [
    "Absolute Throttle Position [%]",
]

HUMAN_CONTROLLED_FEATURES = [
    "Accelerator Pedal Position D [%]",
    "Accelerator Pedal Position E [%]",
]

DEFAULT_NUM_SPLITS = 10

MODEL_TYPES = [
    "lstm",
    "transformer",
    "transformer_forecast",
    "lstm_forecast",
    "lstm_vae",
    "pca",
    "tcn",
    "persistence",
    "naive",
    "usad",
]

WINDOW_SIZE = 20
STRIDE = 1
BATCH_SIZE = 512
EPOCHS = 200
LR = 1e-3
PATIENCE = 20
MIN_DELTA = 1e-6
VAL_QUANTILE = 0.99

HIDDEN_DIM = 96
LATENT_DIM = 32
NUM_LAYERS = 1
VAE_BETA = 1e-3

PCA_COMPONENTS = 16

TCN_CHANNELS = [64, 64]
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

USAD_HIDDEN_DIM = 128
USAD_LATENT_DIM = 32
USAD_ALPHA = 0.5
USAD_BETA = 0.5

D_MODEL = 96
NHEAD = 4
NUM_TRANS_LAYERS = 3
DIM_FEEDFORWARD = 192
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 88


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
