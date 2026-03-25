"""HCRL-specific model hyperparameters for the shared false-positive-study factory."""

from fp_analysis.hcrl.config import (
    DIM_FEEDFORWARD,
    D_MODEL,
    DROPOUT,
    HIDDEN_DIM,
    LATENT_DIM,
    NHEAD,
    NUM_LAYERS,
    NUM_TRANS_LAYERS,
    PCA_COMPONENTS,
    TCN_CHANNELS,
    TCN_DROPOUT,
    TCN_KERNEL_SIZE,
    USAD_HIDDEN_DIM,
    USAD_LATENT_DIM,
    WINDOW_SIZE,
)
from fp_analysis.shared.model_defs import ModelConfig

MODEL_CONFIG = ModelConfig(
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    num_layers=NUM_LAYERS,
    window_size=WINDOW_SIZE,
    pca_components=PCA_COMPONENTS,
    tcn_channels=tuple(TCN_CHANNELS),
    tcn_kernel_size=TCN_KERNEL_SIZE,
    tcn_dropout=TCN_DROPOUT,
    d_model=D_MODEL,
    nhead=NHEAD,
    dim_feedforward=DIM_FEEDFORWARD,
    num_transformer_layers=NUM_TRANS_LAYERS,
    dropout=DROPOUT,
    usad_hidden_dim=USAD_HIDDEN_DIM,
    usad_latent_dim=USAD_LATENT_DIM,
)
