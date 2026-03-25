"""Shared model factory for the false-positive analysis bundles."""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

import torch.nn as nn

from fp_analysis.shared.model_defs import ModelConfig, build_model as build_shared_model

ModelType = Literal[
    "lstm",
    "transformer",
    "transformer_forecast",
    "forecaster",
    "lstm_forecast",
    "lstm_seq2one",
    "lstm_vae",
    "pca",
    "tcn",
    "persistence",
    "naive",
    "usad",
]

MODEL_TYPE_ALIASES = {
    "forecaster": "transformer_forecast",
    "lstm_seq2one": "lstm_forecast",
}


def canonicalize_model_type(model_type: str) -> str:
    """Normalize aliases to the canonical shared model identifier."""
    return MODEL_TYPE_ALIASES.get(model_type.lower(), model_type.lower())


def build_model(
    model_type: ModelType | str,
    input_dim: int,
    *,
    config: ModelConfig,
    allowed_model_types: Collection[str],
) -> nn.Module:
    """Build a shared model after validating against the bundle's allowed types."""
    model_key = canonicalize_model_type(model_type)
    if model_key not in allowed_model_types:
        raise ValueError(
            f"Unknown model_type={model_type}, expected one of {sorted(allowed_model_types)}"
        )
    return build_shared_model(model_key, input_dim=input_dim, config=config)
