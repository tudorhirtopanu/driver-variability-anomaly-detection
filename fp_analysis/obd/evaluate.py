"""Unified evaluation entrypoint for the OBD false-positive study."""

from typing import Dict

from fp_analysis.obd.config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    HUMAN_CONTROLLED_FEATURES,
    IGNORED_FEATURES,
    LR,
    MIN_DELTA,
    MODEL_TYPES,
    PATIENCE,
    STRIDE,
    USAD_ALPHA,
    USAD_BETA,
    VAE_BETA,
    VAL_QUANTILE,
    WINDOW_SIZE,
)
from fp_analysis.obd.data_utils import prepare_split
from fp_analysis.obd.model_config import MODEL_CONFIG
from fp_analysis.obd.common import (
    collect_forecast_feature_mse,
    collect_forecast_scores,
    collect_reconstruction_feature_mse,
    collect_reconstruction_scores,
    run_split_evaluation,
)
from fp_analysis.shared.datasets import (
    build_forecast_dataset,
    build_forecast_dataset_with_ids,
    build_window_dataset,
    build_window_dataset_with_ids,
)
from fp_analysis.shared.model_factory import build_model as build_shared_model
from fp_analysis.shared.train_forecaster import train_forecaster
from fp_analysis.shared.train_reconstruction import TrainingConfig, train_model

TRAINING_CONFIG = TrainingConfig(
    batch_size=BATCH_SIZE,
    device=DEVICE,
    epochs=EPOCHS,
    lr=LR,
    min_delta=MIN_DELTA,
    patience=PATIENCE,
    vae_beta=VAE_BETA,
    usad_alpha=USAD_ALPHA,
    usad_beta=USAD_BETA,
)

FORECAST_MODELS = {
    "transformer_forecast",
    "forecaster",
    "lstm_forecast",
    "lstm_seq2one",
}


def _build_model(model_type: str, input_dim: int):
    return build_shared_model(
        model_type,
        input_dim,
        config=MODEL_CONFIG,
        allowed_model_types=MODEL_TYPES,
    )


def _build_window_dataset(seqs):
    return build_window_dataset(seqs, window_size=WINDOW_SIZE, stride=STRIDE)


def _build_window_dataset_with_ids(seqs, file_ids):
    return build_window_dataset_with_ids(
        seqs,
        file_ids,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )


def _build_forecast_dataset(seqs):
    return build_forecast_dataset(seqs, window_size=WINDOW_SIZE, stride=STRIDE)


def _build_forecast_dataset_with_ids(seqs, file_ids):
    return build_forecast_dataset_with_ids(
        seqs,
        file_ids,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )


def _train_model(train_seqs, val_seqs, model_type):
    return train_model(
        train_seqs,
        val_seqs,
        model_type=model_type,
        config=TRAINING_CONFIG,
        build_model_fn=_build_model,
        build_window_dataset_fn=_build_window_dataset,
    )


def _train_forecaster(train_seqs, val_seqs, model_type):
    return train_forecaster(
        train_seqs,
        val_seqs,
        model_type=model_type,
        config=TRAINING_CONFIG,
        build_model_fn=_build_model,
        build_forecast_dataset_fn=_build_forecast_dataset,
    )


def run_single_split(split_entry: Dict[str, object], model_type: str) -> Dict[str, object]:
    model_key = str(model_type).lower()

    if model_key in FORECAST_MODELS:
        return run_split_evaluation(
            split_entry=split_entry,
            model_type=model_type,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            val_quantile=VAL_QUANTILE,
            human_controlled_features=HUMAN_CONTROLLED_FEATURES,
            ignored_features=IGNORED_FEATURES,
            prepare_split_fn=prepare_split,
            train_fn=_train_forecaster,
            build_val_dataset_fn=_build_forecast_dataset,
            build_test_dataset_with_ids_fn=_build_forecast_dataset_with_ids,
            score_fn=lambda model, dataset: collect_forecast_scores(
                model, dataset, device=DEVICE, batch_size=BATCH_SIZE
            ),
            feature_mse_fn=lambda model, dataset: collect_forecast_feature_mse(
                model, dataset, device=DEVICE, batch_size=BATCH_SIZE
            ),
            label_prefix=f"[{model_type}] ",
            error_label="forecast",
        )

    return run_split_evaluation(
        split_entry=split_entry,
        model_type=model_type,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        val_quantile=VAL_QUANTILE,
        human_controlled_features=HUMAN_CONTROLLED_FEATURES,
        ignored_features=IGNORED_FEATURES,
        prepare_split_fn=prepare_split,
        train_fn=_train_model,
        build_val_dataset_fn=_build_window_dataset,
        build_test_dataset_with_ids_fn=_build_window_dataset_with_ids,
        score_fn=lambda model, dataset: collect_reconstruction_scores(
            model, dataset, device=DEVICE, batch_size=BATCH_SIZE
        ),
        feature_mse_fn=lambda model, dataset: collect_reconstruction_feature_mse(
            model, dataset, device=DEVICE, batch_size=BATCH_SIZE
        ),
        label_prefix=f"[{model_type}] ",
        error_label="reconstruction",
    )
