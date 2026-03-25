"""Shared training loop for forecasting false-positive-study models."""

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fp_analysis.shared.train_reconstruction import TrainingConfig


def _make_forecast_loader(
    seqs: List[np.ndarray],
    build_forecast_dataset_fn: Callable[[List[np.ndarray]], Dataset],
    batch_size: int,
    drop_last: bool = True,
) -> DataLoader:
    ds = build_forecast_dataset_fn(seqs)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)


def train_forecaster(
    train_seqs: List[np.ndarray],
    val_seqs: List[np.ndarray],
    model_type: str,
    *,
    config: TrainingConfig,
    build_model_fn: Callable[[str, int], nn.Module],
    build_forecast_dataset_fn: Callable[[List[np.ndarray]], Dataset],
) -> Tuple[nn.Module, Dict[str, float]]:
    assert len(train_seqs) > 0
    input_dim = train_seqs[0].shape[1]

    model = build_model_fn(model_type, input_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    train_loader = _make_forecast_loader(
        train_seqs,
        build_forecast_dataset_fn,
        batch_size=config.batch_size,
        drop_last=True,
    )
    val_loader = _make_forecast_loader(
        val_seqs,
        build_forecast_dataset_fn,
        batch_size=config.batch_size,
        drop_last=False,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            x, y = batch
            x = x.to(config.device)
            y = y.to(config.device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(config.device)
                y = y.to(config.device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_losses.append(loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0

        print(
            f"\rEpoch {epoch+1:03d}/{config.epochs} "
            f"train={mean_train:.4f} val={mean_val:.4f}",
            end="",
            flush=True,
        )

        if mean_val + config.min_delta < best_val_loss:
            best_val_loss = mean_val
            best_state = model.state_dict()
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(
                    f"\nEarly stopping at epoch {epoch+1} "
                    f"(no improvement for {config.patience} epochs)."
                )
                break

    print()

    if best_state is not None:
        model.load_state_dict(best_state)

    stats = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "input_dim": input_dim,
    }
    return model, stats
