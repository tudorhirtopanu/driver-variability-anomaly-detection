"""Shared training loop for reconstruction-style false-positive-study models."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TrainingConfig:
    """Bundle-specific training hyperparameters for shared training helpers."""

    batch_size: int
    device: torch.device
    epochs: int
    lr: float
    min_delta: float
    patience: int
    vae_beta: float
    usad_alpha: float
    usad_beta: float


def _make_window_loader(
    seqs: List[np.ndarray],
    build_window_dataset_fn: Callable[[List[np.ndarray]], Dataset],
    batch_size: int,
    drop_last: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    ds = build_window_dataset_fn(seqs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _concat_windows(loader: DataLoader) -> torch.Tensor:
    batches = [batch for batch in loader]
    if not batches:
        raise ValueError("No windows available to fit baseline.")
    return torch.cat(batches, dim=0)


def _train_pca_baseline(
    train_seqs: List[np.ndarray],
    val_seqs: List[np.ndarray],
    input_dim: int,
    *,
    config: TrainingConfig,
    build_model_fn: Callable[[str, int], nn.Module],
    build_window_dataset_fn: Callable[[List[np.ndarray]], Dataset],
) -> Tuple[nn.Module, Dict[str, float]]:
    train_loader = _make_window_loader(
        train_seqs,
        build_window_dataset_fn,
        batch_size=config.batch_size,
        drop_last=False,
    )
    train_windows = _concat_windows(train_loader)

    model = build_model_fn("pca", input_dim)
    model.fit(train_windows)
    model = model.to(config.device)

    val_loader = _make_window_loader(
        val_seqs,
        build_window_dataset_fn,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
    )
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(config.device)
            x_hat = model(x)
            mse = ((x_hat - x) ** 2).mean(dim=(1, 2))
            val_losses.extend(mse.detach().cpu().tolist())

    mean_val = float(np.mean(val_losses)) if val_losses else 0.0
    stats = {
        "best_epoch": 0,
        "best_val_loss": mean_val,
        "input_dim": input_dim,
        "n_components": getattr(model, "n_components", None),
    }
    return model, stats


def _train_persistence_baseline(
    train_seqs: List[np.ndarray],
    val_seqs: List[np.ndarray],
    input_dim: int,
    *,
    config: TrainingConfig,
    build_model_fn: Callable[[str, int], nn.Module],
    build_window_dataset_fn: Callable[[List[np.ndarray]], Dataset],
) -> Tuple[nn.Module, Dict[str, float]]:
    del train_seqs
    model = build_model_fn("persistence", input_dim).to(config.device)
    val_loader = _make_window_loader(
        val_seqs,
        build_window_dataset_fn,
        batch_size=config.batch_size,
        drop_last=False,
    )

    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(config.device)
            x_hat = model(x)
            mse = ((x_hat - x) ** 2).mean(dim=(1, 2))
            val_losses.extend(mse.detach().cpu().tolist())

    mean_val = float(np.mean(val_losses)) if val_losses else 0.0
    stats = {
        "best_epoch": 0,
        "best_val_loss": mean_val,
        "input_dim": input_dim,
    }
    return model, stats


def _train_usad(
    train_seqs: List[np.ndarray],
    val_seqs: List[np.ndarray],
    input_dim: int,
    *,
    config: TrainingConfig,
    build_model_fn: Callable[[str, int], nn.Module],
    build_window_dataset_fn: Callable[[List[np.ndarray]], Dataset],
) -> Tuple[nn.Module, Dict[str, float]]:
    model = build_model_fn("usad", input_dim).to(config.device)
    criterion = nn.MSELoss()

    opt1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        lr=config.lr,
    )
    opt2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        lr=config.lr,
    )

    scheduler1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=5, gamma=0.5)
    scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=5, gamma=0.5)

    train_loader = _make_window_loader(
        train_seqs,
        build_window_dataset_fn,
        batch_size=config.batch_size,
    )
    val_loader = _make_window_loader(
        val_seqs,
        build_window_dataset_fn,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    warmup_epochs = 5
    grad_clip = 0.5

    for epoch in range(config.epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch.to(config.device)

            opt1.zero_grad()
            _, w1, _, w3 = model(x, return_all=True)
            loss1 = config.usad_alpha * criterion(w1, x) + (1 - config.usad_alpha) * criterion(w3, x)
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt1.step()

            opt2.zero_grad()
            _, _, w2, w3 = model(x, return_all=True)
            if epoch < warmup_epochs:
                loss2 = criterion(w2, x) + criterion(w3, x)
            else:
                loss2 = config.usad_beta * criterion(w2, x) - (1 - config.usad_beta) * criterion(w3, x)

            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt2.step()

            train_losses.append(loss1.item() + loss2.item())

        scheduler1.step()
        scheduler2.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(config.device)
                x_hat = model(x)
                loss = criterion(x_hat, x)
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
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print()
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_loss": best_val_loss}


def train_model(
    train_seqs: List[np.ndarray],
    val_seqs: List[np.ndarray],
    model_type: str,
    *,
    config: TrainingConfig,
    build_model_fn: Callable[[str, int], nn.Module],
    build_window_dataset_fn: Callable[[List[np.ndarray]], Dataset],
) -> Tuple[nn.Module, Dict[str, float]]:
    assert len(train_seqs) > 0
    model_type = model_type.lower()
    input_dim = train_seqs[0].shape[1]

    if model_type == "pca":
        return _train_pca_baseline(
            train_seqs,
            val_seqs,
            input_dim,
            config=config,
            build_model_fn=build_model_fn,
            build_window_dataset_fn=build_window_dataset_fn,
        )
    if model_type in ("persistence", "naive"):
        return _train_persistence_baseline(
            train_seqs,
            val_seqs,
            input_dim,
            config=config,
            build_model_fn=build_model_fn,
            build_window_dataset_fn=build_window_dataset_fn,
        )
    if model_type == "usad":
        return _train_usad(
            train_seqs,
            val_seqs,
            input_dim,
            config=config,
            build_model_fn=build_model_fn,
            build_window_dataset_fn=build_window_dataset_fn,
        )

    model = build_model_fn(model_type, input_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    is_vae = model_type == "lstm_vae"

    train_loader = _make_window_loader(
        train_seqs,
        build_window_dataset_fn,
        batch_size=config.batch_size,
    )
    val_loader = _make_window_loader(
        val_seqs,
        build_window_dataset_fn,
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
            x = batch.to(config.device)
            optimizer.zero_grad()
            if is_vae:
                x_hat, mu, logvar = model(x, return_latent=True)
                recon_loss = criterion(x_hat, x)
                kld = model.kl_divergence(mu, logvar)
                loss = recon_loss + config.vae_beta * kld
            else:
                x_hat = model(x)
                loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(config.device)
                if is_vae:
                    x_hat, mu, logvar = model(x, return_latent=True)
                    recon_loss = criterion(x_hat, x)
                    kld = model.kl_divergence(mu, logvar)
                    loss = recon_loss + config.vae_beta * kld
                else:
                    x_hat = model(x)
                    loss = criterion(x_hat, x)
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
