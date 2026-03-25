"""Shared baseline model definitions for the false-positive analysis bundles."""

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


@dataclass(frozen=True)
class ModelConfig:
    """Bundle-specific hyperparameters for shared false-positive-study models."""

    hidden_dim: int
    latent_dim: int
    num_layers: int
    window_size: int
    pca_components: int
    tcn_channels: tuple[int, ...]
    tcn_kernel_size: int
    tcn_dropout: float
    d_model: int
    nhead: int
    dim_feedforward: int
    num_transformer_layers: int
    dropout: float
    usad_hidden_dim: int
    usad_latent_dim: int


class LSTMAutoencoder(nn.Module):
    """
    Simple LSTM autoencoder:
      Input:  (B, T, D)
      Output: (B, T, D)
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.latent_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, _ = self.encoder(x)
        h_last = enc_out[:, -1, :]

        z = self.latent_fc(h_last)
        z_dec = self.decoder_fc(z)

        time_steps = x.shape[1]
        dec_input = z_dec.unsqueeze(1).repeat(1, time_steps, 1)

        dec_out, _ = self.decoder(dec_input)
        x_hat = self.output_layer(dec_out)
        return x_hat.to(x.dtype).to(x.device)


class LSTMSeq2One(nn.Module):
    """
    LSTM forecaster: consumes a window (B, T, D) and predicts the next step (B, D).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        y_hat = self.output_layer(h_last)
        return y_hat.to(x.dtype).to(x.device)


class LSTMVAE(nn.Module):
    """
    LSTM-based variational autoencoder for sequence reconstruction.
    Returns reconstructions with an optional KL term for training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc_out, _ = self.encoder(x)
        h_last = enc_out[:, -1, :]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, time_steps: int) -> torch.Tensor:
        z_dec = self.decoder_fc(z)
        dec_input = z_dec.unsqueeze(1).repeat(1, time_steps, 1)
        dec_out, _ = self.decoder(dec_input)
        x_hat = self.output_layer(dec_out)
        return x_hat

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ):
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        x_hat = self.decode(z, time_steps=x.shape[1])
        if return_latent:
            return x_hat.to(x.dtype), mu, logvar
        return x_hat.to(x.dtype)


class NaivePersistence(nn.Module):
    """
    Non-learned baseline: predicts the last timestep value for every position.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.input_dim:
            raise ValueError(f"Expected input shape (B, T, {self.input_dim}), got {tuple(x.shape)}")
        last = x[:, -1:, :]
        return last.repeat(1, x.shape[1], 1).to(x.dtype)


class PCABaseline(nn.Module):
    """
    Non-learned PCA reconstruction baseline.

    Fit computes principal components on windowed training data.
    Forward projects + reconstructs flattened windows to return (B, T, D).
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        n_components: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.n_components = n_components

        self.register_buffer("mean_", torch.empty(0))
        self.register_buffer("components_", torch.empty(0, 0))
        self.fitted = False

    @torch.no_grad()
    def fit(self, windows: torch.Tensor) -> None:
        if windows.ndim != 3:
            raise ValueError(f"Expected (N, T, D) windows, got shape={tuple(windows.shape)}")
        if windows.shape[1] != self.window_size or windows.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected window_size={self.window_size}, input_dim={self.input_dim}, "
                f"got {tuple(windows.shape)}"
            )

        flat = windows.reshape(windows.shape[0], -1)
        mean = flat.mean(dim=0)
        flat_centered = flat - mean

        q = min(self.n_components, flat_centered.shape[0], flat_centered.shape[1])
        if q < 1:
            raise ValueError("Need at least one window to fit PCA baseline.")

        _, _, v = torch.pca_lowrank(flat_centered, q=q)
        comps = v[:, :q].T.contiguous()

        self.mean_ = mean
        self.components_ = comps
        self.n_components = q
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("PCABaseline must be fitted before calling forward().")

        batch_size, time_steps, input_dim = x.shape
        if time_steps != self.window_size or input_dim != self.input_dim:
            raise ValueError(
                f"Expected input shape (_, {self.window_size}, {self.input_dim}), got {tuple(x.shape)}"
            )

        flat = x.reshape(batch_size, -1)
        flat_centered = flat - self.mean_.to(dtype=flat.dtype, device=flat.device)
        comps = self.components_.to(dtype=flat.dtype, device=flat.device)

        z = torch.matmul(flat_centered, comps.t())
        recon_flat = torch.matmul(z, comps) + self.mean_
        return recon_flat.reshape(batch_size, time_steps, input_dim).to(x.dtype)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNAutoencoder(nn.Module):
    """
    Temporal Convolutional Network for sequence reconstruction.

    Input:  (B, T, D)
    Output: (B, T, D)
    """

    def __init__(
        self,
        input_dim: int,
        channels: Sequence[int],
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.channels = list(channels)

        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=self.channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.output_proj = nn.Conv1d(
            in_channels=self.channels[-1],
            out_channels=input_dim,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.transpose(1, 2)
        h = self.tcn(x_perm)
        out = self.output_proj(h)
        return out.transpose(1, 2).to(x.dtype)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = x.shape
        return x + self.pe[:, :time_steps, :]


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based autoencoder for sequence reconstruction.

    Input:  (B, T, D_in)
    Output: (B, T, D_in)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(self.d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        x_proj = self.pos_encoder(x_proj)
        h = self.encoder(x_proj)
        x_hat = self.output_proj(h)
        return x_hat.to(x.dtype).to(x.device)


class TransformerForecaster(nn.Module):
    """
    Transformer-based one-step forecaster.

    Input:  (B, T, D_in)  - T past steps
    Output: (B, D_out)    - next-step prediction (here D_out = D_in)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim if output_dim is not None else input_dim

        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(self.d_model, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        x_proj = self.pos_encoder(x_proj)
        h = self.encoder(x_proj)
        h_last = h[:, -1, :]
        y_hat = self.output_proj(h_last)
        return y_hat.to(x.dtype).to(x.device)


class USAD(nn.Module):
    """
    USAD (KDD'20): two decoders with shared encoder.

    Forward (inference) returns a blended reconstruction x_hat = 0.5*(w1 + w3).
    Training requires intermediate outputs w1, w2, w3 via return_all=True.
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        hidden_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.flat_dim = input_dim * window_size

        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flat_dim),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flat_dim),
        )

    def _reshape_back(self, flat: torch.Tensor, time_steps: int) -> torch.Tensor:
        return flat.view(flat.shape[0], time_steps, self.input_dim)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        batch_size, time_steps, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")
        if time_steps != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got {time_steps}")

        flat = x.reshape(batch_size, -1)
        z = self.encoder(flat)
        w1_flat = self.decoder1(z)
        w2_flat = self.decoder2(z)

        z2 = self.encoder(w1_flat)
        w3_flat = self.decoder2(z2)

        w1 = self._reshape_back(w1_flat, time_steps)
        w2 = self._reshape_back(w2_flat, time_steps)
        w3 = self._reshape_back(w3_flat, time_steps)

        x_hat = 0.5 * (w1 + w3)
        if return_all:
            return x_hat.to(x.dtype), w1.to(x.dtype), w2.to(x.dtype), w3.to(x.dtype)
        return x_hat.to(x.dtype)


def build_model(model_type: str, input_dim: int, config: ModelConfig) -> nn.Module:
    """Instantiate a false-positive-study baseline from explicit bundle config."""

    alias_map = {
        "forecaster": "transformer_forecast",
        "lstm_seq2one": "lstm_forecast",
    }
    model_key = alias_map.get(model_type.lower(), model_type.lower())

    if model_key == "lstm":
        return LSTMAutoencoder(
            input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
        )
    if model_key == "transformer":
        return TransformerAutoencoder(
            input_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_layers=config.num_transformer_layers,
        )
    if model_key == "transformer_forecast":
        return TransformerForecaster(
            input_dim,
            output_dim=None,
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_layers=config.num_transformer_layers,
        )
    if model_key == "lstm_forecast":
        return LSTMSeq2One(
            input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
    if model_key == "lstm_vae":
        return LSTMVAE(
            input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
        )
    if model_key == "pca":
        return PCABaseline(
            input_dim,
            window_size=config.window_size,
            n_components=config.pca_components,
        )
    if model_key == "tcn":
        return TCNAutoencoder(
            input_dim,
            channels=config.tcn_channels,
            kernel_size=config.tcn_kernel_size,
            dropout=config.tcn_dropout,
        )
    if model_key in ("persistence", "naive"):
        return NaivePersistence(input_dim)
    if model_key == "usad":
        return USAD(
            input_dim,
            window_size=config.window_size,
            hidden_dim=config.usad_hidden_dim,
            latent_dim=config.usad_latent_dim,
        )
    raise ValueError(f"Unknown model_type={model_type}")
