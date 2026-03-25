"""Gaussian LSTM forecaster.

Shape conventions:
    x: (B, T, input_size)
    mu: (B, D)
    sigma: (B, D)
    log_sigma: (B, D)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianLSTMForecaster(nn.Module):
    """LSTM forecaster that predicts a diagonal Gaussian per target dimension."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
        min_scale: float = 1e-6,
        init_scale_bias: float = -2.0,
    ) -> None:
        super().__init__()

        # PyTorch applies LSTM dropout only when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        # Separate heads for mean and (raw) scale
        self.mu_head = nn.Linear(hidden_size, output_dim)
        self.scale_head = nn.Linear(hidden_size, output_dim)

        self.min_scale = float(min_scale)
        nn.init.constant_(self.scale_head.bias, float(init_scale_bias))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu:        (B, D)
            sigma:     (B, D) positive
            log_sigma: (B, D)
        """
        out, _ = self.lstm(x)     # (B, T, H)
        last = out[:, -1, :]      # (B, H)

        mu = self.mu_head(last)   # (B, D)
        raw = self.scale_head(last)
        sigma = F.softplus(raw) + self.min_scale
        log_sigma = torch.log(sigma)

        return mu, sigma, log_sigma
