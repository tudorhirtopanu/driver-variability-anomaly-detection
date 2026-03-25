"""Per-feature uncertainty gate.

The gate combines one or more signals into

    alpha = sigmoid(bias + sum_i scale[i] * signal[i])

with alpha in [0, 1] for each feature.
"""

from typing import Sequence, Union

import torch
import torch.nn as nn


Signals = Union[torch.Tensor, Sequence[torch.Tensor]]


class UncertaintyGate(nn.Module):
    """
    Learns a per-feature gate from one or more signals:
        alpha = sigmoid(sum_i scale_i * signal_i + bias)
    """

    def __init__(
        self,
        feature_dim: int,
        num_signals: int = 1,
        init_scale: float = 0.5,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()

        if num_signals < 1:
            raise ValueError("num_signals must be >= 1")

        self.num_signals = int(num_signals)
        feature_dim = int(feature_dim)

        # Learnable per-signal scaling (num_signals, D)
        self.scale = nn.Parameter(
            torch.ones(self.num_signals, feature_dim) * init_scale
        )

        # Per-feature bias (D,)
        self.bias = nn.Parameter(
            torch.ones(feature_dim) * init_bias
        )

    def set_bias_from_prior(self, alpha_prior: torch.Tensor, eps: float = 1e-5) -> None:
        """
        Initialize bias so that sigmoid(bias) approximately matches a desired prior alpha.

        Args:
            alpha_prior: (D,) tensor of prior gate probabilities
            eps: small constant for numerical stability
        """
        alpha = alpha_prior.clamp(eps, 1.0 - eps)
        bias = torch.log(alpha / (1.0 - alpha))

        with torch.no_grad():
            self.bias.copy_(bias)

    def forward(self, signals: Signals) -> torch.Tensor:
        """
        Args:
            signals:
                (B, D) tensor or list/tuple of tensors (B, D)
                Length must equal num_signals.

        Returns:
            alpha: (B, D) in [0, 1]
        """
        if isinstance(signals, (list, tuple)):
            signal_list = list(signals)
        else:
            signal_list = [signals]

        if len(signal_list) != self.num_signals:
            raise ValueError(
                f"Expected {self.num_signals} signals, got {len(signal_list)}"
            )

        gating_signal = self.bias  # (D,)

        for i, s in enumerate(signal_list):
            gating_signal = gating_signal + s * self.scale[i]

        gating_signal = gating_signal.clamp(-15.0, 15.0)

        return torch.sigmoid(gating_signal)
