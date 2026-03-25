"""Marginal density modules.

Shape conventions:
    x: (N, D)
    log_prob(x) -> (N, D)
    surprise(x) -> (N, D)

Marginals are factorized across dimensions.
"""

import math
from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform


def make_1d_rqs_flow(
    num_bins: int = 8,
    tail_bound: float = 10.0,
    hidden_features: int = 64,
) -> Flow:
    """
    Build a 1D rational-quadratic spline flow.

    In 1D, an autoregressive spline acts as a learned monotonic spline transform.
    """
    transform = CompositeTransform(
        [
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=1,
                hidden_features=hidden_features,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
            )
        ]
    )
    base = StandardNormal([1])
    return Flow(transform, base)


class MarginalFlowBank(nn.Module):
    """
    Bank of independent 1D flows: one flow per dimension.

    log_prob returns per-dimension log-probabilities with shape (N, D).
    """

    def __init__(
        self,
        D: int,
        hidden_features: int = 64,
        num_bins: int = 8,
        tail_bound: float = 10.0,
    ) -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        self.flows = nn.ModuleList(
            [
                make_1d_rqs_flow(
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                    hidden_features=hidden_features,
                )
                for _ in range(D)
            ]
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, D)

        Returns:
            (N, D) per-dimension log-probabilities
        """
        outs = []
        for i, flow in enumerate(self.flows):
            xi = x[:, i : i + 1]            # (N, 1)
            outs.append(flow.log_prob(xi))  # (N,)
        return torch.stack(outs, dim=1)     # (N, D)

    def surprise(self, x: torch.Tensor) -> torch.Tensor:
        """Returns negative log-probabilities per dimension: (N, D)."""
        return -self.log_prob(x)


class DiagonalGaussianMarginal(nn.Module):
    """
    Diagonal Gaussian marginal with learnable mean and log-std per dimension.

    log_prob returns per-dimension log-probabilities with shape (N, D).
    """

    def __init__(self, D: int) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(D))
        self.log_std = nn.Parameter(torch.zeros(D))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, D)

        Returns:
            (N, D) per-dimension log-probabilities
        """
        log_std = self.log_std
        std = torch.exp(log_std)
        z = (x - self.mean) / std
        return -0.5 * (z**2 + 2.0 * log_std + math.log(2.0 * math.pi))

    def surprise(self, x: torch.Tensor) -> torch.Tensor:
        """Returns negative log-probabilities per dimension: (N, D)."""
        return -self.log_prob(x)


def make_marginal_expert(
    D: int,
    marginal_type: str = "flow",
    hidden_features: int = 64,
    num_bins: int = 8,
    tail_bound: float = 10.0,
) -> nn.Module:
    """
    Factory for marginal experts.

    Returns:
        - DiagonalGaussianMarginal if marginal_type == "gaussian"
        - MarginalFlowBank if marginal_type == "flow"
    """
    if marginal_type == "gaussian":
        return DiagonalGaussianMarginal(D)
    if marginal_type != "flow":
        raise ValueError(f"Unknown marginal_type: {marginal_type}")
    return MarginalFlowBank(
        D,
        hidden_features=hidden_features,
        num_bins=num_bins,
        tail_bound=tail_bound,
    )


def infer_marginal_type_from_ckpt(ckpt: Any) -> Tuple[str, Mapping[str, Any]]:
    """
    Infer marginal type from a checkpoint.

    Supports:
      - checkpoints with ckpt["meta"]["model"]["marginal_type"]
      - a fallback heuristic based on state_dict keys
    """
    if isinstance(ckpt, Mapping):
        meta = ckpt.get("meta")
        if isinstance(meta, Mapping):
            model_meta = meta.get("model", {})
            if isinstance(model_meta, Mapping):
                marginal_type = model_meta.get("marginal_type", "flow")
                return marginal_type, model_meta

        state_dict = ckpt.get("state_dict")
        if isinstance(state_dict, Mapping):
            if any(k.startswith("marginal_expert.mean") for k in state_dict.keys()):
                return "gaussian", {}

    return "flow", {}
