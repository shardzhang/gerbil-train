"""Neural network construction helpers."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

__all__ = ["FullyConnectedLayer", "get_activation"]


def get_activation(name: str, num_features: int | None = None, dice_dim: int | None = None) -> nn.Module:
    """Get the activation function by name.

    :param name: Activation name (case-insensitive).
        Supported: ``relu``, ``gelu``, ``silu`` / ``swish``,
        ``leaky_relu``, ``prelu``, ``tanh``, ``dice``.
    :return: An nn.Module representing the activation function
    """
    name = name.lower().replace("-", "_")
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "prelu":
        return nn.PReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "dice":
        from .dice import Dice
        return Dice(num_features, dim=dice_dim)
    raise ValueError(f"Unsupported activation: {name}")


class FullyConnectedLayer(nn.Module):
    """Multi-layer perceptron with optional batch norm, dropout, and sigmoid output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        bias: list[bool],
        batch_norm: bool = True,
        dropout: float = 0.1,
        activation: str = "relu",
        sigmoid: bool = False,
    ):
        super().__init__()
        assert len(hidden_dims) >= 1 and len(bias) >= 1, "hidden_dims and bias must be non-empty lists"
        assert len(bias) == len(hidden_dims), "bias must have the same length as hidden_dims"
        self.sigmoid = sigmoid

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias[i]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        self.fc: nn.Sequential = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            return self.output_layer(self.fc(x))
        return self.fc(x)
