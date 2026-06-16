"""Neural network construction helpers."""

from __future__ import annotations

from typing import Sequence

from torch import nn

__all__ = ["build_mlp", "get_activation"]


def get_activation(name: str) -> nn.Module:
    """Get the activation function by name.

    :param name: Activation name (case-insensitive).
        Supported: ``relu``, ``gelu``, ``silu`` / ``swish``,
        ``leaky_relu``, ``prelu``, ``tanh``.
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
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    batch_norm: bool = False,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a multi-layer perceptron (MLP) with optional dropout, batch normalization, and activation.

    :param input_dim: Dimension of the input features
    :param hidden_dims: List of hidden layer dimensions
    :param batch_norm: Whether to apply batch normalization after each hidden layer (default: False)
    :param activation: Activation function to use (default: "relu")
    :param dropout: Dropout rate to apply after each hidden layer (default: 0.0)
    :return: An nn.Sequential model representing the MLP
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return nn.Sequential(*layers)
