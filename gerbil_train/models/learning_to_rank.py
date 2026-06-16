"""Learning to Rank (LTR)

input: MSLR-WEB10K, 136-dimensional features
output: relevance score for each document
evaluation: NDCG@k

Paper:
"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from gerbil_train.config import LTRConfig
from gerbil_train.utils.nn import get_activation

__all__ = ["DeepRankNet"]


class DeepRankNet(nn.Module):
    """Feed-forward ranking model that predicts one relevance score per document."""

    def __init__(self, config: LTRConfig) -> None:
        """Initialize the ranking network.

        :param config: LTR model configuration
        """
        super().__init__()
        input_dim = int(config.input_dim)
        hidden_dims = list(config.hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        activation_name = str(config.activation)
        dropout = float(config.dropout)

        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(self._get_activation(activation_name))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Run a forward pass and return one score per document.

        :param x: Feature tensor of shape ``[num_docs, input_dim]``
        :return: Score tensor of shape ``[num_docs]``
        """
        return self.model(x).squeeze(-1)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Return the configured activation module."""
        normalized_name = name.lower()
        if normalized_name == "relu":
            return nn.ReLU()
        if normalized_name == "gelu":
            return nn.GELU()
        if normalized_name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation: {name}")
