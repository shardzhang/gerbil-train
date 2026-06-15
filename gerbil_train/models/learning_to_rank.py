"""Learning to Rank (LTR)

input: MSLR-WEB10K, 136-dimensional features
output: relevance score for each document
evaluation: NDCG@k

Paper:
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch.nn as nn

__all__ = ["DeepRankNet"]


class DeepRankNet(nn.Module):
    """Feed-forward ranking model that predicts one relevance score per document."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize the ranking network.

        :param config: Model configuration mapping
        """
        super().__init__()
        input_dim = int(config.get("input_dim", 136))
        hidden_dims = self._get_hidden_dims(config)
        activation_name = str(config.get("activation", "relu"))
        dropout = float(config.get("dropout", 0.1))

        layers = []
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
    def _get_hidden_dims(config: Mapping[str, Any]) -> Sequence[int]:
        """Return validated hidden dimensions from the model config."""
        hidden_dims = config.get("hidden_dims", [256, 128, 64])
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        return hidden_dims

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
