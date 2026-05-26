"""Learning to Rank (LTR)

input: MSLR-WEB10K, 136-dimensional features
output: relevance score for each document
evaluation: NDCG@k
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["DeepRankNet"]


class DeepRankNet(nn.Module):
    """Feed-forward ranking model that predicts one relevance score per document."""

    def __init__(self, input_dim=136, hidden_dims=[256, 128, 64]):
        """Initialize the ranking network.

        :param input_dim: Number of input features per document
        :param hidden_dims: Hidden layer sizes for the ranking MLP
        """
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Run a forward pass and return one score per document.

        :param x: Feature tensor of shape ``[num_docs, input_dim]``
        :return: Score tensor of shape ``[num_docs]``
        """
        return self.model(x).squeeze(-1)
