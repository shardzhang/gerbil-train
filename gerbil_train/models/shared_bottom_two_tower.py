"""Shared-Bottom Two-Tower model.

This module implements the Shared-Bottom Two-Tower (SBTT) model for
recommendation tasks.

Typical usage:
    - user tower + item tower
    - shared bottom encoder
    - task-specific tower outputs

Paper: 2019-Improving Relevance Prediction with Transfer Learning in Large-scale Retrieval Systems
    https://openreview.net/pdf?id=SJxPVcSonN
"""

from __future__ import annotations

from typing import Any, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["SharedBottomTwoTower"]


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float = 0.0,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return nn.Sequential(*layers)


@dataclass
class SBTTOutput:
    explicit_query_embedding: Tensor
    explicit_item_embedding: Tensor
    explicit_score: Tensor
    implicit_query_embedding: Tensor
    implicit_item_embedding: Tensor
    implicit_score: Tensor


class SharedBottomTwoTower(nn.Module):
    """Shared-Bottom Two-Tower for retrieval/recommendation.

    Each side (query/item) contains:
      - shared bottom
      - explicit sub-tower
      - implicit sub-tower
    """

    def __init__(
        self,
        query_input_dim: int,
        item_input_dim: int,
        shared_hidden_dims: Sequence[int],
        explicit_hidden_dims: Sequence[int],
        implicit_hidden_dims: Sequence[int],
        embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # query side
        self.query_shared_bottom = build_mlp(
            input_dim=query_input_dim,
            hidden_dims=shared_hidden_dims,
            dropout=dropout,
        )
        query_shared_dim = shared_hidden_dims[-1]

        self.query_explicit_tower = build_mlp(
            input_dim=query_shared_dim,
            hidden_dims=explicit_hidden_dims,
            dropout=dropout,
        )
        self.query_implicit_tower = build_mlp(
            input_dim=query_shared_dim,
            hidden_dims=implicit_hidden_dims,
            dropout=dropout,
        )

        self.query_explicit_head = nn.Linear(explicit_hidden_dims[-1], embedding_dim)
        self.query_implicit_head = nn.Linear(implicit_hidden_dims[-1], embedding_dim)

        # item side
        self.item_shared_bottom = build_mlp(
            input_dim=item_input_dim,
            hidden_dims=shared_hidden_dims,
            dropout=dropout,
        )
        item_shared_dim = shared_hidden_dims[-1]

        self.item_explicit_tower = build_mlp(
            input_dim=item_shared_dim,
            hidden_dims=explicit_hidden_dims,
            dropout=dropout,
        )
        self.item_implicit_tower = build_mlp(
            input_dim=item_shared_dim,
            hidden_dims=implicit_hidden_dims,
            dropout=dropout,
        )

        self.item_explicit_head = nn.Linear(explicit_hidden_dims[-1], embedding_dim)
        self.item_implicit_head = nn.Linear(implicit_hidden_dims[-1], embedding_dim)

    def encode_query_shared(self, query_features: Tensor) -> Tensor:
        """Encode query features through the shared bottom.
        :param: query_features: [batch_size, query_input_dim]
        :return: [batch_size, shared_hidden_dims[-1]]
        """
        return self.query_shared_bottom(query_features)

    def encode_item_shared(self, item_features: Tensor) -> Tensor:
        """Encode item features through the shared bottom.
        :param: item_features: [batch_size, item_input_dim]
        :return: [batch_size, shared_hidden_dims[-1]]
        """
        return self.item_shared_bottom(item_features)

    def encode_query_explicit(self, query_features: Tensor, detach_shared: bool = False,) -> Tensor:
        """Encode query features through the explicit sub-tower.
        :param: query_features: [batch_size, query_input_dim]
        :param: detach_shared: whether to detach the shared bottom features
        :return: [batch_size, embedding_dim]
        """
        q_shared = self.encode_query_shared(query_features)
        if detach_shared:
            q_shared = q_shared.detach()
        q_exp = self.query_explicit_tower(q_shared)
        return self.query_explicit_head(q_exp)

    def encode_item_explicit(self, item_features: Tensor, detach_shared: bool = False,) -> Tensor:
        """Encode item features through the explicit sub-tower.
        :param: item_features: [batch_size, item_input_dim]
        :param: detach_shared: whether to detach the shared bottom features
        :return: [batch_size, embedding_dim]
        """
        i_shared = self.encode_item_shared(item_features)
        if detach_shared:
            i_shared = i_shared.detach()
        i_exp = self.item_explicit_tower(i_shared)
        return self.item_explicit_head(i_exp)

    def encode_query_implicit(self, query_features: Tensor) -> Tensor:
        """Encode query features through the implicit sub-tower.
        :param: query_features: [batch_size, query_input_dim]
        :return: [batch_size, embedding_dim]
        """
        q_shared = self.encode_query_shared(query_features)
        q_imp = self.query_implicit_tower(q_shared)
        return self.query_implicit_head(q_imp)

    def encode_item_implicit(self, item_features: Tensor) -> Tensor:
        """Encode item features through the implicit sub-tower.
        :param: item_features: [batch_size, item_input_dim]
        :return: [batch_size, embedding_dim]
        """
        i_shared = self.encode_item_shared(item_features)
        i_imp = self.item_implicit_tower(i_shared)
        return self.item_implicit_head(i_imp)

    @staticmethod
    def dot_score(lhs: Tensor, rhs: Tensor) -> Tensor:
        """Compute dot product score between two sets of embeddings.
        :param: lhs: [batch_size, embedding_dim]
        :param: rhs: [batch_size, embedding_dim]
        :return: [batch_size]
        """
        return torch.sum(lhs * rhs, dim=-1)

    def forward(self, query_features: Tensor, item_features: Tensor, detach_shared_for_explicit: bool = False,) -> SBTTOutput:
        """Forward pass for the shared bottom two-tower model.
        :param: query_features: [batch_size, query_input_dim]
        :param: item_features: [batch_size, item_input_dim]
        :param: detach_shared_for_explicit: whether to detach the shared bottom features for the explicit sub-tower
        :return: SBTTOutput containing explicit and implicit embeddings and scores
        """
        q_exp = self.encode_query_explicit(query_features, detach_shared=detach_shared_for_explicit,)
        i_exp = self.encode_item_explicit(item_features, detach_shared=detach_shared_for_explicit,)
        q_imp = self.encode_query_implicit(query_features)
        i_imp = self.encode_item_implicit(item_features)

        return SBTTOutput(
            explicit_query_embedding=q_exp,
            explicit_item_embedding=i_exp,
            explicit_score=self.dot_score(q_exp, i_exp),
            implicit_query_embedding=q_imp,
            implicit_item_embedding=i_imp,
            implicit_score=self.dot_score(q_imp, i_imp),
        )