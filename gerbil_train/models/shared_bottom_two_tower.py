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
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SBTTOutput", "SharedBottomTwoTower"]


def get_activation(name: str) -> nn.Module:
    """Get the activation function by name.
    :param name: Name of the activation function ("relu", "gelu", "tanh")
    :return: An nn.Module representing the activation function
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float = 0.0,
    activation: str = "relu",
    batch_norm: bool = False,
) -> nn.Sequential:
    """Build a multi-layer perceptron (MLP) with optional dropout, batch normalization, and activation.
    :param input_dim: Dimension of the input features
    :param hidden_dims: List of hidden layer dimensions
    :param dropout: Dropout rate to apply after each hidden layer (default: 0.0)
    :param activation: Activation function to use (default: "relu")
    :param batch_norm: Whether to apply batch normalization after each hidden layer (default: False)
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
    
    Query side:
      shared bottom -> explicit tower / implicit tower
    
    Item side:
      shared bottom -> explicit tower / implicit tower
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
        activation: str = "relu",
        batch_norm: bool = False,
        normalize_embedding: bool = False,
        temperature: float = 0.07,
    ) -> None:
        """Initialize the Shared-Bottom Two-Tower model.
        :param query_input_dim: Dimensionality of query input features
        :param item_input_dim: Dimensionality of item input features
        :param shared_hidden_dims: List of hidden layer dimensions for the shared bottom
        :param explicit_hidden_dims: List of hidden layer dimensions for the explicit tower
        :param implicit_hidden_dims: List of hidden layer dimensions for the implicit tower
        :param embedding_dim: Dimensionality of the output embeddings
        :param dropout: Dropout rate to apply after each hidden layer (default: 0.0)
        :param activation: Activation function to use (default: "relu")
        :param batch_norm: Whether to apply batch normalization after each hidden layer (default: False)
        :param normalize_embedding: Whether to L2 normalize the output embeddings (default: False)
        :param temperature: Temperature for scaling the implicit scores (default: 0.07)
        """
        super().__init__()

        if len(shared_hidden_dims) == 0:
            raise ValueError("shared_hidden_dims must not be empty")
        if len(explicit_hidden_dims) == 0:
            raise ValueError("explicit_hidden_dims must not be empty")
        if len(implicit_hidden_dims) == 0:
            raise ValueError("implicit_hidden_dims must not be empty")

        self.normalize_embedding = normalize_embedding
        self.temperature = temperature

        # query side
        self.query_shared_bottom = build_mlp(
            input_dim=query_input_dim,
            hidden_dims=shared_hidden_dims,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )
        query_shared_dim = shared_hidden_dims[-1]

        self.query_explicit_tower = build_mlp(
            input_dim=query_shared_dim,
            hidden_dims=explicit_hidden_dims,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.query_implicit_tower = build_mlp(
            input_dim=query_shared_dim,
            hidden_dims=implicit_hidden_dims,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )

        self.query_explicit_head = nn.Linear(explicit_hidden_dims[-1], embedding_dim)
        self.query_implicit_head = nn.Linear(implicit_hidden_dims[-1], embedding_dim)

        # item side
        self.item_shared_bottom = build_mlp(
            input_dim=item_input_dim,
            hidden_dims=shared_hidden_dims,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )
        item_shared_dim = shared_hidden_dims[-1]

        self.item_explicit_tower = build_mlp(
            input_dim=item_shared_dim,
            hidden_dims=explicit_hidden_dims,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.item_implicit_tower = build_mlp(
            input_dim=item_shared_dim,
            hidden_dims=implicit_hidden_dims,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
        )

        self.item_explicit_head = nn.Linear(explicit_hidden_dims[-1], embedding_dim)
        self.item_implicit_head = nn.Linear(implicit_hidden_dims[-1], embedding_dim)

    def _maybe_normalize(self, x: Tensor) -> Tensor:
        """Optionally normalize the input tensor.
        推荐系统双塔模型里非常标准的Embedding归一化函数. 开启后相似度计算从 “内积” 变成标准 “余弦相似度”

        :param: x: [batch_size, embedding_dim]
        :return: [batch_size, embedding_dim]
        """
        if self.normalize_embedding:
            return F.normalize(x, p=2, dim=-1)
        return x

    def dot_score(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        """Compute dot product score between two sets of embeddings.
        :param: lhs: [batch_size, embedding_dim]
        :param: rhs: [batch_size, embedding_dim]
        :return: [batch_size]
        """
        return torch.sum(lhs * rhs, dim=-1) / self.temperature

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

    def encode_query_explicit(self, query_features: Tensor, detach_shared: bool = False, ) -> Tensor:
        """Encode query features through the explicit sub-tower.
        :param: query_features: [batch_size, query_input_dim]
        :param: detach_shared: whether to detach the shared bottom features
        :return: [batch_size, embedding_dim]
        """
        q_shared = self.encode_query_shared(query_features)
        if detach_shared:
            q_shared = q_shared.detach()
        q_exp = self.query_explicit_tower(q_shared)
        q_exp = self.query_explicit_head(q_exp)
        return self._maybe_normalize(q_exp)

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
        i_exp = self.item_explicit_head(i_exp)
        return self._maybe_normalize(i_exp)

    def encode_query_implicit(self, query_features: Tensor) -> Tensor:
        """Encode query features through the implicit sub-tower.
        :param: query_features: [batch_size, query_input_dim]
        :return: [batch_size, embedding_dim]
        """
        q_shared = self.encode_query_shared(query_features)
        q_imp = self.query_implicit_tower(q_shared)
        q_imp = self.query_implicit_head(q_imp)
        return self._maybe_normalize(q_imp)

    def encode_item_implicit(self, item_features: Tensor) -> Tensor:
        """Encode item features through the implicit sub-tower.
        :param: item_features: [batch_size, item_input_dim]
        :return: [batch_size, embedding_dim]
        """
        i_shared = self.encode_item_shared(item_features)
        i_imp = self.item_implicit_tower(i_shared)
        i_imp = self.item_implicit_head(i_imp)
        return self._maybe_normalize(i_imp)

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