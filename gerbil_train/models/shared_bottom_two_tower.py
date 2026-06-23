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

from typing import Any, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from gerbil_train.config.train_config import SBTTConfig
from gerbil_train.utils.nn import FullyConnectedLayer

__all__ = ["SBTTOutput", "SharedBottomTwoTower"]


@dataclass
class SBTTOutput:
    """Structured outputs returned by ``SharedBottomTwoTower.forward``."""

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

    def __init__(self, config: SBTTConfig) -> None:
        """Initialize the Shared-Bottom Two-Tower model.

        :param config: SBTT model configuration
        """
        super().__init__()

        query_input_dim = int(config.query_input_dim)
        item_input_dim = int(config.item_input_dim)
        shared_bottom = config.shared_bottom
        explicit_tower = config.explicit_tower
        implicit_tower = config.implicit_tower
        embedding_dim = int(config.embedding_dim)
        normalize_embedding = bool(config.normalize_embedding)
        temperature = float(config.temperature)

        shared_hidden_dims = self._get_hidden_dims(shared_bottom, "shared_bottom")
        explicit_hidden_dims = self._get_hidden_dims(explicit_tower, "explicit_tower")
        implicit_hidden_dims = self._get_hidden_dims(implicit_tower, "implicit_tower")

        shared_activation = str(shared_bottom.get("activation", "relu"))
        shared_dropout = float(shared_bottom.get("dropout", 0.0))
        shared_batch_norm = bool(shared_bottom.get("batch_norm", False))

        explicit_activation = str(explicit_tower.get("activation", shared_activation))
        explicit_dropout = float(explicit_tower.get("dropout", shared_dropout))
        explicit_batch_norm = bool(explicit_tower.get("batch_norm", shared_batch_norm))

        implicit_activation = str(implicit_tower.get("activation", shared_activation))
        implicit_dropout = float(implicit_tower.get("dropout", shared_dropout))
        implicit_batch_norm = bool(implicit_tower.get("batch_norm", shared_batch_norm))

        self.normalize_embedding = normalize_embedding
        self.temperature = temperature

        # query side
        self.query_shared_bottom = FullyConnectedLayer(
            input_dim=query_input_dim, hidden_dims=shared_hidden_dims,
            bias=[True] * len(shared_hidden_dims),
            batch_norm=shared_batch_norm, activation=shared_activation, dropout=shared_dropout,
        )
        query_shared_dim = shared_hidden_dims[-1]

        self.query_explicit_tower = FullyConnectedLayer(
            input_dim=query_shared_dim, hidden_dims=explicit_hidden_dims,
            bias=[True] * len(explicit_hidden_dims),
            batch_norm=explicit_batch_norm, activation=explicit_activation, dropout=explicit_dropout,
        )
        self.query_implicit_tower = FullyConnectedLayer(
            input_dim=query_shared_dim, hidden_dims=implicit_hidden_dims,
            bias=[True] * len(implicit_hidden_dims),
            batch_norm=implicit_batch_norm, activation=implicit_activation, dropout=implicit_dropout,
        )
        self.query_explicit_head = nn.Linear(explicit_hidden_dims[-1], embedding_dim)
        self.query_implicit_head = nn.Linear(implicit_hidden_dims[-1], embedding_dim)

        # item side
        self.item_shared_bottom = FullyConnectedLayer(
            input_dim=item_input_dim, hidden_dims=shared_hidden_dims,
            bias=[True] * len(shared_hidden_dims),
            batch_norm=shared_batch_norm, activation=shared_activation, dropout=shared_dropout,
        )
        item_shared_dim = shared_hidden_dims[-1]

        self.item_explicit_tower = FullyConnectedLayer(
            input_dim=item_shared_dim, hidden_dims=explicit_hidden_dims,
            bias=[True] * len(explicit_hidden_dims),
            batch_norm=explicit_batch_norm, activation=explicit_activation, dropout=explicit_dropout,
        )
        self.item_implicit_tower = FullyConnectedLayer(
            input_dim=item_shared_dim, hidden_dims=implicit_hidden_dims,
            bias=[True] * len(implicit_hidden_dims),
            batch_norm=implicit_batch_norm, activation=implicit_activation, dropout=implicit_dropout,
        )
        self.item_explicit_head = nn.Linear(explicit_hidden_dims[-1], embedding_dim)
        self.item_implicit_head = nn.Linear(implicit_hidden_dims[-1], embedding_dim)

    @staticmethod
    def _get_tower_config(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        """Return one validated tower config block.

        :param config: Model configuration mapping
        :param name: Tower config name
        :return: Tower configuration mapping
        """
        tower_config = config.get(name)
        if not isinstance(tower_config, Mapping):
            raise ValueError(f"{name} must be a mapping")
        return tower_config

    @staticmethod
    def _get_hidden_dims(config: Mapping[str, Any], name: str) -> Sequence[int]:
        """Return validated hidden dimensions from one tower config block.

        :param config: Tower configuration mapping
        :param name: Human-readable config block name for error messages
        :return: Hidden layer dimensions
        """
        hidden_dims = config.get("hidden_dims")
        if not hidden_dims:
            raise ValueError(f"{name}.hidden_dims must not be empty")
        return hidden_dims

    def _maybe_normalize(self, x: Tensor) -> Tensor:
        """Optionally normalize the input tensor.

        In two-tower retrieval models, enabling embedding normalization turns
        the similarity computation from a raw dot product into cosine-style
        similarity.

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

    def encode_query_explicit(
        self,
        query_features: Tensor,
        detach_shared: bool = False,
    ) -> Tensor:
        """Encode query features through the explicit sub-tower.

        :param: query_features: [batch_size, query_input_dim]
        :param: detach_shared: whether to detach the shared bottom features
        :return: [batch_size, embedding_dim]
        """
        q_shared = self.query_shared_bottom(query_features)
        if detach_shared:
            q_shared = q_shared.detach()
        q_exp = self.query_explicit_head(self.query_explicit_tower(q_shared))
        return self._maybe_normalize(q_exp)

    def encode_item_explicit(
        self,
        item_features: Tensor,
        detach_shared: bool = False,
    ) -> Tensor:
        """Encode item features through the explicit sub-tower.

        :param: item_features: [batch_size, item_input_dim]
        :param: detach_shared: whether to detach the shared bottom features
        :return: [batch_size, embedding_dim]
        """
        i_shared = self.item_shared_bottom(item_features)
        if detach_shared:
            i_shared = i_shared.detach()
        i_exp = self.item_explicit_head(self.item_explicit_tower(i_shared))
        return self._maybe_normalize(i_exp)

    def encode_query_implicit(self, query_features: Tensor) -> Tensor:
        """Encode query features through the implicit sub-tower.

        :param: query_features: [batch_size, query_input_dim]
        :return: [batch_size, embedding_dim]
        """
        q_shared = self.query_shared_bottom(query_features)
        q_imp = self.query_implicit_head(self.query_implicit_tower(q_shared))
        return self._maybe_normalize(q_imp)

    def encode_item_implicit(self, item_features: Tensor) -> Tensor:
        """Encode item features through the implicit sub-tower.

        :param: item_features: [batch_size, item_input_dim]
        :return: [batch_size, embedding_dim]
        """
        i_shared = self.item_shared_bottom(item_features)
        i_imp = self.item_implicit_head(self.item_implicit_tower(i_shared))
        return self._maybe_normalize(i_imp)

    def forward(
        self,
        query_features: Tensor,
        item_features: Tensor,
        detach_shared_for_explicit: bool = False,
    ) -> SBTTOutput:
        """Forward pass for the shared bottom two-tower model.
        :param: query_features: [batch_size, query_input_dim]
        :param: item_features: [batch_size, item_input_dim]
        :param: detach_shared_for_explicit: whether to detach the shared bottom features for the explicit sub-tower
        :return: SBTTOutput containing explicit and implicit embeddings and scores
        """
        q_exp = self.encode_query_explicit(
            query_features, detach_shared=detach_shared_for_explicit
        )
        q_imp = self.encode_query_implicit(query_features)
        i_exp = self.encode_item_explicit(
            item_features, detach_shared=detach_shared_for_explicit
        )
        i_imp = self.encode_item_implicit(item_features)

        return SBTTOutput(
            explicit_query_embedding=q_exp,
            explicit_item_embedding=i_exp,
            explicit_score=self.dot_score(q_exp, i_exp),
            implicit_query_embedding=q_imp,
            implicit_item_embedding=i_imp,
            implicit_score=self.dot_score(q_imp, i_imp),
        )
