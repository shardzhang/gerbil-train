"""DeepFM model.

DeepFM combines three components:
    1. First-order linear terms
    2. Second-order factorization-machine interactions
    3. A deep neural network over dense features and field embeddings

Paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from gerbil_train.utils.nn import build_mlp

__all__ = ["DeepFM"]


class DeepFM(nn.Module):
    """DeepFM model for recommendation and CTR prediction."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize the DeepFM model from a config mapping.

        :param config: Model configuration mapping
        """
        super().__init__()

        self.dense_input_dim = int(config.get("dense_input_dim", 0))
        self.embedding_dim = int(config.get("embedding_dim", 16))
        self.sparse_fields = self._get_sparse_fields(config)
        self.field_names = list(self.sparse_fields.keys())
        self.output_activation = self._get_output_activation(config)

        if self.dense_input_dim <= 0 and not self.field_names:
            raise ValueError("DeepFM requires at least one dense or sparse feature.")

        self.linear_embeddings = nn.ModuleDict()
        self.feature_embeddings = nn.ModuleDict()
        for field_name, field_config in self.sparse_fields.items():
            vocab_size = self._get_vocab_size(field_name, field_config)
            padding_idx = self._get_padding_idx(field_config)
            self.linear_embeddings[field_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=1,
                padding_idx=padding_idx,
            )
            self.feature_embeddings[field_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=self.embedding_dim,
                padding_idx=padding_idx,
            )

        self.linear_dense = (
            nn.Linear(self.dense_input_dim, 1, bias=False)
            if self.dense_input_dim > 0
            else None
        )
        self.bias = nn.Parameter(torch.zeros(1))

        deep_config = self._get_deep_config(config)
        self.deep_hidden_dims = list(deep_config.get("hidden_dims", [128, 64]))
        self.deep_input_dim = (
            self.dense_input_dim + len(self.field_names) * self.embedding_dim
        )

        self.deep_network = (
            build_mlp(
                input_dim=self.deep_input_dim,
                hidden_dims=self.deep_hidden_dims,
                batch_norm=bool(deep_config.get("batch_norm", False)),
                activation=str(deep_config.get("activation", "relu")),
                dropout=float(deep_config.get("dropout", 0.0)),
            )
            if self.deep_input_dim > 0
            else None
        )
        deep_output_dim = (
            self.deep_hidden_dims[-1] if self.deep_hidden_dims else self.deep_input_dim
        )
        self.deep_head = nn.Linear(deep_output_dim, 1) if deep_output_dim > 0 else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trainable parameters."""
        if self.linear_dense is not None:
            nn.init.xavier_uniform_(self.linear_dense.weight)
        if self.deep_head is not None:
            nn.init.xavier_uniform_(self.deep_head.weight)
            nn.init.zeros_(self.deep_head.bias)
        for embedding in self.linear_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        for embedding in self.feature_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        *,
        dense_features: Tensor | None = None,
        sparse_features: Tensor | Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Run the DeepFM forward pass.

        :param dense_features: Dense feature tensor of shape ``[batch_size, dense_input_dim]``
        :param sparse_features: Sparse features provided either as
            ``Tensor[batch_size, num_fields]`` or ``dict[field_name, Tensor[batch_size]]``
        :return: Output score tensor of shape ``[batch_size]``
        """
        batch_size = self._get_batch_size(dense_features, sparse_features)
        dense_features = self._prepare_dense_features(dense_features, batch_size)
        sparse_inputs = self._prepare_sparse_features(sparse_features, batch_size)

        device = self._get_device(dense_features, sparse_inputs)
        logits = self.bias.expand(batch_size).to(device)

        if dense_features is not None and self.linear_dense is not None:
            logits = logits + self.linear_dense(dense_features).squeeze(-1)

        sparse_embedding_list: list[Tensor] = []
        if sparse_inputs:
            first_order_terms: list[Tensor] = []
            for field_name in self.field_names:
                field_values = sparse_inputs[field_name].long()
                first_order_terms.append(
                    self.linear_embeddings[field_name](field_values).squeeze(-1)
                )
                sparse_embedding_list.append(
                    self.feature_embeddings[field_name](field_values)
                )

            logits = logits + torch.stack(first_order_terms, dim=0).sum(dim=0)
            logits = logits + self._compute_fm_term(sparse_embedding_list)

        deep_term = self._compute_deep_term(
            dense_features, sparse_embedding_list, batch_size, device
        )
        logits = logits + deep_term
        return self._apply_output_activation(logits)

    @staticmethod
    def _get_sparse_fields(config: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the sparse-field config mapping."""
        sparse_fields = config.get("sparse_fields", {})
        if not isinstance(sparse_fields, Mapping):
            raise ValueError("sparse_fields must be a mapping")
        return sparse_fields

    @staticmethod
    def _get_deep_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the deep-network config mapping."""
        deep_config = config.get("deep", {})
        if not isinstance(deep_config, Mapping):
            raise ValueError("deep must be a mapping")
        return deep_config

    @staticmethod
    def _get_output_activation(config: Mapping[str, Any]) -> str | None:
        """Return the optional output activation name."""
        output_config = config.get("output", {})
        if not isinstance(output_config, Mapping):
            raise ValueError("output must be a mapping")
        activation = output_config.get("activation")
        if activation is None:
            return None
        normalized_activation = str(activation).lower()
        if normalized_activation in {"", "identity", "none"}:
            return None
        if normalized_activation == "sigmoid":
            return normalized_activation
        raise ValueError(f"Unsupported output activation: {activation}")

    @staticmethod
    def _get_vocab_size(field_name: str, field_config: Any) -> int:
        """Return the vocabulary size for one sparse field."""
        if isinstance(field_config, int):
            return int(field_config)
        if isinstance(field_config, Mapping) and "vocab_size" in field_config:
            return int(field_config["vocab_size"])
        raise ValueError(f"{field_name} must define vocab_size")

    @staticmethod
    def _get_padding_idx(field_config: Any) -> int | None:
        """Return the optional padding index for one sparse field."""
        if isinstance(field_config, Mapping) and "padding_idx" in field_config:
            return int(field_config["padding_idx"])
        return None

    def _get_batch_size(
        self,
        dense_features: Tensor | None,
        sparse_features: Tensor | Mapping[str, Tensor] | None,
    ) -> int:
        """Infer the batch size from input tensors."""
        if dense_features is not None:
            return int(dense_features.size(0))
        if isinstance(sparse_features, Tensor):
            return int(sparse_features.size(0))
        if isinstance(sparse_features, Mapping) and sparse_features:
            first_tensor = next(iter(sparse_features.values()))
            return int(first_tensor.size(0))
        raise ValueError("At least one input tensor is required to infer batch size.")

    def _prepare_dense_features(
        self,
        dense_features: Tensor | None,
        batch_size: int,
    ) -> Tensor | None:
        """Validate and normalize dense feature inputs."""
        if self.dense_input_dim == 0:
            if dense_features is not None:
                raise ValueError("This DeepFM config does not expect dense features.")
            return None

        if dense_features is None:
            raise ValueError("dense_features are required by this DeepFM config.")
        if dense_features.dim() != 2 or dense_features.size(1) != self.dense_input_dim:
            raise ValueError(
                f"dense_features must have shape [batch_size, {self.dense_input_dim}]"
            )
        if dense_features.size(0) != batch_size:
            raise ValueError("dense_features batch size does not match other inputs.")
        return dense_features.float()

    def _prepare_sparse_features(
        self,
        sparse_features: Tensor | Mapping[str, Tensor] | None,
        batch_size: int,
    ) -> dict[str, Tensor]:
        """Validate and normalize sparse feature inputs."""
        if not self.field_names:
            if sparse_features is not None:
                raise ValueError("This DeepFM config does not expect sparse features.")
            return {}

        if sparse_features is None:
            raise ValueError("sparse_features are required by this DeepFM config.")

        if isinstance(sparse_features, Tensor):
            if sparse_features.dim() != 2 or sparse_features.size(1) != len(
                self.field_names
            ):
                raise ValueError(
                    "Tensor sparse_features must have shape "
                    f"[batch_size, {len(self.field_names)}]"
                )
            return {
                field_name: sparse_features[:, index]
                for index, field_name in enumerate(self.field_names)
            }

        if not isinstance(sparse_features, Mapping):
            raise ValueError("sparse_features must be a tensor or mapping")

        normalized_inputs: dict[str, Tensor] = {}
        for field_name in self.field_names:
            if field_name not in sparse_features:
                raise ValueError(f"Missing sparse feature: {field_name}")
            field_values = sparse_features[field_name]
            if field_values.dim() != 1 or field_values.size(0) != batch_size:
                raise ValueError(
                    f"Sparse field {field_name} must have shape [batch_size]"
                )
            normalized_inputs[field_name] = field_values
        return normalized_inputs

    @staticmethod
    def _get_device(
        dense_features: Tensor | None,
        sparse_inputs: Mapping[str, Tensor],
    ) -> torch.device:
        """Return the device used by the input tensors."""
        if dense_features is not None:
            return dense_features.device
        if sparse_inputs:
            return next(iter(sparse_inputs.values())).device
        return torch.device("cpu")

    def _compute_fm_term(self, sparse_embedding_list: Sequence[Tensor]) -> Tensor:
        """Compute second-order factorization-machine interactions."""
        stacked_embeddings = torch.stack(sparse_embedding_list, dim=1)
        summed_embeddings = stacked_embeddings.sum(dim=1)
        squared_sum = summed_embeddings * summed_embeddings
        sum_of_squares = (stacked_embeddings * stacked_embeddings).sum(dim=1)
        return 0.5 * (squared_sum - sum_of_squares).sum(dim=1)

    def _compute_deep_term(
        self,
        dense_features: Tensor | None,
        sparse_embedding_list: Sequence[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Compute the deep-network contribution to the final score."""
        if self.deep_network is None or self.deep_head is None:
            return torch.zeros(batch_size, device=device)

        deep_inputs: list[Tensor] = []
        if dense_features is not None:
            deep_inputs.append(dense_features)
        if sparse_embedding_list:
            deep_inputs.append(torch.cat(list(sparse_embedding_list), dim=-1))
        deep_input = torch.cat(deep_inputs, dim=-1)
        deep_hidden = self.deep_network(deep_input)
        return self.deep_head(deep_hidden).squeeze(-1)

    def _apply_output_activation(self, logits: Tensor) -> Tensor:
        """Apply the optional output activation to the final logits."""
        if self.output_activation == "sigmoid":
            return torch.sigmoid(logits)
        return logits
