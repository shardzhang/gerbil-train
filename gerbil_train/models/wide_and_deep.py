"""Wide & Deep model for CTR prediction.

Wide & Deep = Wide (1st-order linear) + Deep (MLP),
each field can be configured to enter Wide, Deep, or both.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import WideAndDeepModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["WideAndDeep"]


class WideAndDeep(BaseModel):
    """Wide & Deep model for recommendation and CTR prediction.

    Each field in ``embedding_fields`` can be configured with ``wide`` / ``deep``
    flags to control which tower it enters:

    - ``wide=True, deep=True``  (default): both towers
    - ``wide=True, deep=False``: only Wide (linear)
    - ``wide=False, deep=True``: only Deep (MLP)
    """
    def __init__(self, model_cfg: WideAndDeepModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())

        # Separate fields by tower assignment
        self.wide_fields = {n: e for n, e in self.fields_cfg.items() if e.wide}
        self.deep_fields = {n: e for n, e in self.fields_cfg.items() if e.deep}
        self.wide_field_names = list(self.wide_fields.keys())
        self.deep_field_names = list(self.deep_fields.keys())

        # Compute per-field embedding dimensions for deep input
        self.deep_field_dims: dict[str, int] = {}
        for field_name, entry in self.deep_fields.items():
            cat_emb = entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb")
            if cat_emb:
                self.deep_field_dims[field_name] = int(entry.emb_size)
            elif entry.field_type == 0 and entry.concat_type == "direct":
                self.deep_field_dims[field_name] = int(entry.dim)
            else:
                raise ValueError(f"Unsupported field_type={entry.field_type} concat_type={entry.concat_type}")
        self.deep_sum_dim = sum(self.deep_field_dims.values())

        # Linear (wide) embeddings: vocab → 1, for wide-only + both fields
        self.linear_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.wide_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.linear_embedding_bags:
                bag = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=1,
                    mode="sum",
                )
                bag.field_name = f"{field_name}_linear"
                self.linear_embedding_bags[key] = bag

        # Feature (deep) embeddings: vocab → k, for deep-only + both fields
        self.feature_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.deep_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.feature_embedding_bags:
                bag = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )
                bag.field_name = f"{field_name}_deep"
                self.feature_embedding_bags[key] = bag

        # Deep network
        mlp_cfg = model_cfg.mlp
        self.input_bn = nn.BatchNorm1d(self.deep_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None

        hidden_dims = list(mlp_cfg.get("hidden_dims", [128, 64]))
        self.deep_network = FullyConnectedLayer(
            input_dim=self.deep_sum_dim,
            hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", False)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        deep_output_dim = hidden_dims[-1] if hidden_dims else self.deep_sum_dim
        self.deep_head = nn.Linear(deep_output_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    @staticmethod
    def _validate_fields(model_cfg: WideAndDeepModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        """initialize model parameters"""
        nn.init.zeros_(self.bias)
        if self.deep_head is not None:
            nn.init.xavier_uniform_(self.deep_head.weight)
            nn.init.zeros_(self.deep_head.bias)
        for emb in self.linear_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.feature_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass of the Wide and Deep model.
        
        :return: Predicted scores for each sample in the batch.
        """ 
        first_offsets = feature_bags[next(iter(self.fields_cfg.keys()))]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Wide term (1st-order linear)
        #    w_0 + Σ w_i · x_i
        linear_sum = torch.zeros(batch_size, device=device)
        for field_name, entry in self.wide_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            linear_emb = embed_one_field(
                self.linear_embedding_bags[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            linear_sum = linear_sum + linear_emb.squeeze(-1)

        # 2. Deep term: high-order non-linear interactions via MLP
        #    Deep = MLP(concat(\mathbf{e}_1, ..., \mathbf{e}_n))
        # [batch_size, num_fields * embedding_dim]
        deep_emb_list: list[Tensor] = []
        for field_name, entry in self.deep_fields.items():
            cat_emb = entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb")
            if cat_emb:
                feature_emb = embed_one_field(
                    self.feature_embedding_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                )
            elif entry.field_type == 0 and entry.concat_type == "direct":
                feature_emb = feature_bags[field_name]["weights"].view(-1, int(entry.dim))
            else:
                raise ValueError(f"Unsupported field_type={entry.field_type} concat_type={entry.concat_type}")
            deep_emb_list.append(feature_emb)

        deep_input = torch.cat(deep_emb_list, dim=-1)
        if self.input_bn is not None:
            deep_input = self.input_bn(deep_input)
        hidden = self.deep_network(deep_input)
        deep_logit = self.deep_head(hidden).squeeze(-1)

        logits = linear_sum + self.bias + deep_logit
        return torch.sigmoid(logits)
