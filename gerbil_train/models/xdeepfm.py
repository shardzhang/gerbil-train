"""xDeepFM model with Compressed Interaction Network (CIN) support.

xDeepFM = Linear (Wide) + CIN (explicit vector-wise interactions) + Deep (MLP).
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import DeepFMModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["xDeepFM", "CIN"]


class CIN(nn.Module):
    """Compressed Interaction Network from the xDeepFM paper.

    Computes explicit feature interactions at multiple orders via
    convolution on outer products of field embeddings.

    Input:  X_0 — field embeddings ``[batch, num_fields, emb_dim]``
    Output: pooled features ``[batch, sum(layer_units)]``
    """

    def __init__(self, num_fields: int, emb_dim: int, layer_units: list[int]) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.layer_units = layer_units

        self.conv_layers = nn.ModuleList()
        for h in layer_units:
            self.conv_layers.append(nn.Conv1d(emb_dim, h, 1, bias=True))

    def forward(self, X_0: Tensor) -> Tensor:
        """Forward pass of CIN.

        :param X_0: Field embeddings ``[batch, num_fields, emb_dim]``
        :return: Pooled features ``[batch, sum(layer_units)]``
        """
        X_k = X_0  # [batch, m, d] at layer 0
        out: list[Tensor] = []
        for conv in self.conv_layers:
            # Outer product: [batch, H_k, m, d]
            Z = X_k.unsqueeze(2) * X_0.unsqueeze(1)
            batch, H_k, m, d = Z.shape
            Z = Z.view(batch, H_k * m, d)                 # [batch, H_k*m, d]
            X_k = conv(Z.transpose(1, 2))                  # [batch, h, d]
            X_k = X_k.transpose(1, 2)                      # [batch, h, d]
            out.append(X_k.sum(dim=1))                     # [batch, h]
        return torch.cat(out, dim=-1)                      # [batch, Σh]


class xDeepFM(BaseModel):
    """xDeepFM model for recommendation and CTR prediction.

    xDeepFM = Linear (Wide) + CIN (explicit vector-wise interactions) + Deep (MLP).
    """

    def __init__(self, model_cfg: DeepFMModelConfig) -> None:
        super().__init__()
        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        # Split fields by tower assignment
        self.wide_fields = {n: e for n, e in self.fields_cfg.items() if e.wide}
        self.deep_fields = {n: e for n, e in self.fields_cfg.items() if e.deep}
        # CIN requires embedding → only categorical/non-direct deep fields
        self.cin_fields = {
            n: e for n, e in self.deep_fields.items()
            if e.field_type == 1 or (e.field_type == 0 and e.concat_type == "emb")
        }

        self._validate_fields(model_cfg)

        self.deep_field_names = list(self.deep_fields.keys())
        self.cin_field_names = list(self.cin_fields.keys())
        self.wide_field_names = list(self.wide_fields.keys())

        # Per-field dims for deep input
        self.deep_field_dims: dict[str, int] = {}
        for field_name, entry in self.deep_fields.items():
            if entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb"):
                self.deep_field_dims[field_name] = int(entry.emb_size)
            elif entry.field_type == 0 and entry.concat_type == "direct":
                self.deep_field_dims[field_name] = int(entry.dim)
            else:
                raise ValueError(f"Unsupported field_type={entry.field_type} concat_type={entry.concat_type}")
        self.deep_sum_dim = sum(self.deep_field_dims.values())

        # EmbeddingBags for deep + CIN terms
        self.deep_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.deep_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.deep_embedding_bags:
                self.deep_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # EmbeddingBags for wide/linear term (vocab → 1)
        self.linear_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.wide_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.linear_embedding_bags:
                self.linear_embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=1,
                    mode="sum",
                )

        # CIN
        cin_layer_units = list(model_cfg.output.get("cin_layer_units", [128, 128]))
        cin_emb_dim = int(next(iter(self.cin_fields.values())).emb_size) if self.cin_fields else 0
        self.cin = CIN(
            num_fields=len(self.cin_fields),
            emb_dim=cin_emb_dim,
            layer_units=cin_layer_units,
        )
        self.cin_head = nn.Linear(sum(cin_layer_units), 1) if cin_layer_units else None

        # Deep network
        mlp_cfg = model_cfg.mlp
        self.deep_input_bn = nn.BatchNorm1d(self.deep_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None

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
        self.reset_parameters()

    def _validate_fields(self, model_cfg: DeepFMModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if self.cin_fields:
            first_emb = next(iter(self.cin_fields.values())).emb_size
            if not all(v.emb_size == first_emb for v in self.cin_fields.values()):
                raise ValueError("All CIN fields must have the same embedding size")

    def reset_parameters(self) -> None:
        if self.deep_head is not None:
            nn.init.xavier_uniform_(self.deep_head.weight)
            nn.init.zeros_(self.deep_head.bias)
        if self.cin_head is not None:
            nn.init.xavier_uniform_(self.cin_head.weight)
            nn.init.zeros_(self.cin_head.bias)
        for emb in self.deep_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.linear_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[next(iter(self.fields_cfg.keys()))]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Linear term (1st-order)
        linear_sum = torch.zeros(batch_size, device=device)
        for field_name, entry in self.wide_fields.items():
            if entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb"):
                linear_emb = embed_one_field(
                    self.linear_embedding_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                )
                linear_sum = linear_sum + linear_emb.squeeze(-1)
        linear_logit = linear_sum / max(len(self.wide_field_names), 1)

        # 2. CIN term (explicit vector-wise interactions)
        cin_emb_list: list[Tensor] = []
        for field_name, entry in self.cin_fields.items():
            feature_emb = embed_one_field(
                self.deep_embedding_bags[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            cin_emb_list.append(feature_emb)

        if cin_emb_list and self.cin_head is not None:
            cin_emb = torch.stack(cin_emb_list, dim=1)  # [batch, num_cin, d]
            cin_out = self.cin(cin_emb)                 # [batch, Σh]
            cin_logit = self.cin_head(cin_out)           # [batch, 1]
        else:
            cin_logit = torch.zeros(batch_size, device=device)

        # 3. Deep term (high-order non-linear via MLP)
        deep_emb_list: list[Tensor] = []
        for field_name, entry in self.deep_fields.items():
            cat_emb = entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb")
            if cat_emb:
                feature_emb = embed_one_field(
                    self.deep_embedding_bags[str(entry.field_index)],
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

        deep_input_emb = torch.cat(deep_emb_list, dim=-1)
        if self.deep_input_bn is not None:
            deep_input_emb = self.deep_input_bn(deep_input_emb)
        deep_logit = self.deep_head(self.deep_network(deep_input_emb)).squeeze(-1)

        logits = linear_logit + cin_logit.squeeze(-1) + deep_logit
        return torch.sigmoid(logits)
