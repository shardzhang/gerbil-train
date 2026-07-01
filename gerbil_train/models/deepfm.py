"""DeepFM model with EmbeddingBag support for TFRecord input.

DeepFM = Linear (Wide) + FM (2nd-order pair-wise) + Deep (MLP),
with per-field wide/deep control.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import DeepFMModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["DeepFM"]


class DeepFM(BaseModel):
    """DeepFM model for recommendation and CTR prediction.

    Each field can be configured with ``wide`` / ``deep`` flags:

    - ``wide=True, deep=True``  (default): Linear + FM + Deep
    - ``wide=True, deep=False``: Linear only (no FM, no Deep)
    - ``wide=False, deep=True``: FM + Deep only (no Linear)
    - ``field_type=0, concat_type=direct``: Deep only, no FM, no Linear (raw values)
    """

    def __init__(self, model_cfg: DeepFMModelConfig) -> None:
        super().__init__()
        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        # Split fields by tower assignment
        self.wide_fields = {n: e for n, e in self.fields_cfg.items() if e.wide}
        self.deep_fields = {n: e for n, e in self.fields_cfg.items() if e.deep}
        # FM requires embedding → only categorical/non-direct deep fields
        self.fm_fields = {
            n: e for n, e in self.deep_fields.items()
            if e.field_type == 1 or (e.field_type == 0 and e.concat_type == "emb")
        }

        self._validate_fields(model_cfg)

        self.deep_field_names = list(self.deep_fields.keys())
        self.fm_field_names = list(self.fm_fields.keys())
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
        self.fm_sum_dim = sum(int(v.emb_size) for v in self.fm_fields.values())

        # EmbeddingBags for deep term (vocab → k)
        self.deep_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.deep_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.deep_embedding_bags:
                bag = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )
                bag.field_name = f"{field_name}_deep"
                self.deep_embedding_bags[key] = bag

        # EmbeddingBags for wide/linear term (vocab → 1)
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

        # FM reuses deep embedding bags (shared feature embeddings)
        self.fm_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.fm_fields.items():
            key = str(entry.field_index)
            bag = self.deep_embedding_bags[key]
            bag.field_name = f"{field_name}_fm"
            self.fm_embedding_bags[key] = bag

        # Deep network
        mlp_cfg = model_cfg.mlp
        self.deep_input_bn = nn.BatchNorm1d(self.deep_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None
        self.fm_input_bn = nn.BatchNorm1d(self.fm_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None

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
        if self.fm_fields:
            first_emb = next(iter(self.fm_fields.values())).emb_size
            if not all(v.emb_size == first_emb for v in self.fm_fields.values()):
                raise ValueError("All FM fields must have the same embedding size")


    def reset_parameters(self) -> None:
        if self.deep_head is not None:
            nn.init.xavier_uniform_(self.deep_head.weight)
            nn.init.zeros_(self.deep_head.bias)
        for emb in self.deep_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.linear_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)


    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[next(iter(self.fields_cfg.keys()))]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Linear term (1st-order): sum of per-field linear embeddings
        # w_0 + Σ w_i · x_i , where w_i = EmbeddingBag(vocab → 1)
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

        # 2. FM second-order term: pair-wise feature interactions
        # FM = 0.5 * ((Σ v)² - Σ(v²)) = Σ_{i<j} ⟨v_i, v_j⟩
        # where v_i is the feature embedding for field i.
        fm_emb_list: list[Tensor] = []
        for field_name, entry in self.fm_fields.items():
            feature_emb = embed_one_field(
                self.fm_embedding_bags[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            fm_emb_list.append(feature_emb)

        if fm_emb_list:
            fm_input_emb = torch.cat(fm_emb_list, dim=-1)
            if self.fm_input_bn is not None:
                fm_input_emb = self.fm_input_bn(fm_input_emb)
            stacked = fm_input_emb.view(batch_size, len(self.fm_fields), -1)
            summed = stacked.sum(dim=1)
            sum_of_squares = (stacked * stacked).sum(dim=1)
            fm_logits = 0.5 * (summed * summed - sum_of_squares).sum(dim=1) / len(self.fm_fields)
        else:
            fm_logits = torch.zeros(batch_size, device=device)

        # 3. Deep term: high-order non-linear interactions via MLP
        # Deep = MLP(concat(v_1, ..., v_n))
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

        logits = linear_logit + fm_logits + deep_logit
        return torch.sigmoid(logits)
