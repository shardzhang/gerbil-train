"""DeepFM model with EmbeddingBag support for TFRecord input."""

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
    """DeepFM model for recommendation and CTR prediction."""

    def __init__(self, model_cfg: DeepFMModelConfig) -> None:
        super().__init__()

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        # Deep terms
        self.deep_fields = {k: v for (k, v) in self.embedding_fields.items()}
        # FM terms
        self.fm_fields = {k: v for (k, v) in self.embedding_fields.items() if v.field_type == 1}
        # All fields
        self.all_fields = self.deep_fields | self.fm_fields
        print(f"[debug] len(deep_fields): {len(self.deep_fields)}, len(fm_fields): {len(self.fm_fields)}, len(all_fields): {len(self.all_fields)}")
        
        self._validate_fields(model_cfg)

        self.deep_field_names = list(self.deep_fields.keys())
        self.fm_field_names = list(self.fm_fields.keys())
        self.all_field_names = list(self.all_fields.keys())

        # Embedding dimensions for each field
        self.field_embedding_dims: dict[str, int] = {}
        for field_name, entry in self.all_fields.items():
            if entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb"):
                self.field_embedding_dims[field_name] = entry.emb_size
            elif entry.field_type == 0 and entry.concat_type == "direct":
                self.field_embedding_dims[field_name] = entry.dim
            else:
                raise ValueError(f"Unsupported field_type {entry.field_type} or concat_type {entry.concat_type} for field {field_name}")
        self.deep_embedding_sum_dim = sum(self.field_embedding_dims.values())
        self.fm_embedding_sum_dim = sum(v.emb_size for k,v in self.fm_fields.items())

        # 1. EmbeddingBag for deep terms
        self.deep_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.embedding_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                print(f"[debug] Field {field_name}(field_type={entry.field_type}), concat_type={entry.concat_type}, skip Deep embedding")
                continue

            # Feature embedding: vocab → k, shared by FM 2nd-order and Deep terms
            if str(entry.field_index) not in self.deep_embedding_bags:
                feature_bag = nn.EmbeddingBag(
                    num_embeddings=entry.dim,
                    embedding_dim=entry.emb_size,
                    mode="sum",
                )
                feature_bag.field_name = f"{field_name}_deep"
                self.deep_embedding_bags[str(entry.field_index)] = feature_bag

        # 2.1 EmbeddingBag for linear terms
        self.linear_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.all_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                print(f"[debug] Field {field_name}(field_type={entry.field_type}), concat_type={entry.concat_type}, skip Linear embedding")
                continue

            # Linear embedding: vocab → 1, used for the 1st-order term
            if str(entry.field_index) not in self.linear_embedding_bags:
                linear_bag = nn.EmbeddingBag(
                    num_embeddings=entry.dim,
                    embedding_dim=1,
                    mode="sum",
                )
                linear_bag.field_name = f"{field_name}_linear"
                self.linear_embedding_bags[str(entry.field_index)] = linear_bag

        # 2.2 EmbeddingBag for FM terms
        self.fm_embedding_bags = nn.ModuleDict()
        for field_name, entry in self.fm_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                print(f"[debug] Field {field_name}(field_type={entry.field_type}), concat_type={entry.concat_type}, skip FM embedding")
                continue

            # Feature embedding: vocab → k, shared by FM 2nd-order and Deep terms
            if str(entry.field_index) not in self.fm_embedding_bags:
                feature_bag = self.deep_embedding_bags[str(entry.field_index)]
                feature_bag.field_name = f"{field_name}_fm"
                self.fm_embedding_bags[str(entry.field_index)] = feature_bag


        # Deep network
        mlp_cfg = model_cfg.mlp
        
        # BatchNorm on concatenated feature embeddings to prevent logit saturation from mode="sum"
        self.deep_input_bn = nn.BatchNorm1d(self.deep_embedding_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None
        self.fm_input_bn = nn.BatchNorm1d(self.fm_embedding_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None

        hidden_dims = list(mlp_cfg.get("hidden_dims", [128, 64]))
        self.deep_network = FullyConnectedLayer(
            input_dim=self.deep_embedding_sum_dim,
            hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", False)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        deep_output_dim = hidden_dims[-1] if hidden_dims else self.deep_embedding_sum_dim
        self.deep_head = nn.Linear(deep_output_dim, 1)
        self.reset_parameters()


    def _validate_fields(self, model_cfg: DeepFMModelConfig) -> None:
        """Validate that the embedding fields are properly configured."""
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        
        first_emb = next(iter(self.fm_fields.values())).emb_size
        if not all(entry.emb_size == first_emb for entry in self.fm_fields.values()):
            raise ValueError("All fields must have the same embedding size")


    def reset_parameters(self) -> None:
        """Reset model parameters."""
        if self.deep_head is not None:
            nn.init.xavier_uniform_(self.deep_head.weight)
            nn.init.zeros_(self.deep_head.bias)
        for emb in self.deep_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.linear_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.fm_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)


    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass of the DeepFM model.

        DeepFM = Linear (1st-order) + FM (2nd-order pair-wise) + Deep (high-order non-linear),
        all sharing the same feature embeddings.
        """
        first_offsets = feature_bags[self.deep_field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Deep term (high-order non-linear interactions via MLP)
        deep_emb_list: list[Tensor] = []
        for field_name, entry in self.deep_fields.items():
            if entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb"):
                # [batch_size, emb_dim]
                feature_emb = embed_one_field(
                    self.deep_embedding_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                )
                deep_emb_list.append(feature_emb)
            elif entry.field_type == 0 and entry.concat_type == "direct":
                # 连续特征仅参与Deep term
                deep_emb_list.append(feature_bags[field_name]["weights"].view(-1, entry.dim))
            else:
                raise ValueError(f"Unsupported field_type {entry.field_type} or concat_type {entry.concat_type} for field {field_name}")

        # [batch_size, num_fields * embedding_dim]
        deep_input_emb = torch.cat(deep_emb_list, dim=-1)
        # normalize feature embeddings before FM/Deep to prevent logit saturation
        if self.deep_input_bn is not None:
            deep_input_emb = self.deep_input_bn(deep_input_emb)
        # [batch_size, ]
        deep_logit = self.deep_head(self.deep_network(deep_input_emb)).squeeze(-1)

        
        # 2.1 Linear term (1st-order): sum of per-field linear embeddings + global bias
        # [batch_size, ]
        linear_sum = torch.zeros(batch_size, device=device)
        for field_name, entry in self.all_fields.items():
            if entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb"):
                # [batch_size, 1]
                linear_emb = embed_one_field(
                    self.linear_embedding_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                )
                linear_sum = linear_sum + linear_emb.squeeze(-1)
            elif entry.field_type == 0 and entry.concat_type == "direct":
                # 连续特征不参与FM term
                pass
            else:
                raise ValueError(f"Unsupported field_type {entry.field_type} or concat_type {entry.concat_type} for field {field_name}")
        # [batch_size, ]
        linear_logit = linear_sum / len(self.all_fields)

        # 2.2 FM second-order term: pair-wise feature interactions (2nd-order pair-wise)
        fm_emb_list: list[Tensor] = []
        for field_name, entry in self.fm_fields.items():
            if entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb"):
                # [batch_size, emb_dim]
                feature_emb = embed_one_field(
                    self.fm_embedding_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                )
                fm_emb_list.append(feature_emb)
            elif entry.field_type == 0 and entry.concat_type == "direct":
                # 连续特征不参与FM term
                pass
            else:
                raise ValueError(f"Unsupported field_type {entry.field_type} or concat_type {entry.concat_type} for field {field_name}")

        # [batch_size, num_fields * embedding_dim]
        fm_input_emb = torch.cat(fm_emb_list, dim=-1)
        # normalize feature embeddings before FM/Deep to prevent logit saturation
        if self.fm_input_bn is not None:
            fm_input_emb = self.fm_input_bn(fm_input_emb)
        # [batch_size, num_fields, embedding_dim]
        stacked = fm_input_emb.view(batch_size, len(self.fm_fields), -1)
        # [batch_size, emb_size]
        summed = stacked.sum(dim=1)
        # [batch_size, emb_size]
        sum_of_squares = (stacked * stacked).sum(dim=1)
        fm_logits = 0.5 * (summed * summed - sum_of_squares).sum(dim=1) / len(self.fm_fields)

        # total logits
        # [batch_size, ]
        logits = linear_logit + fm_logits + deep_logit
        return torch.sigmoid(logits)