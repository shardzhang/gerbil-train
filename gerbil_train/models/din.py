"""Deep Interest Network (DIN) for binary classification."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config import GwENModelConfig
from gerbil_train.utils.embedding import embed_one_field, to_device
from gerbil_train.utils.nn import build_mlp

__all__ = ["DIN"]


class DIN(nn.Module):
    """DIN for binary classification with behavior-sequence attention."""

    def __init__(self, config: GwENModelConfig, behavior_field: str) -> None:
        super().__init__()
        fields_cfg = config.embedding_fields
        if not fields_cfg:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if behavior_field not in fields_cfg:
            raise ValueError(f"behavior_field '{behavior_field}' not found in embedding_fields")

        self.behavior_field = behavior_field
        self.field_names = [n for n in fields_cfg if n != behavior_field]
        self.field_embedding_dims: dict[str, int] = {}
        self.field_embeddings = nn.ModuleDict()

        behavior_entry = fields_cfg[behavior_field]
        self.behavior_emb_dim = int(behavior_entry.emb_dim)
        self.behavior_embedding = nn.Embedding(
            num_embeddings=int(behavior_entry.vocab_size),
            embedding_dim=self.behavior_emb_dim,
        )

        for field_name in self.field_names:
            entry = fields_cfg[field_name]
            self.field_embedding_dims[field_name] = int(entry.emb_dim)
            self.field_embeddings[field_name] = nn.EmbeddingBag(
                num_embeddings=int(entry.vocab_size),
                embedding_dim=int(entry.emb_dim),
                mode="sum", include_last_offset=False,
            )

        attn_input_dim = self.behavior_emb_dim * 3
        self.attention_unit = nn.Sequential(
            nn.Linear(attn_input_dim, 36),
            nn.ReLU(),
            nn.Linear(36, 1),
        )

        self.embedding_sum_dim = sum(self.field_embedding_dims.values()) + self.behavior_emb_dim
        mlp_cfg = config.mlp
        hidden_dims = list(mlp_cfg.get("hidden_dims", [256, 128]))
        self.input_bn = nn.BatchNorm1d(self.embedding_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None
        self.mlp = build_mlp(
            input_dim=self.embedding_sum_dim, hidden_dims=hidden_dims,
            batch_norm=bool(mlp_cfg.get("batch_norm", True)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        final_hidden_dim = hidden_dims[-1] if hidden_dims else self.embedding_sum_dim
        self.head = nn.Linear(final_hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.behavior_embedding.weight)
        for emb in self.field_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for mod in self.mlp.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _compute_interest(self, behavior_bag: Mapping[str, Tensor], target_emb: Tensor,
                          batch_size: int, device: torch.device) -> Tensor:
        indices = to_device(behavior_bag["indices"].long(), device)
        offsets = to_device(behavior_bag["offsets"].long(), device)

        total_items = int(indices.size(0))
        behavior_embs = self.behavior_embedding(indices)

        if total_items > 0 and offsets.numel() > 1:
            arange = torch.arange(total_items, device=device)
            sample_ids = torch.searchsorted(offsets, arange, right=True) - 1
        else:
            sample_ids = torch.zeros(total_items, dtype=torch.long, device=device)

        target_expanded = target_emb[sample_ids]
        attn_input = torch.cat([behavior_embs, target_expanded, behavior_embs * target_expanded], dim=-1)
        scores = self.attention_unit(attn_input).squeeze(-1)

        padded_offsets = torch.cat([offsets, torch.tensor([total_items], device=device)])
        interest_list: list[Tensor] = []
        for i in range(batch_size):
            start = int(padded_offsets[i])
            end = int(padded_offsets[i + 1])
            if start >= end:
                interest_list.append(torch.zeros(self.behavior_emb_dim, device=device))
                continue
            seg_scores = scores[start:end]
            seg_embs = behavior_embs[start:end]
            w = torch.softmax(seg_scores, dim=0)
            interest_list.append((seg_embs * w.unsqueeze(-1)).sum(dim=0))

        return torch.stack(interest_list, dim=0)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        field_embs = [
            embed_one_field(self.field_embeddings[fn], feature_bags[fn]["indices"],
                            feature_bags[fn]["offsets"], feature_bags[fn]["weights"], device=device)
            for fn in self.field_names
        ]
        all_emb = torch.cat(field_embs, dim=-1)

        target_for_attention = field_embs[0]
        behavior_bag = feature_bags[self.behavior_field]
        interest_emb = self._compute_interest(behavior_bag, target_for_attention, batch_size, device)

        combined = torch.cat([all_emb, interest_emb], dim=-1)
        if self.input_bn is not None:
            combined = self.input_bn(combined)
        hidden = self.mlp(combined)
        return torch.sigmoid(self.head(hidden)).squeeze(-1)
