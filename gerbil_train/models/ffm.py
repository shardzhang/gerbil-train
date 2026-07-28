"""FFM (Field-aware Factorization Machine) for CTR prediction.

FFM extends FM by learning a separate embedding for each field pair,
so feature i uses a different embedding when interacting with field f_j.

    y = w_0 + Σ w_i x_i + Σ_{i<j} ⟨v_{i,f_j}, v_{j,f_i}⟩

Reference: https://doi.org/10.1145/2959100.2959137 (ACM 2016)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.base_model import BaseModel

__all__ = ["FFM"]


class FFM(BaseModel):
    """Field-aware Factorization Machine for CTR prediction."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.fields_cfg.keys())

        # Only categorical fields (not concat_type="direct") participate in FFM
        self.ffm_field_names = [n for n in self.field_names
                                if not (self.fields_cfg[n].field_type == 0 and self.fields_cfg[n].concat_type == "direct")]
        self.num_fields = len(self.ffm_field_names)
        self.emb_size = int(self.fields_cfg[self.ffm_field_names[0]].emb_size)

        # Field pairs for FFM term (upper triangle)
        self.field_pairs: list[tuple[str, str]] = []
        for i, fi in enumerate(self.ffm_field_names):
            for fj in self.ffm_field_names[i + 1:]:
                self.field_pairs.append((fi, fj))

        # Linear embeddings: vocab → 1
        self.linear_embeddings = nn.ModuleDict()
        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.linear_embeddings:
                self.linear_embeddings[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=1,
                    mode="sum",
                )

        # Field-aware pair embeddings: v_{i, f_j} (EmbeddingBag with field_i's vocab)
        # Key: f"{field_name_i}→{field_name_j}" for i's embedding when interacting with j
        self.ffm_embeddings = nn.ModuleDict()
        for fi, fj in self.field_pairs:
            entry_i = self.fields_cfg[fi]
            entry_j = self.fields_cfg[fj]
            key_ij = f"{fi}→{fj}"
            key_ji = f"{fj}→{fi}"
            if key_ij not in self.ffm_embeddings:
                self.ffm_embeddings[key_ij] = nn.EmbeddingBag(
                    num_embeddings=int(entry_i.dim),
                    embedding_dim=int(self.emb_size),
                    mode="sum",
                )
            if key_ji not in self.ffm_embeddings:
                self.ffm_embeddings[key_ji] = nn.EmbeddingBag(
                    num_embeddings=int(entry_j.dim),
                    embedding_dim=int(self.emb_size),
                    mode="sum",
                )

        # Direct/continuous fields → plain concat
        self.direct_field_names = [n for n in self.field_names
                                   if self.fields_cfg[n].field_type == 0 and self.fields_cfg[n].concat_type == "direct"]

        self.bias = nn.Parameter(torch.zeros(1))

        # Final head: linear(1 + 1 + sum_direct → 1)
        # FFM logit + linear sum + direct features
        direct_dim = sum(int(self.fields_cfg[n].dim) for n in self.direct_field_names)
        self.head = nn.Linear(2 + direct_dim, 1)

        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        emb_sizes = {int(e.emb_size) for e in model_cfg.embedding_fields.values()
                     if not (e.field_type == 0 and e.concat_type == "direct")}
        if len(emb_sizes) > 1:
            raise ValueError(f"FFM requires all field embeddings to have the same size, got {emb_sizes}")

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.bias)
        for emb in self.linear_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.ffm_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Linear term: w_0 + Σ w_i · x_i
        linear_sum = self.bias.expand(batch_size).to(device)
        for field_name, entry in self.fields_cfg.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            linear_emb = embed_one_field(
                self.linear_embeddings[str(entry.field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            linear_sum = linear_sum + linear_emb.squeeze(-1)

        # 2. FFM term: Σ_{i<j} ⟨v_{i,f_j}, v_{j,f_i}⟩
        #   v_{i,f_j}: field_i's embedding when interacting with field_j
        #   v_{j,f_i}: field_j's embedding when interacting with field_i
        ffm_logits = torch.zeros(batch_size, device=device)
        for fi, fj in self.field_pairs:
            entry_i = self.fields_cfg[fi]
            v_i_to_j = embed_one_field(
                self.ffm_embeddings[f"{fi}→{fj}"],
                feature_bags[fi]["indices"],
                feature_bags[fi]["offsets"],
                feature_bags[fi]["weights"],
                device=device,
            )
            v_j_to_i = embed_one_field(
                self.ffm_embeddings[f"{fj}→{fi}"],
                feature_bags[fj]["indices"],
                feature_bags[fj]["offsets"],
                feature_bags[fj]["weights"],
                device=device,
            )
            ffm_logits = ffm_logits + (v_i_to_j * v_j_to_i).sum(dim=-1)

        # 3. Direct/continuous features
        direct_embs: list[Tensor] = []
        for fn in self.direct_field_names:
            raw = feature_bags[fn]["weights"]
            direct_embs.append(raw.view(batch_size, -1))
        direct_concat = torch.cat(direct_embs, dim=-1) if direct_embs else torch.zeros(batch_size, 0, device=device)

        # 4. Fusion
        combined = torch.cat([linear_sum.unsqueeze(-1), ffm_logits.unsqueeze(-1), direct_concat], dim=-1)
        logit = self.head(combined).squeeze(-1)
        return torch.sigmoid(logit)
