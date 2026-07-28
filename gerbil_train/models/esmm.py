"""ESMM (Entire Space Multi-Task Model) for CVR prediction.

ESMM jointly predicts CTR (Click-Through Rate) and CVR (Conversion Rate)
using a multi-task architecture. The key insight is that pCTCVR = pCTR × pCVR,
so CVR can be learned on the entire impression space (not just clicked samples).

Reference: https://doi.org/10.1145/3209978.3210104 (KDD 2018)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["ESMM"]


class ESMM(BaseModel):
    """Entire Space Multi-Task Model for CTR + CVR prediction."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.embedding_fields.keys())

        # Embedding bags for all fields
        self.embedding_bags = nn.ModuleDict()
        for field_name, entry in self.embedding_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                continue
            key = str(entry.field_index)
            if key not in self.embedding_bags:
                self.embedding_bags[key] = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                )

        # Input dimension
        self.input_dim = sum(
            int(e.emb_size) for fn, e in self.embedding_fields.items()
            if not (e.field_type == 0 and e.concat_type == "direct")
        )
        direct_dim = sum(
            int(e.dim) for fn, e in self.embedding_fields.items()
            if e.field_type == 0 and e.concat_type == "direct"
        )
        self.input_dim += direct_dim

        # ESMM config
        esmm_cfg: dict[str, Any] = model_cfg.mlp
        ctr_hidden = list(esmm_cfg["ctr_hidden"])
        cvr_hidden = list(esmm_cfg["cvr_hidden"])
        shared_hidden = list(esmm_cfg.get("shared_hidden", []))

        # Shared bottom
        if shared_hidden:
            self.shared_bottom = FullyConnectedLayer(
                input_dim=self.input_dim, hidden_dims=shared_hidden,
                bias=[True] * len(shared_hidden),
                batch_norm=True, activation="relu", dropout=0.1,
            )
            shared_dim = shared_hidden[-1]
        else:
            self.shared_bottom = nn.Identity()
            shared_dim = self.input_dim

        # CTR tower
        self.ctr_tower = FullyConnectedLayer(
            input_dim=shared_dim, hidden_dims=ctr_hidden,
            bias=[True] * len(ctr_hidden),
            batch_norm=True, activation="relu", dropout=0.1,
        )
        self.ctr_head = nn.Linear(ctr_hidden[-1], 1)

        # CVR tower
        self.cvr_tower = FullyConnectedLayer(
            input_dim=shared_dim, hidden_dims=cvr_hidden,
            bias=[True] * len(cvr_hidden),
            batch_norm=True, activation="relu", dropout=0.1,
        )
        self.cvr_head = nn.Linear(cvr_hidden[-1], 1)

        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        for emb in self.embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        nn.init.xavier_uniform_(self.ctr_head.weight)
        nn.init.zeros_(self.ctr_head.bias)
        nn.init.xavier_uniform_(self.cvr_head.weight)
        nn.init.zeros_(self.cvr_head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> dict[str, Tensor]:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # Embed all fields
        emb_list: list[Tensor] = []
        for field_name, entry in self.embedding_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                emb_list.append(feature_bags[field_name]["weights"].view(-1, int(entry.dim)))
            else:
                emb_list.append(embed_one_field(
                    self.embedding_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                ))
        x = torch.cat(emb_list, dim=-1)

        # Shared bottom
        shared = self.shared_bottom(x)

        # CTR tower
        ctr_out = self.ctr_tower(shared)
        pctr = torch.sigmoid(self.ctr_head(ctr_out).squeeze(-1))

        # CVR tower
        cvr_out = self.cvr_tower(shared)
        pcvr = torch.sigmoid(self.cvr_head(cvr_out).squeeze(-1))

        # pCTCVR = pCTR × pCVR
        pctcvr = pctr * pcvr

        return {"pctr": pctr, "pcvr": pcvr, "pctcvr": pctcvr}
