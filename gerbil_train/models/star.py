"""STAR (Star Topology Adaptive Recommender) for multi-scenario CTR.

STAR uses a star topology neural network: a shared center network + scenario-
specific networks. Each scenario's parameters are composed as:

    W = W_shared ⊙ W_scenario,  b = b_shared + b_scenario

Reference: https://dl.acm.org/doi/10.1145/3459637.3482412 (WSDM 2022)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.base_model import BaseModel

__all__ = ["STAR"]


class _StarFC(nn.Module):
    """Single FC layer with star-topology parameter composition.

    For each scenario s:
        W_eff = W_shared ⊙ W_specific[s]   (element-wise)
        b_eff = b_shared + b_specific[s]
    """

    def __init__(self, in_dim: int, out_dim: int, num_scenarios: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.shared_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.shared_bias = nn.Parameter(torch.zeros(out_dim))
        self.specific_weight = nn.Parameter(torch.randn(num_scenarios, out_dim, in_dim) * 0.1)
        self.specific_bias = nn.Parameter(torch.zeros(num_scenarios, out_dim))

    def forward(self, x: Tensor, scenario_ids: Tensor) -> Tensor:
        B = x.size(0)
        device = x.device
        output = torch.zeros(B, self.out_dim, device=device)
        for sid in torch.unique(scenario_ids):
            mask = scenario_ids == sid
            w = self.shared_weight * self.specific_weight[sid]
            b = self.shared_bias + self.specific_bias[sid]
            output[mask] = F.linear(x[mask], w, b)
        return output


class _StarMLP(nn.Module):
    """Stack of StarFC layers with ReLU activation."""

    def __init__(self, dims: list[int], num_scenarios: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(_StarFC(dims[i], dims[i + 1], num_scenarios))

    def forward(self, x: Tensor, scenario_ids: Tensor) -> Tensor:
        for layer in self.layers:
            x = F.relu(layer(x, scenario_ids))
        return x


class STAR(BaseModel):
    """Star Topology Adaptive Recommender for multi-scenario CTR."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.embedding_fields.keys())

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

        self.input_dim = sum(
            int(e.emb_size) for fn, e in self.embedding_fields.items()
            if not (e.field_type == 0 and e.concat_type == "direct")
        )
        direct_dim = sum(
            int(e.dim) for fn, e in self.embedding_fields.items()
            if e.field_type == 0 and e.concat_type == "direct"
        )
        self.input_dim += direct_dim

        star_cfg: dict[str, Any] = model_cfg.mlp
        self.domain_field = str(star_cfg["domain_field"])
        hidden_dims = list(star_cfg["hidden_dims"])
        self.num_scenarios = int(star_cfg["num_scenarios"])

        dims = [self.input_dim] + hidden_dims + [1]
        self.star_mlp = _StarMLP(dims, self.num_scenarios)
        self._validate_fields(model_cfg)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if "domain_field" not in model_cfg.mlp:
            raise ValueError("STAR config must specify domain_field")

    def reset_parameters(self) -> None:
        for emb in self.embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

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

        # Scenario IDs from domain field (single-valued)
        offsets = feature_bags[self.domain_field]["offsets"].to(device)
        indices = feature_bags[self.domain_field]["indices"].to(device)
        scenario_ids = (indices[offsets] % self.num_scenarios)

        logit = self.star_mlp(x, scenario_ids).squeeze(-1)
        return torch.sigmoid(logit)
