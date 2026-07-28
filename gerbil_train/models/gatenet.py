"""GateNet: Gating-enhanced multi-task network with feature-level gates.

GateNet extends MMoE with per-field feature gating:
  1. Each field's embedding passes through a learnable sigmoid gate
  2. Gated features → MMoE backbone (shared experts + task gates + towers)

Reference: https://dl.acm.org/doi/10.1145/3459637.3481952 (KDD 2021)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["GateNet"]


class _FieldGate(nn.Module):
    """Element-wise sigmoid gate for a single field's embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.gate(x))


class GateNet(BaseModel):
    """Gating-enhanced multi-task network."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.field_names = list(self.embedding_fields.keys())

        # Embedding bags
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

        # Per-field gates
        self.field_gates = nn.ModuleDict()
        for field_name, entry in self.embedding_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                raw_dim = int(entry.dim)
                self.field_gates[field_name] = _FieldGate(raw_dim)
            else:
                self.field_gates[field_name] = _FieldGate(int(entry.emb_size))

        # Input dimension after gating
        self.input_dim = sum(
            int(e.dim if (e.field_type == 0 and e.concat_type == "direct") else e.emb_size)
            for fn, e in self.embedding_fields.items()
        )

        # GateNet config
        gt_cfg: dict[str, Any] = model_cfg.mlp
        num_experts = int(gt_cfg["num_experts"])
        expert_hidden = list(gt_cfg["expert_hidden"])
        gate_hidden = list(gt_cfg.get("gate_hidden", []))
        tower_hidden = list(gt_cfg["tower_hidden"])
        num_tasks = int(gt_cfg["num_tasks"])
        self.task_names = list(gt_cfg.get("task_names", [f"task_{i}" for i in range(num_tasks)]))
        self.dropout = float(gt_cfg.get("dropout", 0.1))

        # Experts
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            self.experts.append(FullyConnectedLayer(
                input_dim=self.input_dim, hidden_dims=expert_hidden,
                bias=[True] * len(expert_hidden),
                batch_norm=True, activation="relu", dropout=self.dropout,
            ))
        expert_output_dim = expert_hidden[-1]

        # Task gates
        self.gates = nn.ModuleList()
        for _ in range(num_tasks):
            layers: list[nn.Module] = []
            prev_dim = self.input_dim
            for h in gate_hidden:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, num_experts))
            self.gates.append(nn.Sequential(*layers))

        # Towers
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            self.towers.append(FullyConnectedLayer(
                input_dim=expert_output_dim, hidden_dims=tower_hidden,
                bias=[True] * len(tower_hidden),
                batch_norm=True, activation="relu", dropout=self.dropout,
            ))

        # Heads
        self.heads = nn.ModuleList()
        for _ in range(num_tasks):
            self.heads.append(nn.Linear(tower_hidden[-1], 1))

        self._validate_fields(model_cfg)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        for emb in self.embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> dict[str, Tensor]:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # Gate each field, then concat
        gated_list: list[Tensor] = []
        for field_name, entry in self.embedding_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                feat = feature_bags[field_name]["weights"].view(-1, int(entry.dim))
            else:
                feat = embed_one_field(
                    self.embedding_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                )
            gated_list.append(self.field_gates[field_name](feat))
        x = torch.cat(gated_list, dim=-1)

        # MMoE backbone
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)

        outputs: dict[str, Tensor] = {}
        for t, task_name in enumerate(self.task_names):
            gate_weights = F.softmax(self.gates[t](x), dim=-1)
            weighted = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
            tower_out = self.towers[t](weighted)
            logit = self.heads[t](tower_out).squeeze(-1)
            outputs[task_name] = torch.sigmoid(logit)

        return outputs
