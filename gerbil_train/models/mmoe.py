"""MMoE (Multi-gate Mixture-of-Experts) for multi-task CTR prediction.

MMoE uses K shared expert networks and T task-specific gates. Each gate
produces a weighted combination of experts, which is fed into a task-specific
tower.

    f_t(x) = tower_t( Σ_k g_t(x)_k · expert_k(x) )

Reference: https://dl.acm.org/doi/10.1145/3219819.3220007 (KDD 2018)
"""

from __future__ import annotations

from typing import Mapping, Any, Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["MMoE"]


class MMoE(BaseModel):
    """Multi-gate Mixture-of-Experts for multi-task binary classification."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

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

        # Compute embedding input dimension
        self.emb_concat_dim = sum(
            int(e.emb_size) for fn, e in self.embedding_fields.items()
            if not (e.field_type == 0 and e.concat_type == "direct")
        )
        # Plus direct field dimensions
        direct_dim = sum(
            int(e.dim) for fn, e in self.embedding_fields.items()
            if e.field_type == 0 and e.concat_type == "direct"
        )
        self.input_dim = self.emb_concat_dim + direct_dim

        # MMoE config
        mmoe_cfg: dict[str, Any] = model_cfg.mlp
        num_experts = int(mmoe_cfg["num_experts"])
        expert_hidden = list(mmoe_cfg["expert_hidden"])
        gate_hidden = list(mmoe_cfg.get("gate_hidden", []))
        tower_hidden = list(mmoe_cfg["tower_hidden"])
        num_tasks = int(mmoe_cfg["num_tasks"])
        self.task_names = list(mmoe_cfg.get("task_names", [f"task_{i}" for i in range(num_tasks)]))
        self.dropout = float(mmoe_cfg.get("dropout", 0.0))

        # Experts: K independent MLPs
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            self.experts.append(FullyConnectedLayer(
                input_dim=self.input_dim,
                hidden_dims=expert_hidden,
                bias=[True] * len(expert_hidden),
                batch_norm=True,
                activation="relu",
                dropout=self.dropout,
            ))
        expert_output_dim = expert_hidden[-1]

        # Gates: T task-specific softmax gates
        self.gates = nn.ModuleList()
        for _ in range(num_tasks):
            gate_layers: list[nn.Module] = []
            prev_dim = self.input_dim
            for h in gate_hidden:
                gate_layers.append(nn.Linear(prev_dim, h))
                gate_layers.append(nn.ReLU())
                gate_layers.append(nn.Dropout(self.dropout))
                prev_dim = h
            gate_layers.append(nn.Linear(prev_dim, num_experts))
            self.gates.append(nn.Sequential(*gate_layers))

        # Towers: T task-specific towers
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            self.towers.append(FullyConnectedLayer(
                input_dim=expert_output_dim,
                hidden_dims=tower_hidden,
                bias=[True] * len(tower_hidden),
                batch_norm=True,
                activation="relu",
                dropout=self.dropout,
            ))
        tower_output_dim = tower_hidden[-1]

        # Heads: T task-specific heads (one logit per task)
        self.heads = nn.ModuleList()
        for _ in range(num_tasks):
            self.heads.append(nn.Linear(tower_output_dim, 1))

        self._validate_fields(model_cfg)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")

    def reset_parameters(self) -> None:
        for emb in self.embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for gate in self.gates:
            for m in gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> dict[str, Tensor]:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # Embed all fields and concat
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

        # Experts: [B, K, d]
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)

        outputs: dict[str, Tensor] = {}
        for t, task_name in enumerate(self.task_names):
            # Gate: softmax over experts
            gate_logits = self.gates[t](x)
            gate_weights = F.softmax(gate_logits, dim=-1)        # [B, K]

            # Weighted sum of experts
            weighted = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [B, d]

            # Tower + head
            tower_out = self.towers[t](weighted)
            logit = self.heads[t](tower_out).squeeze(-1)
            outputs[task_name] = torch.sigmoid(logit)

        return outputs
