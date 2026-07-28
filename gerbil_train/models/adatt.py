"""AdaTT (Adaptive Task-to-Task Fusion Network) for multi-task CTR.

AdaTT uses task-specific experts and progressive task-to-task fusion layers.
At each layer, each task adaptively fuses representations from other tasks
via learned gates.

    h_t^{l+1} = FusionLayer( h_1^l, h_2^l, ..., h_T^l, gate_t )

Reference: https://dl.acm.org/doi/10.1145/3485447.3512040 (WWW 2022)
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

__all__ = ["AdaTT"]


class _TaskExpert(nn.Module):
    """Initial task-specific expert network."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _FusionLayer(nn.Module):
    """One AdaTT fusion layer: cross-task gated fusion."""

    def __init__(self, task_dim: int, num_tasks: int, hidden_dim: int):
        super().__init__()
        self.num_tasks = num_tasks
        # Per-task fusion gates (softmax over all tasks)
        self.fusion_gates = nn.ModuleList()
        for _ in range(num_tasks):
            self.fusion_gates.append(nn.Linear(task_dim, num_tasks))

        # Per-task transformation after fusion
        self.transform = nn.ModuleList()
        for _ in range(num_tasks):
            self.transform.append(nn.Sequential(
                nn.Linear(task_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, task_dim),
            ))

    def forward(self, task_states: list[Tensor]) -> list[Tensor]:
        """Fuse task states.

        :param task_states: list of [B, d] for T tasks
        :return: updated list of [B, d]
        """
        stacked = torch.stack(task_states, dim=1)         # [B, T, d]
        new_states: list[Tensor] = []
        for t in range(self.num_tasks):
            # Gate: how much to attend to each task
            gate = F.softmax(self.fusion_gates[t](task_states[t]), dim=-1)  # [B, T]
            # Weighted sum over all tasks
            fused = (gate.unsqueeze(-1) * stacked).sum(dim=1)               # [B, d]
            # Residual transformation
            delta = self.transform[t](fused)
            new_states.append(task_states[t] + delta)
        return new_states


class AdaTT(BaseModel):
    """Adaptive Task-to-Task Fusion Network for multi-task CTR."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

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

        # AdaTT config
        att_cfg: dict[str, Any] = model_cfg.mlp
        num_layers = int(att_cfg.get("num_layers", 2))
        task_dim = int(att_cfg.get("task_dim", 64))
        expert_hidden = int(att_cfg.get("expert_hidden", 128))
        tower_hidden = list(att_cfg["tower_hidden"])
        num_tasks = int(att_cfg["num_tasks"])
        self.task_names = list(att_cfg.get("task_names", [f"task_{i}" for i in range(num_tasks)]))

        # Task-specific initial experts
        self.task_experts = nn.ModuleList()
        for _ in range(num_tasks):
            self.task_experts.append(_TaskExpert(self.input_dim, expert_hidden))

        # Project to task_dim if needed
        self.task_proj = nn.ModuleList()
        for _ in range(num_tasks):
            if expert_hidden != task_dim:
                self.task_proj.append(nn.Linear(expert_hidden, task_dim))
            else:
                self.task_proj.append(nn.Identity())

        # Fusion layers
        self.fusion_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fusion_layers.append(_FusionLayer(task_dim, num_tasks, task_dim))

        # Towers + heads
        self.towers = nn.ModuleList()
        self.heads = nn.ModuleList()
        for _ in range(num_tasks):
            self.towers.append(FullyConnectedLayer(
                input_dim=task_dim, hidden_dims=tower_hidden,
                bias=[True] * len(tower_hidden),
                batch_norm=True, activation="relu", dropout=0.1,
            ))
            self.heads.append(nn.Linear(tower_hidden[-1], 1))

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

        # Task-specific initial experts → task states
        task_states: list[Tensor] = []
        for t in range(len(self.task_names)):
            h = self.task_experts[t](x)
            task_states.append(self.task_proj[t](h))

        # Progressive fusion layers
        for layer in self.fusion_layers:
            task_states = layer(task_states)

        # Towers + heads
        outputs: dict[str, Tensor] = {}
        for t, task_name in enumerate(self.task_names):
            tower_out = self.towers[t](task_states[t])
            logit = self.heads[t](tower_out).squeeze(-1)
            outputs[task_name] = torch.sigmoid(logit)

        return outputs
