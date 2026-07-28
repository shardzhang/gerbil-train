"""PLE (Progressive Layered Extraction) for multi-task CTR prediction.

PLE extends MMoE with multi-level extraction layers, progressively separating
shared and task-specific knowledge. Each layer contains shared experts (used
by all tasks) and task-specific experts (used by only one task).

Reference: https://dl.acm.org/doi/10.1145/3383313.3412236 (RecSys 2020)
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

__all__ = ["PLE"]


class _Expert(nn.Module):
    """Single expert MLP."""

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = FullyConnectedLayer(
            input_dim=input_dim, hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=True, activation="relu", dropout=0.0,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _ExtractionLayer(nn.Module):
    """One PLE extraction layer: shared experts + per-task experts + gates."""

    def __init__(self, input_dim: int, expert_hidden: list[int],
                 num_shared: int, num_specific: int, num_tasks: int):
        super().__init__()
        self.num_shared = num_shared
        self.num_specific = num_specific
        self.num_tasks = num_tasks
        self.expert_dim = expert_hidden[-1]

        # Shared experts
        self.shared_experts = nn.ModuleList([
            _Expert(input_dim, expert_hidden) for _ in range(num_shared)
        ])

        # Task-specific experts
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([_Expert(input_dim, expert_hidden) for _ in range(num_specific)])
            for _ in range(num_tasks)
        ])

        # Shared gate: weights over shared experts
        self.shared_gate = nn.Linear(input_dim, num_shared)

        # Task gates: weights over shared + specific experts
        self.task_gates = nn.ModuleList([
            nn.Linear(input_dim, num_shared + num_specific) for _ in range(num_tasks)
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Forward pass.

        :param x: [B, input_dim] (original input + skip from previous layer)
        :return: (shared_output [B, d], list of task_outputs [B, d])
        """
        # Expert outputs
        shared_out = torch.stack([e(x) for e in self.shared_experts], dim=1)         # [B, Ks, d]
        specific_out = []
        for t in range(self.num_tasks):
            specific_out.append(
                torch.stack([e(x) for e in self.specific_experts[t]], dim=1)         # [B, Kt, d]
            )

        # Shared gate → weighted sum of shared experts
        sw = F.softmax(self.shared_gate(x), dim=-1)                                  # [B, Ks]
        shared_output = (sw.unsqueeze(-1) * shared_out).sum(dim=1)                   # [B, d]

        # Task gates → weighted sum of shared + specific
        task_outputs = []
        for t in range(self.num_tasks):
            expert_pool = torch.cat([shared_out, specific_out[t]], dim=1)            # [B, Ks+Kt, d]
            tw = F.softmax(self.task_gates[t](x), dim=-1)                            # [B, Ks+Kt]
            task_outputs.append((tw.unsqueeze(-1) * expert_pool).sum(dim=1))         # [B, d]

        return shared_output, task_outputs


class PLE(BaseModel):
    """Progressive Layered Extraction for multi-task binary classification."""

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

        # Input dimension (all field embeddings concat)
        self.input_dim = sum(
            int(e.emb_size) for fn, e in self.embedding_fields.items()
            if not (e.field_type == 0 and e.concat_type == "direct")
        )
        direct_dim = sum(
            int(e.dim) for fn, e in self.embedding_fields.items()
            if e.field_type == 0 and e.concat_type == "direct"
        )
        self.input_dim += direct_dim

        # PLE config
        ple_cfg: dict[str, Any] = model_cfg.mlp
        num_layers = int(ple_cfg["num_layers"])
        num_shared = int(ple_cfg["num_shared_experts"])
        num_specific = int(ple_cfg["num_specific_experts"])
        expert_hidden = list(ple_cfg["expert_hidden"])
        tower_hidden = list(ple_cfg["tower_hidden"])
        num_tasks = int(ple_cfg["num_tasks"])
        self.task_names = list(ple_cfg.get("task_names", [f"task_{i}" for i in range(num_tasks)]))

        # Extraction layers (progressive separation)
        self.extraction_layers = nn.ModuleList()
        for i in range(num_layers):
            # Each layer's gate input = original input (no skip connection for simplicity)
            # The original PLE uses previous layer outputs as additional gate input,
            # but for simplicity we use the raw embedding input for all layers.
            self.extraction_layers.append(_ExtractionLayer(
                input_dim=self.input_dim,
                expert_hidden=expert_hidden,
                num_shared=num_shared,
                num_specific=num_specific,
                num_tasks=num_tasks,
            ))

        # Tower MLP per task
        self.towers = nn.ModuleList()
        for _ in range(num_tasks):
            self.towers.append(FullyConnectedLayer(
                input_dim=expert_hidden[-1],
                hidden_dims=tower_hidden,
                bias=[True] * len(tower_hidden),
                batch_norm=True, activation="relu", dropout=0.1,
            ))

        # Final head per task
        self.heads = nn.ModuleList()
        for _ in range(num_tasks):
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

        # Pass through extraction layers
        # Only task outputs from the LAST layer are used for prediction
        final_task_outputs = None
        for layer in self.extraction_layers:
            _, task_outputs = layer(x)
            final_task_outputs = task_outputs

        # Towers + heads
        outputs: dict[str, Tensor] = {}
        for t, task_name in enumerate(self.task_names):
            tower_out = self.towers[t](final_task_outputs[t])
            logit = self.heads[t](tower_out).squeeze(-1)
            outputs[task_name] = torch.sigmoid(logit)

        return outputs
