"""PEPNet (Parameter Efficient Personalized Network) for multi-task CTR.

PEPNet personalizes the base network via two sub-networks:
  - EPNet: generates element-wise scale & bias for input embeddings
  - MMoE backbone: shared experts + task gates + towers

Reference: https://dl.acm.org/doi/10.1145/3543507.3583206 (WWW 2023)
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

__all__ = ["PEPNet"]


class _EPNet(nn.Module):
    """Embedding Personalized Network.

    Generates element-wise scale and bias for the input embedding
    conditioned on a domain/user embedding.
    """

    def __init__(self, input_dim: int, cond_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.generator = FullyConnectedLayer(
            input_dim=cond_dim, hidden_dims=hidden_dims + [input_dim * 2],
            bias=[True] * (len(hidden_dims) + 1),
            batch_norm=False, activation="relu", dropout=0.0,
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply personalized scale & bias.

        :param x:  [B, input_dim] base embedding
        :param cond: [B, cond_dim] condition embedding
        :return: [B, input_dim] personalized embedding
        """
        params = self.generator(cond)                                # [B, 2*input_dim]
        scale, bias = torch.chunk(params, 2, dim=-1)
        return x * torch.tanh(scale) + bias


class PEPNet(BaseModel):
    """Parameter Efficient Personalized Network."""

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

        # PEPNet config
        pep_cfg: dict[str, Any] = model_cfg.mlp
        self.domain_field = str(pep_cfg.get("domain_field", self.field_names[0]))
        epnet_hidden = list(pep_cfg.get("epnet_hidden", [64, 32]))
        num_experts = int(pep_cfg["num_experts"])
        expert_hidden = list(pep_cfg["expert_hidden"])
        tower_hidden = list(pep_cfg["tower_hidden"])
        num_tasks = int(pep_cfg["num_tasks"])
        self.task_names = list(pep_cfg.get("task_names", [f"task_{i}" for i in range(num_tasks)]))
        cond_dim = int(self.embedding_fields[self.domain_field].emb_size)
        self.dropout = float(pep_cfg.get("dropout", 0.1))

        # EPNet: generates scale & bias for the input
        self.epnet = _EPNet(self.input_dim, cond_dim, epnet_hidden)

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
            gate_layers: list[nn.Module] = [
                nn.Linear(self.input_dim, num_experts),
            ]
            self.gates.append(nn.Sequential(*gate_layers))

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
        domain_emb = None
        for field_name, entry in self.embedding_fields.items():
            if entry.field_type == 0 and entry.concat_type == "direct":
                feats = feature_bags[field_name]["weights"].view(-1, int(entry.dim))
            else:
                feats = embed_one_field(
                    self.embedding_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                )
            if field_name == self.domain_field:
                domain_emb = feats
            emb_list.append(feats)
        x = torch.cat(emb_list, dim=-1)

        # EPNet: personalize embeddings with domain condition
        if domain_emb is not None:
            x = self.epnet(x, domain_emb)

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
