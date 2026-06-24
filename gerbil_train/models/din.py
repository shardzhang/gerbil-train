"""Deep Interest Network (DIN) for binary classification."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import GwENModelConfig
from gerbil_train.utils.embedding import bag_to_padded, embed_one_field, to_device
from gerbil_train.utils.nn import FullyConnectedLayer

__all__ = ["DIN"]


class DIN(nn.Module):
    """DIN (Deep Interest Network) for binary classification with behavior-sequence attention."""

    def __init__(self, config: GwENModelConfig, behavior_fields: list[str], target_fields: list[str], softmax_attn=False, target_merge: str = "mean") -> None:
        """
        :param softmax_attn: If True, apply softmax normalization to attention scores;
            otherwise use raw scores (original DIN style).
        :param target_merge: How to merge multiple target fields into an attention query.
            "mean" — average all target embeddings (requires same emb_dim).
            "proj" — concat then linearly project to behavior embedding dim.
        """
        super().__init__()
        
        fields_cfg = config.embedding_fields
        if not fields_cfg:
            raise ValueError("embedding_fields must be a non-empty mapping")
        for bf in behavior_fields:
            if bf not in fields_cfg:
                raise ValueError(f"behavior_field '{bf}' not found in embedding_fields")
        for tf in target_fields:
            if tf not in fields_cfg:
                raise ValueError(f"target_field '{tf}' not found in embedding_fields")

        self.item_num = config.target_size
        self.softmax_attn = softmax_attn
        self.behavior_fields = behavior_fields
        self.target_fields = target_fields
        self.target_merge = target_merge
        reserved = set(behavior_fields) | set(target_fields)
        self.field_names = [n for n in fields_cfg if n not in reserved]

        # Target (candidate item) field embeddings
        self.target_embedding_dims: dict[str, int] = {}
        self.target_embeddings = nn.ModuleDict()
        for tf in target_fields:
            entry = fields_cfg[tf]
            self.target_embedding_dims[tf] = int(entry.emb_dim)
            self.target_embeddings[tf] = nn.EmbeddingBag(
                num_embeddings=int(entry.vocab_size),
                embedding_dim=int(entry.emb_dim),
                mode="sum", 
                include_last_offset=False,
            )

        # When target_merge="proj": concat targets → Linear project to behavior emb_dim
        if target_merge == "proj" and target_fields and behavior_fields:
            total_target_dim = sum(self.target_embedding_dims.values())
            proj_dim = self.behavior_emb_dims[behavior_fields[0]]
            self.target_projection = nn.Linear(total_target_dim, proj_dim)

        # Behavior field (user history sequence) embeddings
        self.behavior_emb_dims: dict[str, int] = {}
        self.behavior_embeddings = nn.ModuleDict()
        self.attention_units = nn.ModuleDict()
        for bf in behavior_fields:
            entry = fields_cfg[bf]
            self.behavior_emb_dims[bf] = int(entry.emb_dim)
            self.behavior_embeddings[bf] = EmbeddingLayer(
                item_num=int(entry.vocab_size), 
                embedding_dim=int(entry.emb_dim),
            )
            self.attention_units[bf] = LocalActivationUnit(
                hidden_dims=[80, 40], 
                bias=[True, True],
                embedding_dim=int(entry.emb_dim), 
                batch_norm=False,
            )

        # Plain feature (non-behavior, non-target) field embeddings
        self.field_embedding_dims: dict[str, int] = {}
        self.field_embeddings = nn.ModuleDict()
        for field_name in self.field_names:
            entry = fields_cfg[field_name]
            self.field_embedding_dims[field_name] = int(entry.emb_dim)
            self.field_embeddings[field_name] = nn.EmbeddingBag(
                num_embeddings=int(entry.vocab_size),
                embedding_dim=int(entry.emb_dim),
                mode="sum", 
                include_last_offset=False,
            )

        self.embedding_sum_dim = sum(self.field_embedding_dims.values()) + sum(self.behavior_emb_dims.values()) + sum(self.target_embedding_dims.values())
        mlp_cfg = config.mlp
        hidden_dims = list(mlp_cfg.get("hidden_dims", [256, 128]))
        self.input_bn = nn.BatchNorm1d(self.embedding_sum_dim) if mlp_cfg.get("input_batch_norm", False) else None
        self.mlp = FullyConnectedLayer(
            input_dim=self.embedding_sum_dim, 
            hidden_dims=hidden_dims,
            bias=[True] * len(hidden_dims),
            batch_norm=bool(mlp_cfg.get("batch_norm", True)),
            activation=str(mlp_cfg.get("activation", "relu")),
            dropout=float(mlp_cfg.get("dropout", 0.0)),
        )
        final_hidden_dim = hidden_dims[-1] if hidden_dims else self.embedding_sum_dim
        self.head = nn.Linear(final_hidden_dim, 1)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # Even without explicit reset, PyTorch nn.EmbeddingBag defaults to Xavier uniform
        for emb in self.field_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.target_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)


    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        first_offsets = feature_bags[self.field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # Embed plain feature fields
        field_embs: list[Tensor] = []
        for fn in self.field_names:
            bag = feature_bags[fn]
            field_emb = embed_one_field(
                self.field_embeddings[fn], 
                bag["indices"], 
                bag["offsets"], 
                bag["weights"], 
                device=device
            )
            field_embs.append(field_emb)

        # Embed target (candidate item) fields
        target_embs: list[Tensor] = []
        for tf in self.target_fields:
            bag = feature_bags[tf]
            target_emb = embed_one_field(
                self.target_embeddings[tf],
                bag["indices"],
                bag["offsets"],
                bag["weights"],
                device=device,
            )
            target_embs.append(target_emb)

        # Merge multiple target embeddings into a single attention query
        if len(target_embs) == 1:
            target_for_attention = target_embs[0]
        elif self.target_merge == "mean":
            target_for_attention = torch.stack(target_embs, dim=0).mean(dim=0)
        else:
            target_for_attention = self.target_projection(torch.cat(target_embs, dim=-1))

        interest_embs: list[Tensor] = []
        for bf in self.behavior_fields:
            indices = to_device(feature_bags[bf]["indices"].long(), device)
            offsets = to_device(feature_bags[bf]["offsets"].long(), device)
            
            # Convert variable-length EmbeddingBag to padded sequence for attention
            padded_ids, lengths, max_seq_len = bag_to_padded(indices, offsets)
            seq_emb = self.behavior_embeddings[bf](padded_ids)
            target_exp = target_for_attention.unsqueeze(1).expand(-1, max_seq_len, -1)
            scores = self.attention_units[bf](seq_emb, target_exp).squeeze(-1)

            # [batch, max_seq_len] — mask out padding and invalid item IDs
            mask = (torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)) & (padded_ids < self.item_num)
            if self.softmax_attn:
                scores = scores.masked_fill(~mask, float("-inf"))
                attn = torch.softmax(scores, dim=-1)
            else:
                scores = scores.masked_fill(~mask, 0.0)
                attn = scores
            
            # (batch, emb_dim)
            interest = (attn.unsqueeze(-1) * seq_emb).sum(dim=1)
            interest_embs.append(interest)

        input_emb = torch.cat(field_embs + target_embs + interest_embs, dim=-1)
        if self.input_bn is not None:
            input_emb = self.input_bn(input_emb)
        hidden = self.mlp(input_emb)
        logit = self.head(hidden)
        # (batch_size, 1) -> (batch_size,)
        return torch.sigmoid(logit).squeeze(-1)


class LocalActivationUnit(nn.Module):
    def __init__(
        self,
        hidden_dims: list[int] = [80, 40],
        bias: list[bool] = [True, True],
        embedding_dim: int = 4,
        batch_norm: bool = False
    ) -> None:
        super(LocalActivationUnit, self).__init__()

        self.fc1 = FullyConnectedLayer(
            input_dim=4 * embedding_dim,
            hidden_dims=hidden_dims,
            bias=bias,
            batch_norm=batch_norm,
            activation="relu",
        )

        self.fc2 = FullyConnectedLayer(
            input_dim=hidden_dims[-1],
            hidden_dims=[1],
            bias=[True],
            batch_norm=batch_norm,
            activation="relu",
        )

    def forward(self, user_behavior: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        :param user_behavior: (batch_size, f_num, embed_dim)
        :param queries: (batch_size, f_num, embed_dim)
        :return: (batch_size, f_num, 1)
        """
        attention_output = self.fc2(self.fc1(
            torch.cat([
                queries,
                user_behavior,
                queries - user_behavior,
                queries * user_behavior,
            ], dim=-1)
        ))
        return attention_output


class EmbeddingLayer(nn.Module):
    def __init__(self, item_num: int, embedding_dim: int):
        super(EmbeddingLayer, self).__init__()
        
        self.embed = nn.Embedding(item_num + 1, embedding_dim, padding_idx=item_num)
        nn.init.normal_(self.embed.weight, 0., 0.0001)
        if self.embed.padding_idx is not None:
            with torch.no_grad():
                # Regularization is handled by the optimizer's weight_decay globally.
                # Only the padding row must be explicitly zeroed so that padded
                # positions do not contribute to attention scores.
                self.embed.weight[self.embed.padding_idx].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.embed(x)
