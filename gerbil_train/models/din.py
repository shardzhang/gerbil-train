"""Deep Interest Network (DIN) for binary classification."""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import DINModelConfig
from gerbil_train.utils.embedding import bag_to_padded, embed_one_field, to_device
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.config.model_config import FieldEntry

__all__ = ["DIN"]


class DIN(nn.Module):
    """DIN (Deep Interest Network) for binary classification with behavior-sequence attention."""

    def __init__(self, model_cfg: DINModelConfig) -> None:
        super().__init__()

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        if not self.fields_cfg:
            raise ValueError("embedding_fields must be a non-empty mapping")
                             
        behavior_fields = model_cfg.behavior_fields
        target_fields = model_cfg.target_fields
        self._validate_fields(model_cfg)

        self.item_num = model_cfg.target_size
        # Whether to use softmax attention
        self.softmax_attn = model_cfg.softmax_attn
        # Target merge strategy: "mean" or "proj"
        self.target_merge = model_cfg.target_merge
        self.behavior_fields = behavior_fields
        self.target_fields = target_fields
        reserved = set(behavior_fields) | set(target_fields)
        self.field_names = [n for n in self.fields_cfg if n not in reserved]

        # Target (candidate item) field embeddings
        self.target_embedding_dims: dict[str, int] = {}
        self.target_embeddings = nn.ModuleDict()
        for f_name in target_fields:
            entry = self.fields_cfg[f_name]
            self.target_embedding_dims[f_name] = int(entry.emb_size)
            self.target_embeddings[str(entry.field_index)] = nn.EmbeddingBag(
                num_embeddings=int(entry.dim),
                embedding_dim=int(entry.emb_size),
                mode="sum", 
                include_last_offset=False,
            )

        # Behavior field (user history sequence) embeddings
        self.behavior_emb_dims: dict[str, int] = {}
        self.behavior_embeddings = nn.ModuleDict()
        self.attention_units = nn.ModuleDict()
        for bf in behavior_fields:
            entry = self.fields_cfg[bf]
            # print(f"[DEBUG] {bf}: dim={entry.dim}, emb_size={entry.emb_size}")
            self.behavior_emb_dims[bf] = int(entry.emb_size)
            self.behavior_embeddings[bf] = EmbeddingLayer(
                item_num=int(entry.dim), 
                embedding_dim=int(entry.emb_size),
            )
            lau: dict[str, Any] = model_cfg.local_activation_unit
            # 各个行为序列特征词表独占，不同享
            self.attention_units[bf] = LocalActivationUnit(
                hidden_dims=lau.get("hidden_dims", [80, 40]),
                bias=lau.get("bias", [True, True]),
                embedding_dim=int(entry.emb_size),
                batch_norm=lau.get("batch_norm", False),
            )

        # When target_merge="proj": concat targets → Linear project to behavior emb_dim
        if self.target_merge == "proj" and self.target_fields and self.behavior_fields:
            total_target_dim = sum(self.target_embedding_dims.values())
            proj_dim = int(self.fields_cfg[self.behavior_fields[0]].emb_size)
            self.target_projection = nn.Linear(total_target_dim, proj_dim)

        # Plain feature (non-behavior, non-target) field embeddings
        self.field_embedding_dims: dict[str, int] = {}
        self.field_embeddings = nn.ModuleDict()
        for field_name in self.field_names:
            entry = self.fields_cfg[field_name]
            self.field_embedding_dims[field_name] = int(entry.emb_size)
            self.field_embeddings[field_name] = nn.EmbeddingBag(
                num_embeddings=int(entry.dim),
                embedding_dim=int(entry.emb_size),
                mode="sum", 
                include_last_offset=False,
            )

        self.embedding_sum_dim = sum(self.field_embedding_dims.values()) + sum(self.behavior_emb_dims.values()) + sum(self.target_embedding_dims.values())
        mlp_cfg: dict[str, Any] = model_cfg.mlp
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


    @staticmethod
    def _validate_fields(model_cfg: DINModelConfig) -> None:
        """Validate the model configuration."""
        fields_cfg = model_cfg.embedding_fields
        if not fields_cfg:
            raise ValueError("embedding_fields must be a non-empty mapping")
        for bf in model_cfg.behavior_fields:
            if bf not in fields_cfg:
                raise ValueError(f"behavior_field '{bf}' not found in embedding_fields")
        for tf in model_cfg.target_fields:
            if tf not in fields_cfg:
                raise ValueError(f"target_field '{tf}' not found in embedding_fields")

    def reset_parameters(self) -> None:
        # Even without explicit reset, PyTorch nn.EmbeddingBag defaults to Xavier uniform
        for emb in self.field_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.target_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)


    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass of the DIN model."""
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
        for field_name in self.target_fields:
            bag = feature_bags[field_name]
            target_emb = embed_one_field(
                self.target_embeddings[str(self.fields_cfg[field_name].field_index)],
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
            # [batch, max_seq_len, emb_dim]
            seq_emb = self.behavior_embeddings[bf](padded_ids)
            # [batch, max_seq_len, 1]
            target_exp = target_for_attention.unsqueeze(1).expand(-1, max_seq_len, -1)
            # [batch, max_seq_len]
            scores = self.attention_units[bf](seq_emb, target_exp).squeeze(-1)

            # [batch, max_seq_len] — mask out padding and invalid item IDs
            mask = (torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)) & (padded_ids < self.item_num)
            if self.softmax_attn:
                scores = scores.masked_fill(~mask, float("-inf"))
                # [batch, max_seq_len]
                attn = torch.softmax(scores, dim=-1)
            else:
                scores = scores.masked_fill(~mask, 0.0)
                attn = scores
            
            # [batch, emb_dim]
            interest = (attn.unsqueeze(-1) * seq_emb).sum(dim=1)
            interest_embs.append(interest)

        # [batch, embedding_sum_dim]
        input_emb = torch.cat(field_embs + target_embs + interest_embs, dim=-1)
        if self.input_bn is not None:
            input_emb = self.input_bn(input_emb)
        
        # [batch, final_hidden_dim]
        hidden = self.mlp(input_emb)
        
        # [batch, 1]
        logit = self.head(hidden)
        
        # [batch, ]
        return torch.sigmoid(logit).squeeze(-1)


class LocalActivationUnit(nn.Module):
    """Local Activation Unit (LAU) for DIN model."""
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
        """Forward pass of the LAU.

        :param user_behavior: (batch_size, f_num, embed_dim)
        :param queries: (batch_size, f_num, embed_dim)
        :return: (batch_size, f_num, 1)
        """
        batch, seq_len, emb_dim = user_behavior.shape
        # [batch, seq_len, 4*emb_dim]
        concat = torch.cat([
                queries, 
                user_behavior,
                queries - user_behavior, 
                queries * user_behavior,
            ], 
            dim=-1
        )  
        concat_2d = concat.view(-1, 4 * emb_dim)   # [batch*seq_len, 4*emb_dim]
        output = self.fc2(self.fc1(concat_2d))     # [batch*seq_len, 1]
        return output.view(batch, seq_len, 1)      # [batch, seq_len, 1]

class EmbeddingLayer(nn.Module):
    """Embedding layer for DIN model."""
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
        """Forward pass of the embedding layer.
        
        :param x: (batch_size, f_num)
        :return: (batch_size, f_num, embed_dim)
        """
        return self.embed(x)
