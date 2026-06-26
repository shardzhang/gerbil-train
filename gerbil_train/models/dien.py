"""Deep Interest Evolution Network (DIEN) for binary classification."""

from __future__ import annotations

from typing import Mapping, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gerbil_train.config.model_config import DIENModelConfig, FieldEntry
from gerbil_train.utils.embedding import bag_to_padded, embed_one_field, to_device
from gerbil_train.models.layers import FullyConnectedLayer
from gerbil_train.models.base_model import BaseModel

__all__ = ["DIEN"]


class GRUCell(nn.Module):
    """GRU cell exposing reset/update/new gates for AUGRU modulation."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(Tensor(3 * hidden_size))
        self.bias_hh = nn.Parameter(Tensor(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.data.dim() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (h_new, reset_gate, update_gate, new_gate)."""
        gates = (F.linear(x, self.weight_ih, self.bias_ih)
                 + F.linear(h, self.weight_hh, self.bias_hh))
        r, z, n = gates.chunk(3, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)
        h_new = (1 - z) * n + z * h
        return h_new, r, z, n


class AUGRUCell(nn.Module):
    """Attention-based GRU cell: ũ_t = a_t · u_t."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = GRUCell(input_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor, att_score: Tensor) -> Tensor:
        """Return h_new with attention-modulated update gate.

        :param x: [batch_size, input_size]
        :param h: [batch_size, hidden_size]
        :param att_score: [batch_size] — attention score for this timestep
        :return: [batch_size, hidden_size]
        """
        _, _, z, n = self.gru(x, h)        # standard GRU gates
        u_tilde = att_score.unsqueeze(-1) * z   # modulated update gate
        return (1 - u_tilde) * n + u_tilde * h


class InterestExtractorLayer(nn.Module):
    """GRU over behavior sequence: extracts interest evolution."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=False)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """:param x: [batch, seq_len, input_size]
           :param lengths: [batch]
           :return: all hidden states [batch, seq_len, hidden_size]"""
        batch, seq_len, _ = x.shape
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu,
                                                    batch_first=True,
                                                    enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out,
                                                   batch_first=True,
                                                   total_length=seq_len)
        return out


class DIEN(BaseModel):
    """DIEN (Deep Interest Evolution Network) for binary classification.

    DIEN extends DIN by:
    1. GRU (InterestExtractor): models interest evolution over behavior sequence
    2. AUGRU (Interest Evolving): attention-modulated GRU to evolve interest toward target
    3. Auxiliary loss: GRU hidden states predict next behavior item for better training
    """

    def __init__(self, model_cfg: DIENModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)
        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields
        self.item_num = model_cfg.target_size

        self.target_merge = model_cfg.target_merge
        self.behavior_fields = model_cfg.behavior_fields
        self.target_fields = model_cfg.target_fields
        reserved = set(self.behavior_fields) | set(self.target_fields)
        self.plain_field_names = [n for n in self.fields_cfg if n not in reserved]

        # 1. Target (candidate item) field embeddings
        self.target_embedding_dims: dict[str, int] = {}
        self.target_embedding_bags = nn.ModuleDict()
        for f_name in self.target_fields:
            entry = self.fields_cfg[f_name]
            self.target_embedding_dims[f_name] = int(entry.emb_size)
            key = str(entry.field_index)
            if key not in self.target_embedding_bags:
                bag = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                    include_last_offset=False,
                )
                bag.field_name = f_name
                self.target_embedding_bags[key] = bag

        # 2. Behavior field embeddings + Interest Extractor + AUGRU
        ie_cfg: dict[str, Any] = model_cfg.interest_extractor
        lau_cfg: dict[str, Any] = model_cfg.local_activation_unit
        ie_hidden = int(ie_cfg.get("hidden_size", 64))
        self.behavior_emb_dims: dict[str, int] = {}
        self.behavior_embedding_bags = nn.ModuleDict()
        self.interest_extractors = nn.ModuleDict()
        self.augru_cells = nn.ModuleDict()
        self.behavior_emb_size: int | None = None

        for bf in self.behavior_fields:
            entry = self.fields_cfg[bf]
            emb_dim = int(entry.emb_size)
            self.behavior_emb_size = emb_dim
            self.behavior_emb_dims[bf] = emb_dim
            self.behavior_embedding_bags[bf] = nn.Embedding(
                num_embeddings=int(entry.dim),
                embedding_dim=emb_dim,
                padding_idx=int(entry.dim) - 1,
            )
            self.interest_extractors[bf] = InterestExtractorLayer(
                input_size=emb_dim,
                hidden_size=ie_hidden,
                num_layers=int(ie_cfg.get("num_layers", 1)),
            )
            self.augru_cells[bf] = AUGRUCell(
                input_size=ie_hidden,
                hidden_size=ie_hidden,
            )

        # 3. Target → ie_hidden projection for attention
        # target_merge="mean" → single emb_size; "proj" → sum of all target emb_sizes
        ie_hidden = int(ie_cfg.get("hidden_size", 64))
        target_dim_for_proj = sum(self.target_embedding_dims.values()) if self.target_merge == "proj" else next(iter(self.target_embedding_dims.values()))
        if target_dim_for_proj != ie_hidden:
            self.target_proj_to_hidden = nn.Linear(target_dim_for_proj, ie_hidden)

        # 4. Attention scoring MLP (same structure as DIN's LocalActivationUnit)
        self.score_mlp = FullyConnectedLayer(
            input_dim=4 * ie_hidden,
            hidden_dims=lau_cfg.get("hidden_dims", [32, 16]),
            bias=lau_cfg.get("bias", [True, True]),
            batch_norm=lau_cfg.get("batch_norm", False),
            activation="relu",
        )
        self.score_head = nn.Linear(lau_cfg.get("hidden_dims", [32, 16])[-1], 1)

        # 4. Plain feature field embeddings
        self.plain_field_embedding_dims: dict[str, int] = {}
        self.plain_field_bags = nn.ModuleDict()
        for field_name in self.plain_field_names:
            entry = self.fields_cfg[field_name]
            if entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb"):
                self.plain_field_embedding_dims[field_name] = int(entry.emb_size)
            elif entry.field_type == 0 and entry.concat_type == "direct":
                self.plain_field_embedding_dims[field_name] = int(entry.dim)
            else:
                raise ValueError(f"Unsupported field_type {entry.field_type} "
                                 f"concat_type {entry.concat_type} for {field_name}")
            key = str(entry.field_index)
            if key not in self.plain_field_bags:
                bag = nn.EmbeddingBag(
                    num_embeddings=int(entry.dim),
                    embedding_dim=int(entry.emb_size),
                    mode="sum",
                    include_last_offset=False,
                )
                bag.field_name = field_name
                self.plain_field_bags[key] = bag

        embed_sum = sum(self.plain_field_embedding_dims.values())
        target_sum = sum(self.target_embedding_dims.values())
        behavior_sum = len(self.behavior_fields) * ie_hidden
        self.embedding_sum_dim = embed_sum + target_sum + behavior_sum

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

    def _validate_fields(self, model_cfg: DIENModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if not model_cfg.behavior_fields:
            raise ValueError("DIEN requires at least one behavior_field")
        for bf in model_cfg.behavior_fields:
            if bf not in model_cfg.embedding_fields:
                raise ValueError(f"behavior_field '{bf}' not found in embedding_fields")
        for tf in model_cfg.target_fields:
            if tf not in model_cfg.embedding_fields:
                raise ValueError(f"target_field '{tf}' not found in embedding_fields")

    def reset_parameters(self) -> None:
        for emb in self.plain_field_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.target_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.behavior_embedding_bags.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        """Forward pass. Returns sigmoid probabilities."""
        return self._forward_with_aux(feature_bags)[0]

    def forward_with_aux(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
        """Forward pass returning (sigmoid, aux_logits_dict)."""
        return self._forward_with_aux(feature_bags)

    def _forward_with_aux(self, feature_bags: Mapping[str, Mapping[str, Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
        """Internal: returns (sigmoid, {bf_name: aux_logits})."""
        first_offsets = feature_bags[self.plain_field_names[0]]["offsets"]
        batch_size = int(first_offsets.size(0))
        device = next(self.parameters()).device

        # 1. Plain field embeddings
        plain_embs: list[Tensor] = []
        for field_name in self.plain_field_names:
            entry = self.fields_cfg[field_name]
            if entry.field_type == 1 or (entry.field_type == 0 and entry.concat_type == "emb"):
                emb = embed_one_field(
                    self.plain_field_bags[str(entry.field_index)],
                    feature_bags[field_name]["indices"],
                    feature_bags[field_name]["offsets"],
                    feature_bags[field_name]["weights"],
                    device=device,
                )
            elif entry.field_type == 0 and entry.concat_type == "direct":
                emb = feature_bags[field_name]["weights"].view(-1, int(entry.dim))
            else:
                raise ValueError(f"Unsupported field_type {entry.field_type} "
                                 f"concat_type {entry.concat_type} for {field_name}")
            plain_embs.append(emb)

        # 2. Target field embeddings
        target_embs: list[Tensor] = []
        for field_name in self.target_fields:
            emb = embed_one_field(
                self.target_embedding_bags[str(self.fields_cfg[field_name].field_index)],
                feature_bags[field_name]["indices"],
                feature_bags[field_name]["offsets"],
                feature_bags[field_name]["weights"],
                device=device,
            )
            target_embs.append(emb)

        if len(target_embs) == 1:
            target_for_att = target_embs[0]
        elif self.target_merge == "mean":
            target_for_att = torch.stack(target_embs, dim=0).mean(dim=0)
        else:
            target_for_att = self.target_projection(torch.cat(target_embs, dim=-1))

        # Project target to ie_hidden for attention if needed
        if hasattr(self, "target_proj_to_hidden"):
            target_for_att = self.target_proj_to_hidden(target_for_att)

        # 3. Behavior fields: GRU → attention → AUGRU
        interest_embs: list[Tensor] = []
        aux_logits: dict[str, Tensor] = {}

        for bf in self.behavior_fields:
            entry = self.fields_cfg[bf]
            emb_dim = int(entry.emb_size)
            ie_hidden = self.interest_extractors[bf].gru.hidden_size

            indices = to_device(feature_bags[bf]["indices"].long(), device)
            offsets = to_device(feature_bags[bf]["offsets"].long(), device)

            # (a) Convert flat EmbeddingBag format to padded sequence
            padded_ids, lengths, max_seq_len = bag_to_padded(indices, offsets)

            # (b) Embedding lookup
            seq_emb = self.behavior_embedding_bags[bf](padded_ids)   # [batch, seq_len, emb_dim]

            # (c) Interest Extractor: GRU over behavior sequence
            all_hidden = self.interest_extractors[bf](seq_emb, lengths)  # [batch, seq_len, ie_hidden]

            # (d) AUGRU: compute attention scores → modulated GRU
            target_exp = target_for_att.unsqueeze(1).expand(-1, max_seq_len, -1)  # [batch, seq_len, emb_dim]
            combined = torch.cat([
                target_exp,
                all_hidden,
                target_exp - all_hidden,
                target_exp * all_hidden,
            ], dim=-1)  # [batch, seq_len, ie_hidden + 2*emb_dim + ie_hidden*emb_dim]
            combined_2d = combined.view(-1, combined.size(-1))  # [batch*seq_len, ...]
            scores_2d = self.score_head(self.score_mlp(combined_2d))  # [batch*seq_len, 1]
            att_scores = scores_2d.view(batch_size, max_seq_len)    # [batch, seq_len]
            mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            att_scores = att_scores.masked_fill(~mask, float("-inf"))
            att_weights = torch.softmax(att_scores, dim=-1)  # [batch, seq_len]

            # (e) AUGRU: evolve interest with attention-modulated update gate
            h = torch.zeros(batch_size, ie_hidden, device=device)
            for t in range(max_seq_len):
                mask_t = lengths > t
                if not mask_t.any():
                    continue
                h_prev = h * mask_t.unsqueeze(-1).float()
                h = self.augru_cells[bf](
                    all_hidden[:, t, :] * mask_t.unsqueeze(-1).float(),
                    h_prev,
                    att_weights[:, t],
                )
            interest_embs.append(h)

        # 4. Concat all → MLP → sigmoid
        input_emb = torch.cat(plain_embs + target_embs + interest_embs, dim=-1)
        if self.input_bn is not None:
            input_emb = self.input_bn(input_emb)
        hidden = self.mlp(input_emb)
        logit = self.head(hidden)
        sigmoid = torch.sigmoid(logit).squeeze(-1)

        return sigmoid, aux_logits
