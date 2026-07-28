"""Word2Vec (Skip-gram with Negative Sampling) for item representation learning.

Learns item embeddings from behavior sequences: items appearing in similar
contexts will have similar embeddings.

Architecture:
  target_embedding(vocab → d) + context_embedding(vocab → d)
  score = ⟨target_emb[target_id], context_emb[context_id]⟩

Reference: https://arxiv.org/abs/1310.4546 (NIPS 2013)
"""

from __future__ import annotations

from typing import Mapping, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.utils.embedding import embed_one_field
from gerbil_train.models.base_model import BaseModel

__all__ = ["Word2Vec"]


class Word2Vec(BaseModel):
    """Skip-gram Word2Vec with item embeddings."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()
        self._validate_fields(model_cfg)

        self.fields_cfg: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        # Word2Vec config
        w2v_cfg: dict[str, Any] = model_cfg.mlp
        self.item_field = str(w2v_cfg["item_field"])
        self.emb_size = int(self.fields_cfg[self.item_field].emb_size)
        self.vocab_size = int(self.fields_cfg[self.item_field].dim)

        # Target embedding (input → hidden)
        self.target_embedding = nn.Embedding(self.vocab_size, self.emb_size)

        # Context embedding (hidden → output)
        self.context_embedding = nn.Embedding(self.vocab_size, self.emb_size)

        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if "item_field" not in model_cfg.mlp:
            raise ValueError("Word2Vec config must specify item_field")

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.target_embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)

    def forward(self, target_ids: Tensor, context_ids: Tensor) -> Tensor:
        """Score for (target, context) pairs.

        :param target_ids: [N] target item IDs
        :param context_ids: [N] context item IDs
        :return: [N] logits = ⟨target_emb, context_emb⟩
        """
        target_emb = self.target_embedding(target_ids)
        context_emb = self.context_embedding(context_ids)
        return (target_emb * context_emb).sum(dim=-1)

    def embed(self, item_ids: Tensor) -> Tensor:
        """Get final item embedding (target + context averaged).

        :param item_ids: [N] item IDs
        :return: [N, d] item embeddings
        """
        return (self.target_embedding(item_ids) + self.context_embedding(item_ids)) / 2
