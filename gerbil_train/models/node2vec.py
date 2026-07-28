"""Node2Vec for item representation learning.

Node2Vec learns item embeddings by simulating biased random walks on an
item co-occurrence graph, then applying Skip-gram (Word2Vec) on the walks.

Two key parameters control the walk behavior:
  - p (return):  likelihood of immediately returning to the previous node
  - q (in-out):  likelihood of exploring outward vs. staying local

Reference: https://arxiv.org/abs/1607.00653 (KDD 2016)
"""

from __future__ import annotations

from typing import Mapping, Any

import numpy as np
import torch
from torch import Tensor, nn

from gerbil_train.config.model_config import BaseModelConfig, FieldEntry
from gerbil_train.models.base_model import BaseModel

__all__ = ["Node2Vec", "build_cooccurrence_graph", "BiasedRandomWalker"]


def build_cooccurrence_graph(sequences: list[list[int]], vocab_size: int, window_size: int = 5) -> np.ndarray:
    """Build weighted adjacency matrix from behavior sequences.

    Edge weight = co-occurrence count within a sliding window.

    :param sequences: list of item ID sequences
    :param vocab_size: number of unique items
    :param window_size: co-occurrence window radius
    :return: [vocab_size, vocab_size] weighted adjacency matrix
    """
    adj = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for seq in sequences:
        L = len(seq)
        for t, node in enumerate(seq):
            start = max(0, t - window_size)
            end = min(L, t + window_size + 1)
            for c in range(start, end):
                if c == t:
                    continue
                adj[node, seq[c]] += 1.0
    return adj


def _alias_setup(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build alias tables for O(1) weighted sampling.

    :param probs: [K] probability distribution
    :return: (alias, prob) tables
    """
    K = len(probs)
    alias = np.zeros(K, dtype=np.int64)
    prob = np.zeros(K, dtype=np.float32)
    scaled = probs * K
    small, large = [], []
    for i, s in enumerate(scaled):
        (small if s < 1.0 else large).append(i)
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] + scaled[s] - 1.0
        (small if scaled[l] < 1.0 else large).append(l)
    for i in small + large:
        prob[i] = 1.0
    return alias, prob


class _BiasedRandomWalker:
    """Biased random walk on a weighted graph (Node2Vec)."""

    def __init__(self, adj: np.ndarray, p: float, q: float, walk_length: int):
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_nodes = adj.shape[0]

        # Transition probabilities (normalized)
        row_sum = adj.sum(axis=1, keepdims=True).clip(min=1e-8)
        self.trans_probs = adj / row_sum

        # Alias tables for first step (unbiased)
        self.first_alias: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for v in range(self.num_nodes):
            neighbors = np.where(self.trans_probs[v] > 0)[0]
            if len(neighbors) > 0:
                probs = self.trans_probs[v, neighbors]
                self.first_alias[v] = (neighbors, _alias_setup(probs))

        # Precompute alias tables for all (v, t) pairs lazily
        self.alias_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _biased_probs(self, v: int, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Get alias tables for biased step from v (came from t)."""
        key = (v, t)
        if key in self.alias_cache:
            return self.alias_cache[key]
        neighbors = np.where(self.trans_probs[v] > 0)[0]
        if len(neighbors) == 0:
            return np.array([], dtype=np.int64), (np.array([], dtype=np.float32), np.array([], dtype=np.int64))
        unnormalized = np.ones(len(neighbors), dtype=np.float32)
        t_neighbors = set(np.where(self.trans_probs[t] > 0)[0]) if t >= 0 else set()
        for i, x in enumerate(neighbors):
            if x == t:
                unnormalized[i] = 1.0 / self.p
            elif x in t_neighbors:
                unnormalized[i] = 1.0
            else:
                unnormalized[i] = 1.0 / self.q
        probs = unnormalized / unnormalized.sum()
        alias_tables = _alias_setup(probs)
        self.alias_cache[key] = (neighbors, alias_tables)
        return neighbors, alias_tables

    def _alias_sample(self, alias_tables: tuple[np.ndarray, np.ndarray], rng: np.random.Generator) -> int:
        """Sample from alias tables."""
        prob, alias = alias_tables
        i = int(rng.integers(len(prob)))
        return i if rng.random() < prob[i] else int(alias[i])

    def walk(self, start: int, rng: np.random.Generator) -> list[int]:
        """Generate one random walk starting from node `start`.

        :param start: starting node ID
        :param rng: numpy random Generator
        :return: list of node IDs in the walk
        """
        walk = [start]
        if start not in self.first_alias:
            return walk
        # First step (from v = start, t = -1, use unbiased)
        neighbors, alias_tables = self.first_alias[start]
        if len(neighbors) == 0:
            return walk
        idx = self._alias_sample(alias_tables, rng)
        walk.append(int(neighbors[idx]))
        # Subsequent steps (biased)
        for _ in range(self.walk_length - 2):
            v = walk[-1]
            t = walk[-2]
            neighbors, alias_tables = self._biased_probs(v, t)
            if len(neighbors) == 0:
                break
            idx = self._alias_sample(alias_tables, rng)
            walk.append(int(neighbors[idx]))
        return walk


class Node2Vec(BaseModel):
    """Node2Vec: biased random walks + Skip-gram item embeddings."""

    def __init__(self, model_cfg: BaseModelConfig) -> None:
        super().__init__()

        self.embedding_fields: Mapping[str, FieldEntry] = model_cfg.embedding_fields

        n2v_cfg: dict[str, Any] = model_cfg.mlp
        self.item_field = str(n2v_cfg["item_field"])
        self.emb_size = int(self.embedding_fields[self.item_field].emb_size)
        self.vocab_size = int(self.embedding_fields[self.item_field].dim)

        # Same Skip-gram architecture as Word2Vec
        self.target_embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.context_embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self._validate_fields(model_cfg)
        self.reset_parameters()

    def _validate_fields(self, model_cfg: BaseModelConfig) -> None:
        if not model_cfg.embedding_fields:
            raise ValueError("embedding_fields must be a non-empty mapping")
        if "item_field" not in model_cfg.mlp:
            raise ValueError("Node2Vec config must specify item_field")

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.target_embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)

    def forward(self, target_ids: Tensor, context_ids: Tensor) -> Tensor:
        target_emb = self.target_embedding(target_ids)
        context_emb = self.context_embedding(context_ids)
        return (target_emb * context_emb).sum(dim=-1)

    def embed(self, item_ids: Tensor) -> Tensor:
        return (self.target_embedding(item_ids) + self.context_embedding(item_ids)) / 2
