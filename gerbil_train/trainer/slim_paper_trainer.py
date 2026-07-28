"""Trainer for the paper-implementation of SLIM (coordinate descent)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse as sp
from torch.utils.data import DataLoader

from gerbil_train.data.tfrecord_dataset import (
    BinaryTFRecordDataset, BatchCollator,
    collect_tfrecord_part_files, load_field_stats, load_field_specs,
)
from gerbil_train.metrics.classification import auc, average_precision, gauc, map_score, mrr_score
from gerbil_train.models.slim_paper import solve_slim_fs

__all__ = ["SLIMPaperTrainer"]


class SLIMPaperTrainer:
    """Build A, solve W, evaluate."""

    def __init__(self, data_cfg: dict[str, Any], slim_cfg: dict[str, Any]) -> None:
        self.data_cfg = data_cfg
        self.slim_cfg = slim_cfg
        self.beta = float(slim_cfg.get("beta", 1.0))
        self.lambda_ = float(slim_cfg.get("lambda", 0.5))
        self.max_iter = int(slim_cfg.get("max_iter", 20))
        self.tol = float(slim_cfg.get("tol", 1e-4))
        self.top_k = int(slim_cfg.get("top_k", 200))
        self.max_samples = int(slim_cfg.get("max_samples", 0))

        # Shared mapping: build once, reuse across splits
        self.user_to_idx: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self._mapping_frozen = False

    # ------------------------------------------------------------------
    #  Build matrix A
    # ------------------------------------------------------------------

    def _build_matrix(self, split_dir: str) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
        """Build sparse user-item matrix A. Uses & updates shared ID mapping."""
        files = collect_tfrecord_part_files(split_dir)
        if not files:
            return sp.csc_matrix((0, 0)), np.array([], dtype=float), np.array([], dtype=int)

        # Only use 1 file when sampling (enough for evaluation)
        use_files = files[:1] if self.max_samples else files

        pm_path = Path(self.data_cfg["paths"]["pos_map_txt"])
        entries = [fe for fe in load_field_specs(pm_path)
                   if fe.field_name in ("user_id", "movie_id")]
        stats = load_field_stats(Path(self.data_cfg["paths"]["pos_map_json"]))

        dataset = BinaryTFRecordDataset(
            use_files, entries, field_stats=stats, shuffle_files=False,
        )
        loader = DataLoader(
            dataset, batch_size=1024, shuffle=False, num_workers=0,
            collate_fn=BatchCollator([e.field_name for e in entries]),
        )

        rows, cols, vals = [], [], []
        user_labels, user_ids = [], []
        n_loaded = 0

        for batch in loader:
            # rating → binary label
            targets = batch["targets"]
            labels = (targets > 3).float().numpy()

            uid_bag = batch["feature_bags"]["user_id"]
            mid_bag = batch["feature_bags"]["movie_id"]

            o_u = uid_bag["offsets"]
            ends_u = torch.cat([o_u[1:], o_u.new_tensor([uid_bag["indices"].size(0)])])
            u_vals = uid_bag["indices"][ends_u - 1].numpy()

            o_i = mid_bag["offsets"]
            ends_i = torch.cat([o_i[1:], o_i.new_tensor([mid_bag["indices"].size(0)])])
            i_vals = mid_bag["indices"][ends_i - 1].numpy()

            for u, i, lb in zip(u_vals, i_vals, labels):
                if u == 0 or i == 0:
                    continue
                # Assign contiguous indices lazily
                if u not in self.user_to_idx:
                    if self._mapping_frozen:
                        continue  # skip unknown users in val/test
                    self.user_to_idx[u] = len(self.user_to_idx)
                    self.idx_to_user[self.user_to_idx[u]] = u
                if i not in self.item_to_idx:
                    if self._mapping_frozen:
                        continue  # skip unknown items in val/test
                    self.item_to_idx[i] = len(self.item_to_idx)
                    self.idx_to_item[self.item_to_idx[i]] = i

                rows.append(self.user_to_idx[u])
                cols.append(self.item_to_idx[i])
                vals.append(float(lb))
                user_labels.append(int(lb))
                user_ids.append(self.user_to_idx[u])
                n_loaded += 1

            if self.max_samples > 0 and n_loaded >= self.max_samples:
                break

        if not rows:
            return sp.csc_matrix((0, 0)), np.array([], dtype=float), np.array([], dtype=int)

        n_users = len(self.user_to_idx) if not self._mapping_frozen else len(self.idx_to_user)
        n_items = len(self.item_to_idx) if not self._mapping_frozen else len(self.idx_to_item)
        A = sp.coo_matrix(
            (np.array(vals, dtype=np.float64), (rows, cols)),
            shape=(n_users, n_items),
        ).tocsc()

        user_idx_arr = np.array([self.user_to_idx[u] for u in user_ids], dtype=int)
        print(f"  {n_loaded} samples, {A.shape[0]} users, "
              f"{A.shape[1]} items, density={A.nnz/(A.shape[0]*A.shape[1]):.3%}")
        return A, np.array(user_labels, dtype=float), user_idx_arr

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self, A: sp.csc_matrix, W: sp.csc_matrix,
        labels: np.ndarray, user_indices: np.ndarray,
        prefix: str = "",
    ) -> dict[str, float]:
        if A.shape[0] == 0 or W.nnz == 0:
            return {}

        S = (A @ W).toarray()
        rows_nz, cols_nz = A.nonzero()
        scores = S[rows_nz, cols_nz]

        t_labels = torch.from_numpy(labels)
        t_scores = torch.from_numpy(scores)
        result = {
            f"{prefix}auc": round(auc(t_labels, t_scores), 4),
            f"{prefix}ap": round(average_precision(t_labels, t_scores), 4),
        }

        unique_u = np.unique(user_indices)
        if len(unique_u) > 1:
            t_uids = torch.from_numpy(user_indices)
            valid = t_uids != 0
            if valid.any():
                result[f"{prefix}gauc"] = round(gauc(t_uids[valid], t_labels[valid], t_scores[valid]), 4)
                result[f"{prefix}map"] = round(map_score(t_uids[valid], t_labels[valid], t_scores[valid]), 4)
                result[f"{prefix}mrr"] = round(mrr_score(t_uids[valid], t_labels[valid], t_scores[valid]), 4)
        return result

    # ------------------------------------------------------------------
    #  Full pipeline
    # ------------------------------------------------------------------

    def run(
        self, train_dir: str, val_dir: str | None = None, test_dir: str | None = None,
    ) -> sp.csc_matrix:
        print("=" * 60)
        print("SLIM (paper) — coordinate descent")
        print("=" * 60)

        ms = self.max_samples
        m_str = f" (max {ms} samples)" if ms else ""
        print(f"\n[1/3] Building matrix A{m_str} …")
        self.A_train, _, _ = self._build_matrix(train_dir)
        self._mapping_frozen = True
        print(f"  Training set done. {len(self.item_to_idx)} items total.")

        if val_dir:
            self.A_val, self.val_labels, self.val_users = self._build_matrix(val_dir)
        if test_dir:
            self.A_test, self.test_labels, self.test_users = self._build_matrix(test_dir)

        n_users, n_items = self.A_train.shape
        print(f"\n[2/3] Solving W …")
        print(f"  A_train: {n_users}×{n_items}, {self.A_train.nnz} interactions")
        print(f"  β={self.beta}, λ={self.lambda_}, top_k={self.top_k}")

        W = solve_slim_fs(
            self.A_train, beta=self.beta, lambda_=self.lambda_,
            max_iter=self.max_iter, tol=self.tol, top_k=self.top_k,
            verbose=True,
        )
        print(f"  W: {W.nnz} non-zeros / {n_items*n_items} "
              f"({W.nnz/(n_items*n_items):.4%})")

        if val_dir and self.A_val is not None:
            print(f"\n[3/3] Validation …")
            for k, v in self._evaluate(self.A_val, W, self.val_labels, self.val_users, "val_").items():
                print(f"  {k}: {v}")

        if test_dir and self.A_test is not None:
            print(f"  Test …")
            for k, v in self._evaluate(self.A_test, W, self.test_labels, self.test_users, "test_").items():
                print(f"  {k}: {v}")

        return W
