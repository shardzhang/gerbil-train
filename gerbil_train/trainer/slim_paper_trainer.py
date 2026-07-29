"""Trainer for the paper-implementation of SLIM (coordinate descent)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse as sp
from torch.utils.data import DataLoader

from gerbil_train.data.tfrecord_dataset import (
    BatchCollator, RatingTFRecordDataset,
    collect_tfrecord_part_files, load_field_stats, load_field_specs,
)
from gerbil_train.models.slim_paper import solve_slim_fs

__all__ = ["SLIMPaperTrainer"]


class SLIMPaperTrainer:
    """Build A, solve W, evaluate."""

    def __init__(self, data_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> None:
        slim_cfg = train_cfg.get("slim", {})
        self.data_cfg = data_cfg
        self.slim_cfg = slim_cfg
        self.beta = float(slim_cfg["beta"])
        self.lambda_ = float(slim_cfg["lambda"])
        self.max_iter = int(slim_cfg["max_iter"])
        self.tol = float(slim_cfg["tol"])
        self.top_k = int(slim_cfg["top_k"])
        self.max_samples = int(slim_cfg["max_samples"])

        self.A_shape: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    #  Build matrix A
    # ------------------------------------------------------------------
    def _build_matrix(self, split_dir: str, split_name: str = "") -> tuple[sp.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """Build sparse user-item matrix A from TFRecord.
        Returns (A, labels, rows, cols)."""
        files = collect_tfrecord_part_files(split_dir)
        if not files:
            return sp.csc_matrix(self.A_shape or (0, 0)), np.array([], dtype=float), np.array([], dtype=int), np.array([], dtype=int)

        use_files = files[:1] if self.max_samples else files
        pm_path = Path(self.data_cfg["paths"]["pos_map_txt"])
        field_specs = [
            fe for fe in load_field_specs(pm_path)
            if fe.field_name in ("user_id", "movie_id")
        ]
        stats = load_field_stats(Path(self.data_cfg["paths"]["pos_map_json"]))

        # Determine matrix shape from field vocab size
        if self.A_shape is None:
            dims = {e.field_name: int(e.dim) for e in field_specs}
            self.A_shape = (dims["user_id"], dims["movie_id"])

        dataset = RatingTFRecordDataset(
            tfrecord_files=use_files,
            field_specs=field_specs,
            field_stats=stats,
            shuffle_files=True,
            shuffle_buffer=self.slim_cfg["shuffle_buffer"],
            seed=42,
        )
        loader = DataLoader(
            dataset,
            batch_size=1024,
            num_workers=0,
            collate_fn=BatchCollator([e.field_name for e in field_specs]),
        )

        rows, cols, vals, all_labels = [], [], [], []
        n_loaded = 0

        for batch in loader:
            targets = batch["targets"]

            uid_bag = batch["feature_bags"]["user_id"]
            mid_bag = batch["feature_bags"]["movie_id"]

            o_u = uid_bag["offsets"]
            ends_u = torch.cat([o_u[1:], o_u.new_tensor([uid_bag["indices"].size(0)])])
            u_vals = uid_bag["indices"][ends_u - 1].numpy()

            o_i = mid_bag["offsets"]
            ends_i = torch.cat([o_i[1:], o_i.new_tensor([mid_bag["indices"].size(0)])])
            i_vals = mid_bag["indices"][ends_i - 1].numpy()

            for u, i, lb in zip(u_vals, i_vals, targets):
                rows.append(u)
                cols.append(i)
                vals.append(float(lb))
                all_labels.append(float(lb))
                n_loaded += 1

            if self.max_samples > 0 and n_loaded >= self.max_samples:
                break

        rows_arr = np.array(rows, dtype=int)
        cols_arr = np.array(cols, dtype=int)

        if not rows:
            return sp.csc_matrix(self.A_shape), np.array([], dtype=float), rows_arr, cols_arr

        A = sp.coo_matrix(
            (np.array(vals, dtype=np.float64), (rows_arr, cols_arr)),
            shape=self.A_shape,
        ).tocsc()

        tag = f"[{split_name}] " if split_name else ""
        print(f"  {tag}{n_loaded} samples, {A.shape[0]} users, {A.shape[1]} items, density={A.nnz / (A.shape[0] * A.shape[1]):.3%}")
        return A, np.array(all_labels, dtype=float), rows_arr, cols_arr

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    def _evaluate(
        self, 
        W: sp.csc_matrix,
        rows: np.ndarray,
        cols: np.ndarray,
        prefix: str = "",
    ) -> dict[str, float]:
        """
        推荐时，对用户 u 计算 scores = A_train[u, :] @ W
        即用用户已交互过的 item 加权聚合, 得到所有 item 的推荐得分
        """
        if len(rows) == 0 or W.nnz == 0:
            return {}

        n_users, n_items = self.A_train.shape
        S = (self.A_train @ W).toarray()

        # For each held-out pair, find rank of the item
        hits = 0
        rr_sum = 0.0
        for u, i in zip(rows, cols):
            scores = S[u].copy()
            # Exclude items already in training set
            # 把用户 u 已经交互过的 item(训练集里非零的列)的得分设为负无穷, 避免推荐已看过/买过的 item
            scores[self.A_train[u].nonzero()[1]] = -np.inf

            # 统计有多少个 item 的得分比 held-out item i 的得分高. 加 1 就是 i 的排名. 
            # 例如有 2 个 item 得分比 i 高, 则 rank = 3
            rank = int(np.sum(scores > scores[i])) + 1
            if rank <= 10:
                hits += 1

            # 累加 reciprocal rank（倒数排名），排名第 1 贡献 1/1=1，排名第 3 贡献 1/3≈0.33。
            rr_sum += 1.0 / rank

        # hits / n = HR@10（Hit Rate @ 10）
        # rr_sum / n = ARHR（Average Reciprocal Hit-Rank）
        n = len(rows)
        return {
            f"{prefix}hr@10": round(hits / n, 4),
            f"{prefix}arhr": round(rr_sum / n, 4),
        }

    # ------------------------------------------------------------------
    #  Full pipeline
    # ------------------------------------------------------------------
    def run(
        self, 
        train_dir: str, 
        val_dir: str | None = None, 
        test_dir: str | None = None,
    ) -> sp.csc_matrix:
        print("=" * 60)
        print("SLIM (paper) — coordinate descent")
        print("=" * 60)

        ms = self.max_samples
        m_str = f" (max {ms} samples)" if ms else ""
        print(f"\n[1/3] Building matrix A{m_str} …")
        self.A_train, _, _, _ = self._build_matrix(train_dir, "Train")
        print(f"  Training set done. {self.A_shape[1]} items total.")

        if val_dir:
            self.A_val, self.val_labels, self.val_rows, self.val_cols = self._build_matrix(val_dir, "Val")
        if test_dir:
            self.A_test, self.test_labels, self.test_rows, self.test_cols = self._build_matrix(test_dir, "Test")

        n_users, n_items = self.A_train.shape
        print(f"\n[2/3] Solving W …")
        print(f"  A_train: {n_users} × {n_items}, {self.A_train.nnz} interactions")
        print(f"  β={self.beta}, λ={self.lambda_}, top_k={self.top_k}")

        W = solve_slim_fs(
            self.A_train, 
            beta=self.beta, 
            lambda_=self.lambda_,
            max_iter=self.max_iter, 
            tol=self.tol, 
            top_k=self.top_k,
            verbose=True,
        )
        print(f"  W: {W.nnz} non-zeros / {n_items*n_items} ({W.nnz / (n_items * n_items):.4%})")

        if val_dir and len(self.val_rows) > 0:
            print(f"\n[3/3] Validation …")
            for k, v in self._evaluate(W, self.val_rows, self.val_cols, "val_").items():
                print(f"  {k}: {v}")

        if test_dir and len(self.test_rows) > 0:
            print(f"  Test …")
            for k, v in self._evaluate(W, self.test_rows, self.test_cols, "test_").items():
                print(f"  {k}: {v}")
        return W
