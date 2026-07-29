"""Coordinate-descent SLIM solver with fsSLIM (feature selection).

Implements:
    Ning & Karypis, "SLIM: Sparse Linear Methods for Top-N Recommender Systems", ICDM 2011.

Solves, for each column j:

    min_w_j  ½||a_j - A·w_j||²₂ + ½·β·||w_j||²₂ + λ·||w_j||₁
    s.t.     w_j ≥ 0,  w_{jj} = 0

via coordinate descent + soft-thresholding.

When top_k > 0, uses fsSLIM: for each item j, pre-selects the top-K
most similar items (by cosine similarity) as candidate features,
dramatically reducing training time.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ["solve_slim_fs", "_soft_threshold"]


def _soft_threshold(x: float, gamma: float) -> float:
    if abs(x) <= gamma:
        return 0.0
    return (abs(x) - gamma) * (1.0 if x > 0 else -1.0)


def _select_top_k_similar(A: sp.csc_matrix, top_k: int) -> list[np.ndarray]:
    """For each item j, select the top-K most similar items by cosine similarity.

    Returns a list of length n, where result[j] is an array of candidate item indices.
    """
    n = A.shape[1]
    # Cosine similarity on binary data = Jaccard-like
    # Use sklearn's cosine_similarity which handles sparse efficiently
    sim = cosine_similarity(A.T, dense_output=False)  # (n, n) sparse
    candidates = []
    for j in range(n):
        row = sim.getrow(j).toarray().ravel()
        row[j] = -1  # exclude self
        # Top-K (exclude self)
        k = min(top_k, n - 1)
        top_idx = np.argpartition(row, -k)[-k:]
        # Sort by similarity descending
        top_idx = top_idx[np.argsort(-row[top_idx])]
        candidates.append(top_idx)
    return candidates


def solve_slim_fs(
    A: sp.csc_matrix,
    beta: float = 1.0,
    lambda_: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-4,
    top_k: int = 100,
    verbose: bool = True,
) -> sp.csc_matrix:
    """Solve SLIM with feature selection (fsSLIM).

    Parameters
    ----------
    A : csc_matrix, shape (m, n)
        User-item interaction matrix (binary/rating).
    beta : float
        L2 regularisation.
    lambda_ : float
        L1 regularisation.
    max_iter : int
        Maximum coordinate-descent epochs.
    tol : float
        Convergence threshold.
    top_k : int
        Number of candidate items per column (fsSLIM). Set to 0 to
        disable feature selection (use all items, slower).

    Returns
    -------
    W : csc_matrix, shape (n, n), diag(W)=0, W>=0
    """
    A = A.tocsc().astype(np.float64)
    m, n = A.shape

    # Precompute column norms
    # col_norm_sq[k] = ‖A_k‖², item k 列的非零平方和, 更新公式的分母会用到
    col_norm_sq = np.array((A.multiply(A)).sum(axis=0)).ravel()

    # Feature selection
    if top_k > 0 and top_k < n:
        if verbose:
            print(f"  Feature selection: top-{top_k} neighbours per item …")
        candidates = _select_top_k_similar(A, top_k)
    else:
        candidates = [np.arange(n) for _ in range(n)]

    # Initialise W columns as dicts for efficient sparse access
    W_rows: list[list[int]] = [[] for _ in range(n)]
    W_vals: list[list[float]] = [[] for _ in range(n)]

    # Precompute all columns as dense arrays for O(1) inner-loop access
    A_cols = [A[:, j].toarray().ravel() for j in range(n)]
    A_csc = A  # keep for matvec (A_csc @ w_j)

    for epoch in range(max_iter):
        max_change = 0.0
        for j in range(n):
            if col_norm_sq[j] == 0:
                continue

            a_j = A_cols[j]                         # (m,) 真实标签
            w_j = np.zeros(n)                       # (n,) 权重向量
            for idx, k in enumerate(W_rows[j]):
                w_j[k] = W_vals[j][idx]

            # Current prediction: A·w_j
            pred = (A_csc @ w_j).ravel()            # (m,) 预测值

            # Coordinate descent over candidate features only
            for k in candidates[j]:
                if k == j or col_norm_sq[k] == 0:
                    continue

                w_kj = w_j[k]
                A_k = A_cols[k]

                # Residual without feature k
                residual = a_j - pred + A_k * w_kj
                numerator = np.dot(A_k, residual)
                denominator = col_norm_sq[k] + beta
                w_new = numerator / denominator

                # Soft-thresholding + non-negativity
                gamma = lambda_ / denominator
                w_new = _soft_threshold(w_new, gamma)
                if w_new < 0:
                    w_new = 0.0

                change = abs(w_new - w_kj)
                if change > max_change:
                    max_change = change

                if abs(w_new) > 1e-12:
                    # Update prediction incrementally
                    delta = w_new - w_kj
                    pred += A_k * delta
                    w_j[k] = w_new
                # 抹零会把 w_j[k] 设为 0.0, 导致本轮结束 np.nonzero(w_j) 不再包含 k. 下一 epoch 这个 k 就不会被加载进 w_j, 相当于彻底"遗忘"了这个特征
                elif w_kj != 0:
                    w_j[k] = 0.0

            # Store updated column
            nz = np.nonzero(w_j)[0]
            W_rows[j] = nz.tolist()
            W_vals[j] = w_j[nz].tolist()

        nnz = sum(len(r) for r in W_rows)
        if verbose:
            print(
                f"  epoch {epoch+1:3d}/{max_iter}  "
                f"nnz(W)={nnz}  "
                f"density={nnz/(n*n):.4%}  "
                f"max_change={max_change:.6g}"
            )
        if max_change < tol and epoch > 0:
            if verbose:
                print(f"Converged at epoch {epoch+1}")
            break

    # Build final sparse matrix
    all_rows, all_cols, all_vals = [], [], []
    for j in range(n):
        if W_rows[j]:
            all_rows.append(np.array(W_rows[j], dtype=int))
            all_cols.append(np.full(len(W_rows[j]), j, dtype=int))
            all_vals.append(np.array(W_vals[j], dtype=float))

    if all_rows:
        W = sp.csc_matrix(
            (np.concatenate(all_vals), (np.concatenate(all_rows), np.concatenate(all_cols))),
            shape=(n, n),
        )
    else:
        W = sp.csc_matrix((n, n))

    # 删除稀疏矩阵中显式存储的零值. 确保最终 W 矩阵里没有显式存储的零, 减少存储空间并提升下游 @ 运算效率
    W.eliminate_zeros()
    return W
