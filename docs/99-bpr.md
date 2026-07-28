# BPR (Bayesian Personalized Ranking)

## Model Architecture

Bayesian Personalized Ranking (BPR) optimizes a **pairwise ranking loss** for implicit feedback. The model is a simple Matrix Factorization:

```
score_ui = ⟨user_emb_u, item_emb_i⟩
```

### Pairwise Ranking Loss

For a user u with positive item i and sampled negative item j:

```
Loss_BPR = -ln σ(score_ui - score_uj)
         = ln(1 + exp(score_uj - score_ui))
```

This pushes positive items above negative items in the ranking.

### Training

1. Sample a batch of (user, positive_item) from user interactions
2. For each positive, sample `num_neg` negative items uniformly
3. Compute BPR loss over all (pos, neg) pairs
4. Optimize via Adam

### Inference

Direct dot product: `score = user_emb[u] · item_emb[i]`

Can be used for:
- **Ranking**: Score all items for a user, take top-K
- **ANN retrieval**: Export embeddings to Faiss for fast nearest-neighbor search

## Configuration

```yaml
mlp:
  user_field: user_id
  item_field: movie_id

optimizer:
  bpr_num_neg: 5  # negatives per positive
```

## Launch

```bash
python -m gerbil_train.cli.99-bpr_train --config configs/99-bpr/experiment.yaml
```

## References

- Rendle, S., et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI 2009.
