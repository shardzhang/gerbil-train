# DLRM (Deep Learning Recommendation Model)

## Model Architecture

DLRM processes sparse features (embeddings) and dense features (bottom MLP),
computes **all pairwise dot products**, and feeds them into a top MLP.

```mermaid
graph TB
    subgraph "Output"
        OUT["sigmoid"]
    end
    subgraph "Top MLP"
        TOP["concat(interaction, dense_emb)"]
        TOP_MLP["MLP → 256 → 128 → 64"]
        HEAD["Linear"]
    end
    subgraph "Interaction"
        INTER["all pairwise dot products"]
        PAIRS["N×(N-1)/2 pairs"]
    end
    subgraph "Sparse Features"
        SPARSE_EMB["embedding bags<br/>vocab → d"]
    end
    subgraph "Dense Features"
        DENSE["concat continuous"]
        BOTTOM["bottom MLP<br/>→ dense_emb (d-dim)"]
    end
    I[feature_bags] --> SPARSE_EMB
    I --> DENSE --> BOTTOM
    SPARSE_EMB & BOTTOM --> INTER
    INTER & BOTTOM --> TOP --> TOP_MLP --> HEAD --> OUT

    style SPARSE_EMB fill:#f96,stroke:#333
    style BOTTOM fill:#fc9,stroke:#333
    style INTER fill:#cfc,stroke:#333
    style TOP_MLP fill:#9bd,stroke:#333
```

### Interaction Layer

For N features (each d-dim), computes all pairwise dot products:

```
gram = X · X^T        # [B, N, N]
interaction = upper_tri(gram, offset=1)  # [B, N(N-1)/2]
```

For N=40 features and d=8, this produces 780 interaction terms.

## Configuration

```yaml
mlp:
  bottom_hidden: [128, 64]    # dense → d-dim vector
  top_hidden: [256, 128, 64]  # interaction + dense → prediction
```

## Launch

```bash
python -m gerbil_train.cli.6-dlrm_train --config configs/6-dlrm/experiment.yaml
```

## References

- Naumov, M., et al. "Deep Learning Recommendation Model for Personalization and Recommendation Systems." RecSys 2019.
