# EGES (Enhanced Graph Embedding with Side Information)

## Model Architecture

EGES enhances item embeddings by incorporating **side information** (genre, title, brand, etc.) via learned attention weights:

```
e_item = softmax(a)₀·v_base + Σ softmax(a)ₖ·v_sideₖ
```

```mermaid
graph TB
    subgraph "Output"
        OUT["sigmoid"]
    end
    subgraph "MLP"
        CONCAT["Concat user + enhanced item"]
        MLP_NET["MLP"]
        HEAD["Linear Head"]
    end
    subgraph "EGES Aggregation"
        ATT["softmax over sources"]
        SUM["weighted sum → enhanced item"]
    end
    subgraph "Side Info Embeddings"
        E0["base item emb<br/>movie_id → d"]
        E1["side emb 1<br/>movie_genres → d"]
        EK["side emb K<br/>movie_title → d"]
    end
    subgraph "User Features"
        U_EMB["user field embeddings"]
    end
    FB --> U_EMB & E0 & E1 & EK
    E0 & E1 & EK --> ATT --> SUM --> CONCAT
    U_EMB --> CONCAT --> MLP_NET --> HEAD --> OUT

    style E0 fill:#f96,stroke:#333
    style E1 fill:#fc9,stroke:#333
    style EK fill:#9df,stroke:#333
    style ATT fill:#cfc,stroke:#333
```

### Attention Weights

Global learned attention vector `a = [a_0, a_1, ..., a_K]`:
- `a_0`: weight for base item embedding
- `a_k`: weight for k-th side information

After softmax, `Σ softmax(a)_k = 1`.

## Configuration

```yaml
mlp:
  item_field: movie_id
  side_fields: [movie_genres, movie_title]
  hidden_dims: [128, 64]
```

## Launch

```bash
python -m gerbil_train.cli.31-eges_train --config configs/31-eges/experiment.yaml
```

## References

- Wang, J., et al. "Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba." KDD 2018.
