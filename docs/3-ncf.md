# NCF (Neural Collaborative Filtering)

## Model Architecture

NeuMF (Neural Matrix Factorization) fuses **GMF** and **MLP** at the **vector level**:

- **GMF**: p_u^G ⊙ q_i^G → φ_GMF (element-wise product vector)
- **MLP**: [p_u^M; q_i^M] → MLP → φ_MLP (last hidden layer vector)
- **Fusion**: h^T · [φ_GMF; φ_MLP; plain] → σ (no bias)

GMF and MLP use **separate** user/item embeddings (per paper Section 3.4).

```mermaid
graph TB
    subgraph "Output"
        OUT["sigmoid"]
    end
    subgraph "NeuMF Fusion"
        FUSION["Concat φ_GMF + φ_MLP + plain"]
        HEAD["h^T · φ   (no bias)"]
    end
    subgraph "MLP Path"
        MLP_USER["mlp_user_emb"]
        MLP_ITEM["mlp_item_emb"]
        MLP_CONCAT["Concat"]
        MLP_NET["MLP layers<br/>2d → 64 → 32 → 16"]
    end
    subgraph "GMF Path"
        GMF_USER["gmf_user_emb"]
        GMF_ITEM["gmf_item_emb"]
        GMF_PROD["Element-wise product<br/>p ⊙ q → φ_GMF"]
    end
    subgraph "Embeddings"
        U_EMB["user_id"]
        I_EMB["item_id"]
        P_EMB["Plain fields"]
    end
    FB --> U_EMB & I_EMB & P_EMB
    U_EMB --> GMF_USER & MLP_USER
    I_EMB --> GMF_ITEM & MLP_ITEM
    GMF_USER & GMF_ITEM --> GMF_PROD
    MLP_USER & MLP_ITEM --> MLP_CONCAT --> MLP_NET
    GMF_PROD & MLP_NET & P_EMB --> FUSION --> HEAD --> OUT

    style GMF_PROD fill:#f96,stroke:#333
    style MLP_NET fill:#9bd,stroke:#333
    style FUSION fill:#cfc,stroke:#333
    style HEAD fill:#cfc,stroke:#333
```

## Differences from Simple MF

| | MF (Matrix Factorization) | NCF (NeuMF) |
|--|--------------------------|-------------|
| Interaction | ⟨p_u, q_i⟩ (inner product) | GMF(⊙) + MLP(concat→MLP) |
| Linearity | Linear only | **Linear (GMF) + Non-linear (MLP)** |
| Embeddings | Shared | **Separate for GMF / MLP** |
| Fusion | N/A | **Vector-level concat → h^T** |

## Configuration

```yaml
mlp:
  user_field: user_id         # designate user field
  item_field: user_movie_rate # designate item field
  hidden_dims: [64, 32, 16]   # MLP layers (tower pattern)
  activation: relu
  dropout: 0.1
  batch_norm: true
```

## Launch

```bash
python -m gerbil_train.cli.3-ncf_train --config configs/3-ncf/experiment.yaml
```

## References

- He, X., et al. "Neural Collaborative Filtering." WWW 2017.
