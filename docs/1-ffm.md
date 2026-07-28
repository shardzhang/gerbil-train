# FFM (Field-aware Factorization Machine)

## Model Architecture

FFM extends FM by learning a **separate embedding per field pair**, so each feature interacts with different fields in different ways.

```mermaid
graph TB
    subgraph "Output"
        OUT["sigmoid"]
    end
    subgraph "Fusion"
        HEAD["Linear Head<br/>linear + ffm + direct"]
    end
    subgraph "FFM_Term"
        PAIRS["For each pair (i, j):"]
        DOT["⟨v_{"i,f_j"}, v_{"j,f_i"}⟩"]
        SUM_ALL["Sum over all pairs"]
    end
    subgraph "Linear_Term"
        L1["Linear EmbeddingBag<br/>dim=1 per field"]
        LS["sum"]
    end
    subgraph "Field_Aware_Embeddings"
        E1["field i → j: EmbeddingBag"]
        E2["field j → i: EmbeddingBag"]
    end
    subgraph "Input"
        FB["feature_bags"]
    end

    FB --> L1 --> LS --> HEAD
    FB --> E1 & E2 --> DOT --> SUM_ALL --> HEAD
    FB --> HEAD

    style PAIRS fill:#f96,stroke:#333
    style DOT fill:#fc9,stroke:#333
    style E1 fill:#9df,stroke:#333
    style E2 fill:#9df,stroke:#333
```

### FFM vs FM

| | FM | FFM |
|--|-----|-----|
| Pairwise interaction | ⟨v_i, v_j⟩ | **⟨v_{i,f_j}, v_{j,f_i}⟩** |
| Embeddings per field | 1 | **N-1** (one per other field) |
| Parameters | O(Kd) | **O(K²d)** |
| Expressiveness | Single view per feature | **Field-aware views** |

### Intuition

The same feature (e.g., movie_id=123) interacts **differently** with:
- `user_id`: "did this user like this movie?"
- `user_age`: "is this movie suitable for this age group?"
- `context_time_hour`: "is this movie typically watched at this hour?"

FFM captures these different interaction patterns via separate embeddings.

## Number of Field Pairs

With N categorical fields, FFM computes:

$$ \text{FFM}(x) = \sum_{i=1}^{N} \sum_{j=i+1}^{N} \langle v_{i, f_j}, v_{j, f_i} \rangle $$

For N fields, that's N×(N-1)/2 interaction terms, each requiring its own pair of embeddings.

## Launch

```bash
python -m gerbil_train.cli.1-ffm_train --config configs/1-ffm/experiment.yaml
```
