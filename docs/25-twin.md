# TWIN (Two-stage Interest Discovery Network)

## Model Architecture

TWIN discovers multi-interest via prototype assignment, then aggregates via target-aware attention:

```mermaid
graph TB
    subgraph "Output"
        OUT["sigmoid"]
    end
    subgraph "Stage 2: Interest Aggregation"
        ATT["target-attention over K interests"]
    end
    subgraph "Stage 1: Interest Discovery"
        PROTOS["K learned prototypes"]
        ASSIGN["soft assignment items → prototypes"]
        INTERESTS["K interest vectors"]
    end
    subgraph "Encoder"
        SEQ["behavior items"]
        EMB["item embeddings"]
    end
    subgraph "Target"
        TGT["target embedding"]
    end
    FB --> SEQ --> EMB --> ASSIGN
    PROTOS --> ASSIGN --> INTERESTS
    FB --> TGT --> ATT
    INTERESTS --> ATT --> CONCAT["concat(plain, target, interest)"] --> MLP --> OUT

    style PROTOS fill:#f96,stroke:#333
    style ASSIGN fill:#fc9,stroke:#333
    style ATT fill:#cfc,stroke:#333
    style INTERESTS fill:#9df,stroke:#333
```

### Stage 1: Interest Discovery

Each item assigns to K interest prototypes via softmax similarity:

```
assignment = softmax(seq_emb · prototypes^T)
interest_k = Σ assignment_k · item_emb / ||...||
```

### Stage 2: Interest Aggregation

Target attends over K discovered interests:

```
score_k = MLP(concat(interest_k, target))
attention = softmax(scores)
interest = Σ attention_k · interest_k
```

## Sequential Model Comparison

| Model | Interest Extraction | Number of Interests | Mechanism |
|-------|-------------------|-------------------|-----------|
| DIN | Single | 1 | Attention over all items |
| MIND | Multiple | K (config) | Dynamic routing (CapsNet) |
| **TWIN** | **Multiple** | **K (config)** | **Prototype assignment + target attn** |

## Configuration

```yaml
interest_extractor:
  num_interests: 4
```

## Launch

```bash
python -m gerbil_train.cli.25-twin_train --config configs/25-twin/experiment.yaml
```
