# MIND (Multi-Interest Network with Dynamic Routing)

## Model Architecture

MIND introduces **multi-interest representation** via capsule-based dynamic routing, allowing the model to capture **multiple aspects** of a user's interests from behavior sequences.

```mermaid
graph TB
    subgraph Output
        OUT[sigmoid]
    end
    subgraph MLP
        CONCAT[Concat: plain + target + interest]
        MLP_NET[MLP]
        HEAD[Linear Head]
    end
    subgraph Label-Aware Attention
        ATTN[softmax target ✕ V_i]
        SUM[weighted sum over K interests]
    end
    subgraph Dynamic Routing
        INIT[Behavior Capsules<br/>B, T, d]
        ITER[Iterative routing x3]
        V1[Interest Capsule 1]
        V2[Interest Capsule 2]
        VK[Interest Capsule K]
    end
    subgraph Target
        TGT[Target Embedding<br/>B, d]
    end
    FB[feature_bags] --> INIT
    FB --> TGT
    INIT --> ITER --> V1 & V2 & VK
    V1 & V2 & VK --> ATTN
    TGT --> ATTN
    ATTN --> SUM --> CONCAT
    FB --> CONCAT --> MLP_NET --> HEAD --> OUT

    style ATTN fill:#fc9,stroke:#333
    style ITER fill:#cfc,stroke:#333
    style V1 fill:#f96,stroke:#333
    style V2 fill:#f96,stroke:#333
    style VK fill:#f96,stroke:#333
```

### Dynamic Routing (B2I)

Each behavior item is a **behavior capsule**. Interest capsules are iteratively refined:

1. **Bilinear projection**: `u_hat = W * behavior_emb` — projects behavior to K interest spaces
2. **Routing by agreement**: for each iteration:
   - Softmax routing logits → attention weights
   - Weighted sum → interest capsule candidates
   - Squash non-linearity
   - Update logits by `agreement = u_hat · v`

### Label-Aware Attention

For target item t, select the most relevant interest capsule:

```
α_i = exp(t^T v_i) / Σ exp(t^T v_j)
interest = Σ α_i v_i
```

This lets the model choose which interest dimension is triggered by the current target.

## Configuration

```yaml
interest_extractor:
  num_interests: 4    # K interest capsules
  routing_iters: 3    # routing iterations
```

## Launch

```bash
python -m gerbil_train.cli.12-mind_train --config configs/12-mind/experiment.yaml
```

## Sequential Model Comparison

| Model | Interest Type | Extraction Mechanism |
|-------|--------------|---------------------|
| DIN | Single vector | Attention pooling over all items |
| DIEN | Single evolving | GRU + AUGRU with auxiliary loss |
| DSIN | Session vectors | Session division + Bi-LSTM + Self-Attn |
| MIMN | Multi-slot distribution | Memory network (write/read) |
| SIM | Single top-K | GSU retrieval + ESU cross-attention |
| **MIND** | **Multi-vector (K)** | **Dynamic routing (CapsNet)** |
