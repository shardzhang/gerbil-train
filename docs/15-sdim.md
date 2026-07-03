# SDIM (Semantic Deep Interest Model)

## Model Architecture

SDIM learns a **probabilistic semantic mask** over behavior sequences via Gumbel-Sigmoid reparameterization, softly or discretely filtering out target-irrelevant items.

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
    subgraph Semantic Mask
        GATE[Gate Network<br/>[target; item; target*item; target-item]]
        P[mask = sigmoid / Gumbel-Sigmoid<br/>∈ [0, 1]]
        FILL[mask out irrelevant items]
        AGG[Masked aggregation + normalize]
    end
    subgraph Input
        SEQ[Behavior items<br/>B, T, d]
        TGT[Target<br/>B, d]
    end
    FB --> SEQ --> GATE
    FB --> TGT --> GATE
    GATE --> P --> FILL
    SEQ --> FILL --> AGG --> CONCAT
    FB --> CONCAT --> MLP_NET --> HEAD --> OUT

    style GATE fill:#f96,stroke:#333
    style P fill:#fc9,stroke:#333
    style FILL fill:#cfc,stroke:#333
```

### Semantic Mask

For each behavior item k, the mask probability is:

```
p_k = σ(MLP([v_target; v_behavior_k; v_target ⊙ v_behavior_k; v_target - v_behavior_k]))
```

During training, Gumbel-Sigmoid makes the mask discrete {0, 1} with straight-through gradients. During inference, sigmoid gives a soft mask.

### Aggregation

```
interest = Σ(p_k · v_k) / max(Σ p_k, 1)
```

Only items with high semantic relevance pass through. Items with p_k ≈ 0 are effectively filtered out.

## Configuration

```yaml
interest_extractor:
  mask_hidden: [64, 32]   # gate network hidden dims
  gumbel_tau: 1.0         # temperature
  gumbel_hard: false      # discrete masking
```

## Launch

```bash
python -m gerbil_train.cli.15-sdim_train --config configs/15-sdim/experiment.yaml
```

## Sequential Model Comparison

| Model | Filtering | Discrete | Training |
|-------|-----------|----------|----------|
| DIN | Soft (attention) | No | Standard |
| MIND | Capsule routing | No | Standard |
| ETA | LSH hash match | Yes (eval) | STE |
| **SDIM** | **Probabilistic mask** | **Yes (train)** | **Gumbel-Sigmoid** |
