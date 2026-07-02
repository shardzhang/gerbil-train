# NFM (Neural Factorization Machine)

## Model Architecture

NFM replaces FM's scalar-weighted sum with **Bi-Interaction Pooling** followed by an MLP for non-linear transformation.

$$ \hat{y} = \underbrace{w_0 + \sum_i w_i x_i}_{\text{Linear}} + \underbrace{\text{MLP}(\sum_{i=1}^n \sum_{j=i+1}^n v_i \odot v_j)}_{\text{Bi-Interaction + Deep}} $$

### Bi-Interaction Pooling

$$ f_{BI} = \frac{1}{2} \left[ \big( \sum_{i=1}^n v_i \big)^2 - \sum_{i=1}^n v_i^2 \right] \in \mathbb{R}^k $$

Same mathematical form as FM, but outputs a **k-dimensional vector** (not a scalar), preserving information for subsequent non-linear transformation.

```mermaid
graph TB
    subgraph Output
        OUT[sigmoid]
    end
    subgraph Fusion
        ADD[+]
    end
    subgraph Linear
        L1[Linear Embedding<br/>dim=1 per field]
        LS[sum]
    end
    subgraph Bi_Interaction
        BI[Bi-Interaction Pooling<br/>Σ_i Σ_{j>i} v_i ⊙ v_j → k-dim vector]
    end
    subgraph Deep
        MLP[MLP<br/>k → 128 → 64 → 1]
    end
    subgraph Input
        I[feature_bags]
    end

    I --> L1 --> LS --> ADD
    I --> BI --> MLP --> ADD
    ADD --> OUT

    style BI fill:#f96,stroke:#333
    style MLP fill:#9bd,stroke:#333
    style OUT fill:#4a9,stroke:#333
```

## Key Insight

FM outputs a **scalar** (no room for non-linear transformation).
DeepFM concatenates embeddings (high-dim input to MLP).
**NFM** pools interactions into a **k-dim vector** (same as embedding size), then feeds to MLP — much more parameter-efficient than DeepFM while capturing non-linear interactions.

## Configuration

```yaml
# configs/3-nfm/model.yaml
mlp:
  hidden_dims:
  - 128
  - 64
  activation: relu
  dropout: 0.0
  batch_norm: false
  input_batch_norm: false
```

## Launch

```bash
python -m gerbil_train.cli.3-nfm_train --config configs/3-nfm/experiment.yaml
```

## Comparison

| Model | Pair-wise Aggregation | Deep Component | MLP Input Dim |
|-------|----------------------|----------------|---------------|
| FM | Sum → scalar | None | — |
| AFM | Weighted sum → scalar | None | — |
| **NFM** | Bi-Interaction → **k-dim vector** | MLP(k → ...) | **k** |
| DeepFM | Sum → scalar (+ concat all embs) | MLP(n·k → ...) | **n·k** |
