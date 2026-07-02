# PNN (Product-based Neural Network)

## Model Architecture

PNN captures pair-wise feature interactions via a **Product Layer** that computes all inner products between field embeddings, then concatenates them with the raw embeddings and feeds the result to an MLP.

$$ \hat{y} = \underbrace{w_0 + \sum_i w_i x_i}_{\text{Linear}} + \underbrace{\text{MLP}( [v_1, ..., v_n, \langle v_1,v_2\rangle, ..., \langle v_{n-1},v_n\rangle] )}_{\text{Product Layer + Deep}} $$

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
    subgraph Product_Layer
        CONCAT[Concat]
    end
    subgraph Deep
        MLP[MLP<br/>n·k + C(n,2) → ... → 1]
    end
    subgraph Embeddings
        E1[Field Embedding 1<br/>dim=k]
        E2[Field Embedding 2<br/>dim=k]
        E3[...]
    end
    subgraph Inner_Products
        IP[⟨v₁, v₂⟩, ⟨v₁, v₃⟩, ...]
    end

    I[feature_bags] --> L1 --> LS --> ADD
    I --> E1 & E2 & E3
    E1 & E2 & E3 --> CONCAT
    E1 & E2 & E3 --> IP
    IP --> CONCAT
    CONCAT --> MLP --> ADD
    ADD --> OUT

    style Product_Layer fill:#fc9,stroke:#333
    style IP fill:#f96,stroke:#333
    style MLP fill:#9bd,stroke:#333
```

### Product Layer Details

The Product Layer combines two signals:

| Signal | Formula | Shape |
|--------|---------|-------|
| **Linear (z)** | `concat(v₁, ..., vₙ)` | `[n·k]` |
| **Product (p)** | `[⟨v₁,v₂⟩, ⟨v₁,v₃⟩, ..., ⟨vₙ₋₁,vₙ⟩]` | `[C(n,2)]` |

## Configuration

```yaml
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
python -m gerbil_train.cli.3-pnn_train --config configs/3-pnn/experiment.yaml
```

## Comparison

| Model | Pair-wise Aggregation | MLP Input |
|-------|----------------------|-----------|
| NFM | Bi-Interaction Pooling → [k] | **k** |
| **PNN** | concat(embs) + C(n,2) inner products | **n·k + C(n,2)** |
| DeepFM | concat(embs) → FM + MLP | n·k |
