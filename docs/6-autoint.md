# AutoInt (Automatic Feature Interaction)

## Model Architecture

AutoInt treats each feature field as a **token** and uses **multi-head self-attention** (Transformer encoder) to automatically learn feature interactions at various orders.

```mermaid
graph TB
    subgraph Output
        OUT[sigmoid]
    end
    subgraph Fusion
        ADD[+]
    end
    subgraph Linear_Term
        L1[Linear Embedding<br/>dim=1 per field]
        LS[sum]
    end
    subgraph Transformer
        SUB1[Interacting Layer × N]
    end
    subgraph Interacting_Layer
        MHA[Multi-Head<br/>Self-Attention]
        RES1[(+)]
        FFN[Feed-Forward]
        RES2[(+)]
        LN1[LayerNorm]
        LN2[LayerNorm]
    end
    subgraph Deep
        CONCAT[Concat all field outputs]
        MLP[MLP]
        HEAD[Linear Head]
    end
    subgraph Embeddings
        EMB[Field Embeddings<br/>dim=k per field]
    end

    I[feature_bags] --> L1 --> LS --> ADD
    I --> EMB --> SUB1 --> CONCAT --> MLP --> HEAD --> ADD
    SUB1 -.->|stacked| SUB1

    ADD --> OUT

    SUB1 --> LN1 --> MHA --> RES1 --> LN2 --> FFN --> RES2
    RES1 -.-> SUB1
    RES2 -.-> SUB1

    style MHA fill:#fc9,stroke:#333
    style FFN fill:#f96,stroke:#333
    style EMB fill:#ccf,stroke:#333
    style HEAD fill:#9bd,stroke:#333
```

### Multi-Head Self-Attention over Fields

Each attention head learns a different interaction pattern between fields:

$$ \text{head}_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_h $$

$$ \text{MHA}(X) = \text{concat}(\text{head}_1, ..., \text{head}_H) W^O $$

### Interacting Layer (Transformer Encoder)

$$
\begin{aligned}
X' &= X + \text{MHA}(\text{LayerNorm}(X)) \\
X'' &= X' + \text{FFN}(\text{LayerNorm}(X'))
\end{aligned}
$$

## Configuration

```yaml
auto_attention:
  num_layers: 3      # number of stacked Transformer layers
  num_heads: 2       # number of attention heads
  attn_dim: 32       # attention dimension (must be divisible by num_heads)
  dropout: 0.0       # dropout rate

mlp:
  hidden_dims:
  - 128
  activation: relu
  dropout: 0.0
  batch_norm: false
  input_batch_norm: false
```

## Launch

```bash
python -m gerbil_train.cli.6-autoint_train --config configs/6-autoint/experiment.yaml
```

## Comparison with Other Models

| Model | Interaction Type | Mechanism |
|-------|----------------|-----------|
| FM | 2nd-order explicit | Pair-wise dot product |
| PNN | Explicit inner products | Product Layer + MLP |
| DeepFM | FM + MLP implicit | Shared embeddings |
| **AutoInt** | Multi-order **learned** | Transformer self-attention |
