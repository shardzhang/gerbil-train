# DCN (Deep & Cross Network)

## Model Architecture

DCN combines a **Cross Network** for explicit bounded-degree feature interactions with a standard **Deep MLP**.

$$ \hat{y} = \sigma\big( \text{Linear} + \text{Combine}(\text{Cross}(x_0), \text{Deep}(x_0)) \big) $$

```mermaid
graph TB
    subgraph Output
        OUT[sigmoid]
    end
    subgraph Combination
        COMB[Concat → Linear]
    end
    subgraph Cross
        CN[Cross Network<br/>x₀⊙(Wₗ·xₗ+bₗ)+xₗ]
    end
    subgraph Deep
        MLP[MLP]
    end
    subgraph Input
        EMB[Concat Field Embeddings]
    end
    I[feature_bags] --> EMB
    EMB --> CN --> COMB
    EMB --> MLP --> COMB
    COMB --> OUT
    style CN fill:#f96,stroke:#333
    style MLP fill:#9bd,stroke:#333
    style COMB fill:#cfc,stroke:#333
```

### Cross Layer

Each cross layer computes:

$$ x_{l+1} = x_0 \odot (W_l \cdot x_l + b_l) + x_l \in \mathbb{R}^d $$

where $\odot$ is element-wise multiplication. Each layer adds one more order of interaction.

### Layer-Wise Interaction Orders

```
Layer 0: linear combinations of raw features
Layer 1: 2nd-order interactions
Layer 2: 3rd-order interactions
...
```

## Configuration

```yaml
field_attention:
  num_cross_layers: 3

mlp:
  hidden_dims: [128, 64]
  activation: relu
  dropout: 0.0
  batch_norm: false
  input_batch_norm: false
```

## Launch

```bash
python -m gerbil_train.cli.6-dcn_train --config configs/6-dcn/experiment.yaml
```
