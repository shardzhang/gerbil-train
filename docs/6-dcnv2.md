# DCNv2 (Deep & Cross Network V2)

## Model Architecture

DCNv2 improves DCN-V1 by using a **full d×d weight matrix** per cross layer (vs. vector in V1), enabling richer cross patterns. Also supports low-rank approximation.

### V1 vs V2 Cross Layer

| | DCN-V1 | DCN-V2 |
|---|--------|--------|
| Transformation | `x₀ ⊙ (wₗ · xₗ + bₗ) + xₗ` | `x₀ ⊙ (Wₗ · xₗ + bₗ) + xₗ` |
| Weight | vector `wₗ ∈ ℝᵈ` | matrix `Wₗ ∈ ℝᵈˣᵈ` |
| Cross params/layer | `d` | `d²` (or `2·d·r` with low-rank) |
| Expressiveness | Element-wise scaling | Full linear transform |

```mermaid
graph TB
    subgraph Output
        OUT[sigmoid]
    end
    subgraph Combination
        COMB[Concat → Linear]
    end
    subgraph Cross
        subgraph V2
            W[Wₗ · xₗ<br/>d×d matrix]
        end
        MUL[x₀ ⊙ (·) + xₗ]
        W --> MUL
    end
    subgraph Deep
        MLP[MLP]
    end
    subgraph Input
        EMB[Concat Field Embeddings]
    end
    I[feature_bags] --> EMB
    EMB --> W --> MUL --> COMB
    EMB --> MLP --> COMB
    COMB --> OUT
    style V2 fill:#f96,stroke:#333
    style MLP fill:#9bd,stroke:#333
    style COMB fill:#cfc,stroke:#333
```

### Low-Rank Approximation

When `cross_rank = r > 0`, the weight matrix is factorized:

$$ W_l = U_l \cdot V_l^T \quad (U_l \in \mathbb{R}^{d \times r}, V_l \in \mathbb{R}^{d \times r}) $$

$$ W_l \cdot x_l = U_l \cdot (V_l^T \cdot x_l) = (x_l^T \cdot V_l) \cdot U_l^T $$

This reduces parameters from `d²` to `2·d·r` per layer (e.g., `d=400, r=10`: 160k → 8k, ~95% reduction).

## Configuration

```yaml
field_attention:
  num_cross_layers: 3
  cross_rank: 0            # 0 = full matrix, > 0 = low-rank with rank=r

mlp:
  hidden_dims: [128, 64]
  activation: relu
  dropout: 0.0
  batch_norm: false
  input_batch_norm: false
```

## Launch

```bash
python -m gerbil_train.cli.6-dcnv2_train --config configs/6-dcnv2/experiment.yaml
```
