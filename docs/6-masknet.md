# MaskNet

## Model Architecture

MaskNet uses **instance-guided mask blocks**: each sample generates its own
mask from its feature embeddings, applies it element-wise, then passes through
a FC layer.

```mermaid
graph TB
    subgraph "Output"
        OUT["sigmoid"]
    end
    subgraph "Head"
        FC["Linear(direct + block_out)"]
    end
    subgraph "Mask Blocks ×L"
        BLOCK_L["Mask Block L"]
        BLOCK_2["Mask Block 2"]
        BLOCK_1["Mask Block 1"]
    end
    subgraph "Mask Block Detail"
        MEAN["mean pool over fields"]
        GEN["Linear → ReLU → Linear"]
        MASK["mask = σ / ReLU"]
        MUL["⊙ element-wise"]
        FC_BLOCK["FC + ReLU"]
    end
    subgraph "Input"
        EMB["field embeddings"]
    end
    I[feature_bags] --> EMB --> BLOCK_1 --> BLOCK_2 --> BLOCK_L
    EMB --> MEAN --> GEN --> MASK --> MUL --> FC_BLOCK
    EMB --> MUL
    BLOCK_L --> FC --> OUT
    I --> FC

    style MEAN fill:#f96,stroke:#333
    style GEN fill:#fc9,stroke:#333
    style MASK fill:#cfc,stroke:#333
    style MUL fill:#9df,stroke:#333
```

### Instance-Guided Mask

```
M = ReLU(W_2 · LayerNorm(mean_pool(E)) + b_2)    # sample-dependent
E_masked = E ⊙ M                                    # element-wise
output = ReLU(W_1 · E_masked + b_1)                 # FC
```

## Comparison

| Model | Interaction | Instance-dependent | Mask |
|-------|------------|-------------------|------|
| DCN | Cross network | No | No |
| AutoInt | Self-attention | Yes | Attention |
| **MaskNet** | **Instance mask + FC** | **Yes** | **Feature-level mask** |

## Configuration

```yaml
mlp:
  num_blocks: 2
  mask_hidden: [256]
  reduction_ratio: 4
```

## Launch

```bash
python -m gerbil_train.cli.6-masknet_train --config configs/6-masknet/experiment.yaml
```
