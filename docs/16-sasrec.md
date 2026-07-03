# SASRec (Self-Attentive Sequential Recommendation)

## Model Architecture

SASRec uses **causal (left-to-right) self-attention** over behavior sequences, producing interest at each point using only past items — no target leakage.

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
    subgraph Causal Transformer
        POS[+ Positional Encoding]
        CAUSAL[Causal Mask<br/>item_i attends to item_{<i} only]
        TE1[Transformer Layer 1]
        TE2[Transformer Layer 2]
        LAST[Take last valid output → interest]
    end
    subgraph Input
        SEQ[Behavior items<br/>B, T, d]
    end
    FB --> SEQ --> POS --> CAUSAL --> TE1 --> TE2 --> LAST --> CONCAT
    FB --> CONCAT --> MLP_NET --> HEAD --> OUT

    style CAUSAL fill:#f96,stroke:#333
    style LAST fill:#cfc,stroke:#333
```

### Causal Mask

```
         item₁  item₂  item₃  ...  itemₜ
item₁      0     -∞     -∞          -∞
item₂      0      0     -∞          -∞
item₃      0      0      0          -∞
...                              ...
itemₜ      0      0      0           0
```

Each item can only attend to itself and earlier items. This prevents information leakage from future to past.

### Key Difference from BST

| | BST | SASRec |
|--|-----|--------|
| Attention | Bidirectional (full) | **Causal (left-to-right)** |
| Target | **Appended to sequence** | Separate (concat in MLP) |
| Interest | Target-aware output | **Behavior-only (last position)** |
| Use case | CTR | **Next-item prediction** |

## Configuration

```yaml
interest_extractor:
  num_heads: 4       # attention heads
  num_layers: 2      # transformer layers
  ffn_hidden: 128    # FFN hidden size
  dropout: 0.1       # dropout
```

## Launch

```bash
python -m gerbil_train.cli.16-sasrec_train --config configs/16-sasrec/experiment.yaml
```

## Sequential Model Comparison

| Model | Attention Type | Target in Seq? | Item-Item |
|-------|---------------|----------------|-----------|
| DIN | Target-behavior | No | No |
| BST | Bidirectional | **Appended** | **Full** |
| **SASRec** | **Causal** | **No** | **Causal** |
