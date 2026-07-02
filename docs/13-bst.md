# BST (Behavior Sequence Transformer)

## Model Architecture

BST applies **Transformer encoder** to user behavior sequences, where **self-attention** captures complex pairwise item-item interactions.

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
    subgraph Transformer
        POS[+ Positional Encoding]
        TE1[Transformer Layer 1<br/>Multi-Head Self-Attn + FFN]
        TE2[Transformer Layer 2<br/>Multi-Head Self-Attn + FFN]
        GATHER[Take target position output]
    end
    subgraph Input
        SEQ[Behavior Seq<br/>B, T, d]
        TGT[Target<br/>B, 1, d]
        CONCAT_SEQ[Concat behavior + target]
    end
    FB[feature_bags] --> SEQ --> CONCAT_SEQ
    FB --> TGT --> CONCAT_SEQ
    CONCAT_SEQ --> POS --> TE1 --> TE2 --> GATHER --> CONCAT
    FB --> CONCAT --> MLP_NET --> HEAD --> OUT

    style TE1 fill:#f96,stroke:#333
    style TE2 fill:#f96,stroke:#333
    style POS fill:#fc9,stroke:#333
    style GATHER fill:#cfc,stroke:#333
```

### Key Insight

DIN uses simple attention weighted-sum over all items. BST uses **self-attention** which models **item-item interactions** (e.g., "if user bought iPhone + AirPods, they're likely to buy MacBook"). This pairwise interaction is missing in DIN/DIEN.

### Positional Encoding

Since self-attention is permutation-invariant, BST adds learned position embeddings so the model knows item order in the sequence.

### Transformer Encoder

Standard Transformer with:
- Multi-head self-attention
- Feed-forward network (FFN)
- Residual connections + LayerNorm
- Key-padding mask for variable-length sequences

## Configuration

```yaml
interest_extractor:
  num_heads: 4       # attention heads
  num_layers: 2      # transformer layers
  ffn_hidden: 128    # FFN hidden size
  dropout: 0.1       # dropout rate
```

## Launch

```bash
python -m gerbil_train.cli.13-bst_train --config configs/13-bst/experiment.yaml
```

## Sequential Model Comparison

| Model | Interest Extraction | Item-Item Interaction |
|-------|-------------------|----------------------|
| DIN | Attention over all items | No (independent) |
| DIEN | GRU evolution + AUGRU | Sequential only |
| DSIN | Session segmentation + Bi-LSTM | Within-session |
| MIMN | Memory network | Via memory slots |
| SIM | GSU retrieval + ESU | Among top-K only |
| MIND | Dynamic routing (CapsNet) | No (independent) |
| **BST** | **Transformer** | **Full pairwise via self-attn** |
