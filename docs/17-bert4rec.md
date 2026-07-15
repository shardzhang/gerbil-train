# BERT4Rec

## Model Architecture

BERT4Rec applies **bidirectional Transformer** (no causal masking) on behavior sequences. A `[CLS]` token is prepended and its output serves as the global interest representation.

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
    subgraph Bidirectional Transformer
        CLS[prepend [CLS]]
        POS[+ Positional Encoding]
        BI[No causal mask<br/>bidirectional attention]
        TE1[Transformer Layer 1]
        TE2[Transformer Layer 2]
        CLS_OUT[[CLS] output → interest]
    end
    subgraph Input
        SEQ[Behavior items<br/>B, T, d]
    end
    FB --> SEQ --> CLS --> POS --> BI --> TE1 --> TE2 --> CLS_OUT --> CONCAT
    FB --> CONCAT --> MLP_NET --> HEAD --> OUT

    style CLS fill:#f96,stroke:#333
    style BI fill:#fc9,stroke:#333
    style CLS_OUT fill:#cfc,stroke:#333
```

### Attention Comparison

```
SASRec (Causal):        BERT4Rec (Bidirectional):
item₁  item₂  item₃     item₁  item₂  item₃
  0     -∞     -∞         0      0      0
  0      0     -∞         0      0      0
  0      0      0         0      0      0
```

- **SASRec**: item at position t can only attend to items at positions ≤ t
- **BERT4Rec**: all items attend to all items (only padding is masked)

### [CLS] Token

A learnable embedding prepended to the sequence. After bidirectional Transformer, its output aggregates information from all behavior items.

## Launch

```bash
python -m gerbil_train.cli.17-bert4rec_train --config configs/17-bert4rec/experiment.yaml
```

## Transformer Model Comparison

| Model | Attention | Target in Seq | Interest Source |
|-------|-----------|---------------|-----------------|
| BST | **Bidirectional** | **Yes (appended)** | Target position |
| SASRec | **Causal** (L→R) | No | Last behavior |
| **BERT4Rec** | **Bidirectional** | No | **[CLS] token** |
