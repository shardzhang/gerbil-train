# MIMN (Multi-channel Interest with Moment Network)

## Model Architecture

MIMN uses a **multi-slot memory network** to capture multiple aspects of user interests. Unlike DIN/DIEN which compress behavior into a single vector, MIMN maintains K memory slots that jointly represent the interest distribution.

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
    subgraph Memory
        MEM[Memory Slots<br/>K x d]
        WRITE[Write: attention over slots<br/>sequence to memory]
        READ[Read: target query<br/>memory to interest]
    end
    subgraph Encoder
        EMB[Behavior Embedding<br/>B, T, d]
        LSTM[Bi-LSTM<br/>d to 2h]
    end
    subgraph Target
        TGT[Target Embedding<br/>B, d]
        PROJ[Project to 2h]
    end
    FB[feature_bags] --> EMB --> LSTM --> WRITE
    WRITE -.-> MEM
    FB --> TGT --> PROJ --> READ
    MEM -.-> READ
    READ --> CONCAT
    FB --> CONCAT --> MLP_NET --> HEAD --> OUT

    style MEM fill:#f96,stroke:#333
    style WRITE fill:#fc9,stroke:#333
    style READ fill:#cfc,stroke:#333
    style LSTM fill:#f9d,stroke:#333
```

### Memory Write

Each LSTM state at step t writes to the memory via attention:

w_t = softmax(Proj(h_t) x M^T)

M = M + sum(w_t_i x h_t)

### Memory Read

The target item reads from memory via attention:

w = softmax(Proj(q) x M^T)

interest = sum(w_i x M_i)

## Configuration

```yaml
interest_extractor:
  lstm_hidden: 64
  num_memory_slots: 8
```

## Launch

```bash
python -m gerbil_train.cli.10-mimn_train --config configs/10-mimn/experiment.yaml
```

## Sequential Model Comparison

| Model | Interest Representation | Key Technique |
|-------|------------------------|---------------|
| DIN | Single vector | Attention pooling |
| DIEN | Single evolving vector | GRU + AUGRU |
| DSIN | Session vectors | Bi-LSTM + Self-Attn |
| MIMN | Multi-slot distribution | Memory network |
