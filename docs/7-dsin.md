# DSIN (Deep Session Interest Network)

## Model Architecture

DSIN models user behavior as **sessions** rather than a flat sequence. Sessions are divided by time gaps, and each session captures a distinct user intent.

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

    subgraph Interest_Extraction
        SESS[Split into sessions<br/>B, S, L]
        EMB[Behavior Embedding<br/>B, S, L, d]
        BIAS[Bias Encoding<br/>session + position]
        LSTM[Bi-LSTM per session<br/>d to 2h]
        POOL[Mean pooling<br/>B, S, 2h]
    end

    subgraph Session_Interaction
        SA[Multi-Head Self-Attention<br/>across sessions]
        RES[(+) Residual]
    end

    subgraph Attention_Pooling
        AU[Concat + Linear + ReLU + Linear]
        SM[Softmax]
        WS[Weighted Sum to B, 2h]
    end

    subgraph Target
        TGT[Target Item Embedding]
    end

    FB[feature_bags] --> SESS
    SESS --> EMB --> BIAS --> LSTM --> POOL --> SA --> RES
    RES --> AU
    TGT --> AU
    AU --> SM --> WS --> CONCAT
    FB --> CONCAT
    CONCAT --> MLP_NET --> HEAD --> OUT

    style SESS fill:#f9d,stroke:#333
    style LSTM fill:#f96,stroke:#333
    style SA fill:#fc9,stroke:#333
    style AU fill:#cfc,stroke:#333
```

### Session Interest Extractor

Each session is processed independently by a Bi-LSTM:

session_vec_k = (1/L) * sum(BiLSTM(x_k)) in R^{2h}

### Multi-Head Self-Attention across Sessions

Sessions interact to share contextual information:

Attended = MultiHead(session_vecs, session_vecs, session_vecs)

session_final = session_vecs + Attended

### Target Attention (DIN-style)

Each session interest is weighted by its relevance to the target item:

a_k = MLP(concat(session_k, target))

w_k = softmax(a_k)

v_interest = sum(w_k * session_k)

## Configuration

```yaml
interest_extractor:
  num_sessions: 4
  session_len: 10
  lstm_hidden: 64
  attn_heads: 4
  attn_hidden: 64
```

## Launch

```bash
python -m gerbil_train.cli.7-dsin_train --config configs/7-dsin/experiment.yaml
```

## Sequential Model Comparison

| Model | Behavior Modeling | Key Technique |
|-------|------------------|---------------|
| DIN | Flat sequence | Attention pooling |
| DIEN | Flat sequence + evolution | GRU + AUGRU |
| DSIN | Session-based | Bi-LSTM + Self-Attention |
