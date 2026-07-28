# Word2Vec (Skip-gram with Negative Sampling)

## Model Architecture

Learns item embeddings from behavior sequences:

```mermaid
graph LR
    subgraph "Training"
        POS["Positive: (t, t±w)"]
        NEG["Negative: (t, random)"]
        LOSS["BCE loss"]
    end
    subgraph "Embeddings"
        TGT["target_embedding<br/>vocab → d"]
        CTX["context_embedding<br/>vocab → d"]
    end
    TGT -->|"⟨emb(tgt), emb(ctx)⟩"| POS
    TGT --> NEG
    CTX --> POS
    CTX --> NEG
    POS & NEG --> LOSS

    style TGT fill:#f96,stroke:#333
    style CTX fill:#9df,stroke:#333
```

### Skip-gram

For each target item at position `t`, predict context items within a window:

```
window_size = 5
context = seq[t-5:t] + seq[t+1:t+6]
```

### Final Embedding

After training, the item embedding is the average of target and context:

```python
item_emb = (target_embedding[id] + context_embedding[id]) / 2
```

## Configuration

```yaml
mlp:
  item_field: user_movie_rate   # behavior sequence field
```

Hyperparameters (hardcoded in trainer):
- `window_size = 5`
- `num_neg = 5`

## Launch

```bash
python -m gerbil_train.cli.32-word2vec_train --config configs/32-word2vec/experiment.yaml
```

## References

- Mikolov, T., et al. "Distributed Representations of Words and Phrases and their Compositionality." NIPS 2013.
