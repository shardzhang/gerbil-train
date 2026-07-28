# Node2Vec

## Model Architecture

Node2Vec = biased random walks + Skip-gram

```mermaid
graph TB
    subgraph "Step 3: Skip-gram"
        TGT["target_embedding<br/>vocab → d"]
        CTX["context_embedding<br/>vocab → d"]
        LOSS["BCE (pos + neg)"]
    end
    subgraph "Step 2: Random Walks"
        WALKS["biased walks<br/>p=1.0, q=0.5"]
    end
    subgraph "Step 1: Graph"
        SEQ["behavior sequences"]
        GRAPH["co-occurrence graph<br/>weighted adjacency"]
    end
    SEQ --> GRAPH --> WALKS --> TGT & CTX --> LOSS

    style GRAPH fill:#f96,stroke:#333
    style WALKS fill:#fc9,stroke:#333
    style TGT fill:#9df,stroke:#333
    style CTX fill:#9df,stroke:#333
```

### Biased Random Walk

At node v (came from t), transition to x with weight:

```
    ┌ 1/p  if x == t         (return)
w = ┤ 1    if x is neighbor of t  (local)
    └ 1/q  otherwise          (explore)
```

- **p < 1**: walk tends to return to previous node (BFS-like)
- **q < 1**: walk tends to explore outward (DFS-like)
- **p = 1, q = 0.5**: default — moderate outward exploration

## Configuration

```yaml
mlp:
  item_field: user_movie_rate
```

Hyperparameters (hardcoded in trainer):
- `num_walks = 50` (per node)
- `walk_length = 20`
- `p = 1.0`, `q = 0.5`
- `window_size = 5`, `num_neg = 5`

## Launch

```bash
python -m gerbil_train.cli.33-node2vec_train --config configs/33-node2vec/experiment.yaml
```

## References

- Grover, A., and Leskovec, J. "node2vec: Scalable Feature Learning for Networks." KDD 2016.
