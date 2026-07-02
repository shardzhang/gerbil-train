# AFM (Attentional Factorization Machine)

## Model Architecture

AFM extends FM by learning an **attention weight** for each pair-wise feature interaction via a small MLP.

$$ \hat{y} = \underbrace{w_0 + \sum_i w_i x_i}_{\text{Linear}} + \underbrace{\sum_{i=1}^n \sum_{j=i+1}^n a_{ij} \cdot (v_i \odot v_j)}_{\text{Attentional FM}} $$

### Attention Mechanism

Each pair's attention score is computed by:

$$ a'_{ij} = h^T \cdot \text{ReLU}(W \cdot (v_i \odot v_j) + b) $$

$$ a_{ij} = \frac{\exp(a'_{ij})}{\sum_{(p,q)} \exp(a'_{pq})} $$

```mermaid
graph TB
    subgraph Output
        OUT[sigmoid]
    end
    subgraph Fusion
        ADD[+]
    end
    subgraph Linear
        L1[Linear Embedding<br/>dim=1 per field]
        LS[sum]
    end
    subgraph Attentional_FM
        P[pair-wise products<br/>v_i ⊙ v_j]
        ATTN[Attention Net<br/>Linear + ReLU + Linear]
        SM[softmax over pairs]
        WS[weighted sum]
    end
    subgraph Input
        I[feature_bags]
    end

    I --> L1 --> LS --> ADD
    I --> P --> ATTN --> SM --> WS --> ADD
    ADD --> OUT

    style OUT fill:#4a9,stroke:#333
    style ATTN fill:#fc9,stroke:#333
    style WS fill:#9cf,stroke:#333
```

## Configuration

```yaml
# configs/3-afm/model.yaml
afm_attention:
  hidden_size: 128
  dropout: 0.0

mlp:
  hidden_dims: []
  activation: relu
  dropout: 0.0
  batch_norm: false
  input_batch_norm: false
```

## Launch

```bash
python -m gerbil_train.cli.3-afm_train --config configs/3-afm/experiment.yaml
```
