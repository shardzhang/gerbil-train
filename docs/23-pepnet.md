# PEPNet (Parameter Efficient Personalized Network)

## Model Architecture

PEPNet personalizes the base model via **EPNet** (Embedding Personalized Network):

```
EPNet: condition_emb → MLP → scale & bias → applied to input features
MMoE:  personalized features → experts → gates → towers → tasks
```

```mermaid
graph TB
    subgraph "Outputs"
        O1["task_1: sigmoid"]
        O2["task_T: sigmoid"]
    end
    subgraph "MMoE Backbone"
        EXP["shared experts"]
        G1["gate_1"]
        G2["gate_T"]
        T1["tower_1"]
        T2["tower_T"]
    end
    subgraph "EPNet"
        COND["domain/user embedding"]
        EP["MLP → tanh(scale) + bias"]
    end
    subgraph "Input"
        EMB["Concat all field embeddings"]
    end
    I[feature_bags] --> EMB
    I --> COND
    COND --> EP
    EMB --> EP
    EP --> EXP
    EP --> G1 & G2
    EXP -->|g₁ weighted| G1 --> T1 --> O1
    EXP -->|g_T weighted| G2 --> T2 --> O2

    style EP fill:#f96,stroke:#333
    style EXP fill:#fc9,stroke:#333
    style COND fill:#9df,stroke:#333
```

### EPNet Detail

```
condition_embedding → [Linear → ReLU → Linear] → [scale (tanh), bias]
x_personalized = x * tanh(scale) + bias
```

### Multi-Task Model Comparison

| Model | Personalization | Expert Sharing | Parameter Generation |
|-------|----------------|----------------|---------------------|
| MMoE | None | Shared | None |
| PLE | None | Shared + specific | None |
| **PEPNet** | **EPNet: scale/bias** | Shared | **Conditioned on domain** |

## Configuration

```yaml
mlp:
  domain_field: user_id
  epnet_hidden: [64, 32]
  num_experts: 8
  expert_hidden: [128, 64]
  tower_hidden: [64, 32]
  num_tasks: 2
  task_names: [rating, click]
```

## Launch

```bash
python -m gerbil_train.cli.23-pepnet_train --config configs/23-pepnet/experiment.yaml
```

## References

- Chang, J., et al. "PEPNet: Parameter Efficient Personalized Network." WWW 2023.
