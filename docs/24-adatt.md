# AdaTT (Adaptive Task-to-Task Fusion Network)

## Model Architecture

AdaTT uses **task-specific experts** and **progressive task-to-task fusion**:

```mermaid
graph TB
    subgraph "Outputs"
        O1["task_1: sigmoid"]
        O2["task_T: sigmoid"]
    end
    subgraph "Towers"
        T1["tower_1"]
        T2["tower_T"]
    end
    subgraph "Fusion Layers × L"
        FL["h_t^{l+1} = h_t^l + MLP( Σ gate_{tt'} · h_{t'}^l )"]
    end
    subgraph "Task Experts"
        E1["expert_1 (task_1 MLP)"]
        E2["expert_T (task_T MLP)"]
    end
    subgraph "Input"
        EMB["Concat field embeddings"]
    end
    I[feature_bags] --> EMB
    EMB --> E1 & E2
    E1 & E2 --> FL --> T1 & T2 --> O1 & O2

    style E1 fill:#f96,stroke:#333
    style E2 fill:#9df,stroke:#333
    style FL fill:#cfc,stroke:#333
```

### Task-to-Task Fusion

Each task t computes a **sample-dependent gate** over all tasks:

```
gate_t = softmax(W_gate_t · h_t)
h_fused = Σ_{t'} gate_t[t'] · h_{t'}
h_t^{l+1} = h_t^l + ReLU(Linear(h_fused))
```

## Multi-Task Model Comparison

| Model | Experts | Task Interaction | Fusion Type |
|-------|---------|-----------------|-------------|
| MMoE | Shared | Via shared experts | Softmax gate |
| PLE | Shared + specific | Per-layer shared/specific | Extraction |
| ESMM | Shared bottom | CTR ↔ CVR product | pCTCVR = pCTR × pCVR |
| PEPNet | Shared (personalized) | EPNet scale/bias | Personalization |
| **AdaTT** | **Per-task** | **Task-to-task gate** | **Cross-task attention** |

## Configuration

```yaml
mlp:
  num_layers: 2
  task_dim: 64
  expert_hidden: 128
  tower_hidden: [32]
  num_tasks: 2
  task_names: [rating, click]
```

## Launch

```bash
python -m gerbil_train.cli.24-adatt_train --config configs/24-adatt/experiment.yaml
```
