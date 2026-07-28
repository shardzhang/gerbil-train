# MMoE (Multi-gate Mixture-of-Experts)

## Model Architecture

MMoE shares K expert networks across T tasks, with task-specific gates:

```python
f_t(x) = tower_t( Σ_k g_t(x)_k · expert_k(x) )
```

```mermaid
graph TB
    subgraph "Outputs"
        O1["task_1: sigmoid"]
        O2["task_T: sigmoid"]
    end
    subgraph "Towers"
        T1["tower_1 MLP"]
        T2["tower_T MLP"]
    end
    subgraph "Gates"
        G1["gate_1<br/>softmax over K"]
        G2["gate_T<br/>softmax over K"]
    end
    subgraph "Experts"
        E1["expert_1 MLP"]
        E2["expert_2 MLP"]
        EK["expert_K MLP"]
    end
    subgraph "Input"
        EMB["Concat all field embeddings"]
    end
    I[feature_bags] --> EMB
    EMB --> E1 & E2 & EK
    EMB --> G1 & G2
    E1 & E2 & EK -->|"g_1 weighted sum"| G1
    E1 & E2 & EK -->|"g_T weighted sum"| G2
    G1 --> T1 --> O1
    G2 --> T2 --> O2

    style E1 fill:#f96,stroke:#333
    style E2 fill:#fc9,stroke:#333
    style EK fill:#f96,stroke:#333
    style G1 fill:#cfc,stroke:#333
    style G2 fill:#cfc,stroke:#333
```

### Key Insight

Unlike Shared-Bottom (one shared representation for all tasks), MMoE allows
each task to **selectively use different experts**, so tasks can share knowledge
when beneficial while maintaining task-specific capacity.

## Configuration

```yaml
mlp:
  num_experts: 8              # K shared experts
  expert_hidden: [128, 64]    # each expert's MLP
  gate_hidden: [32]           # gate MLP (optional)
  tower_hidden: [64, 32]      # per-task tower
  num_tasks: 2                # number of tasks
  task_names: [rating, click] # task output names
  dropout: 0.1
```

## Launch

```bash
python -m gerbil_train.cli.21-mmoe_train --config configs/21-mmoe/experiment.yaml
```

## Multi-Task Model Comparison

| Model | Task Relationship | Sharing Mechanism |
|-------|------------------|-------------------|
| Shared-Bottom | **Hard** parameter sharing | Single shared bottom |
| **MMoE** | **Soft** routing via gates | Weighted experts per task |
