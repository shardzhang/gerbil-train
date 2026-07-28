# PLE (Progressive Layered Extraction)

## Model Architecture

PLE extends MMoE with **multi-level extraction layers**, progressively separating shared and task-specific knowledge:

```
f_t(x) = tower_t( gate_t( expert_shared ⊕ expert_specific_t ) )
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
    subgraph "Extraction Layer L"
        L_SHARED_OUT["shared output"]
        L_T1_OUT["task_1 output"]
        L_T2_OUT["task_T output"]
        L_SHARED_GATE["shared gate"]
        L_T1_GATE["task_1 gate"]
        L_T2_GATE["task_T gate"]
        L_SHARED_EXP["shared experts"]
        L_T1_EXP["task_1 experts"]
        L_T2_EXP["task_T experts"]
    end
    subgraph "Extraction Layer 1"
        M_INPUT["input embeddings"]
        L1_SHARED["shared experts"]
        L1_T1["task_1 experts"]
        L1_T2["task_T experts"]
        L1_GATE_S["shared gate"]
        L1_GATE_T1["task_1 gate"]
        L1_GATE_T2["task_T gate"]
        L1_S_OUT["shared output"]
        L1_T1_OUT["task_1 output"]
        L1_T2_OUT["task_T output"]
    end
    I[feature_bags] --> M_INPUT
    M_INPUT --> L1_SHARED & L1_T1 & L1_T2
    M_INPUT --> L1_GATE_S & L1_GATE_T1 & L1_GATE_T2
    L1_SHARED & L1_GATE_S --> L1_S_OUT
    L1_SHARED & L1_T1 & L1_GATE_T1 --> L1_T1_OUT
    L1_SHARED & L1_T2 & L1_GATE_T2 --> L1_T2_OUT
    L1_S_OUT & L1_T1_OUT & L1_T2_OUT --> L_SHARED_EXP & L_T1_EXP & L_T2_EXP
    L_SHARED_EXP & L_SHARED_GATE --> L_SHARED_OUT
    L_SHARED_EXP & L_T1_EXP & L_T1_GATE --> L_T1_OUT
    L_SHARED_EXP & L_T2_EXP & L_T2_GATE --> L_T2_OUT
    L_T1_OUT --> T1 --> O1
    L_T2_OUT --> T2 --> O2

    style L1_SHARED fill:#f96,stroke:#333
    style L1_T1 fill:#9df,stroke:#333
    style L1_T2 fill:#df9,stroke:#333
    style L_SHARED_EXP fill:#f96,stroke:#333
    style L_T1_EXP fill:#9df,stroke:#333
    style L_T2_EXP fill:#df9,stroke:#333
```

### Progressive Separation

| Layer | Shared Experts | Task-Specific Experts | Gate Selection |
|-------|---------------|----------------------|----------------|
| 1 | High overlap | Few | Both shared & specific |
| L | Separated | Many | Task-specific focus |

## Multi-Task Model Comparison

| Model | Layers | Expert Separation | Gate |
|-------|--------|-------------------|------|
| Shared-Bottom | 1 | None | None |
| MMoE | 1 | Shared only | Per task |
| **PLE** | **L** | **Shared + per-task per layer** | **Per task (shared ⊕ specific)** |

## Configuration

```yaml
mlp:
  num_layers: 2
  num_shared_experts: 4
  num_specific_experts: 2
  expert_hidden: [64]
  tower_hidden: [32]
  num_tasks: 2
  task_names: [rating, click]
```

## Launch

```bash
python -m gerbil_train.cli.22-ple_train --config configs/22-ple/experiment.yaml
```

## References

- Tang, H., et al. "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning Model for Personalized Recommendations." RecSys 2020.
