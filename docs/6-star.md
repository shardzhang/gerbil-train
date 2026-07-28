# STAR (Star Topology Adaptive Recommender)

## Model Architecture

STAR uses a **star topology** neural network for multi-scenario CTR:

```
W_eff = W_shared ⊙ W_scenario
b_eff = b_shared + b_scenario
```

```mermaid
graph TB
    subgraph "Output"
        OUT["sigmoid"]
    end
    subgraph "Star MLP"
        L1_W["W = W_shared ⊙ W_scenario₁<br/>b = b_shared + b_scenario₁"]
        L1["StarFC + ReLU"]
        L2_W["W = W_shared ⊙ W_scenario₂<br/>b = b_shared + b_scenario₂"]
        L2["StarFC"]
    end
    subgraph "Scenario Selection"
        DOMAIN["domain_field (e.g. context_time_area)"]
        SID["scenario_id ∈ {0..K-1}"]
    end
    subgraph "Input"
        EMB["Concat all field embeddings"]
    end
    I[feature_bags] --> EMB
    I --> DOMAIN --> SID
    EMB --> L1
    SID --> L1_W & L2_W
    L1_W --> L1 --> L2_W --> L2 --> OUT

    style L1_W fill:#f96,stroke:#333
    style L2_W fill:#fc9,stroke:#333
    style SID fill:#cfc,stroke:#333
```

### Parameter Composition

```
Layer 1 (shared):    W_shared_1, b_shared_1
Layer 1 (scenario):  W_scenario_1[s], b_scenario_1[s]
Layer 1 (effective): W = W_shared_1 ⊙ W_scenario_1[s]
                     b = b_shared_1 + b_scenario_1[s]
```

## Configuration

```yaml
mlp:
  domain_field: context_time_area   # scenario indicator
  hidden_dims: [128, 64]
  num_scenarios: 7                  # K scenarios
```

## Launch

```bash
python -m gerbil_train.cli.6-star_train --config configs/6-star/experiment.yaml
```

## References

- Sheng, X., et al. "STAR: Star Topology Adaptive Recommender." WSDM 2022.
