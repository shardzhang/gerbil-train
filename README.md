# gerbil-train

**Offline training and evaluation for GERBIL recommender systems.**

**gerbil-train** is the offline training component of **GERBIL** (**G**eneral **E**fficient **R**ecommender for **B**enchmarking, **I**nference, and **L**earning). It provides config-driven, reproducible training and evaluation for multiple recommendation model families. Built with Python and PyTorch.

- **[gerbil-data](https://github.com/shardzhang/gerbil-data)** — Spark-based feature engineering and data processing
- **[gerbil-serving](https://github.com/shardzhang/gerbil-serving)** — Online inference and model serving

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| **GwEN** (Group-wise Embedding Network) | Multiclass | Base architecture for item recommendation. EmbeddingBag per field + optional field-level attention + MLP. |
| **GwEN Binary** | CTR | Binary classification variant with sigmoid output for rating/click prediction. |
| **DIN** (Deep Interest Network) | Sequential | Behavior-sequence attention via LocalActivationUnit. Supports multi-behavior and multi-target fields. |
| **DeepFM** | CTR | Deep Factorization Machine: Linear + FM (pair-wise) + Deep (MLP) terms sharing feature embeddings. |
| **Shared-Bottom Two-Tower** | Retrieval | Two-stage training (implicit pre-train + explicit fine-tune) for query-item retrieval. |
| **Learning-to-Rank** | Ranking | Feed-forward network with configurable losses (LambdaRank, RankNet, ListNet, ListMLE). |

## Highlights

### 1. Config-Driven, Reproducible Runs

Every experiment produces a timestamped run directory with model checkpoint, training curves, and config snapshots:

```
checkpoints/gwen_ml1m_multiclass/20260615220526/
├── best_model.pth
├── training_curves_loss.png / .txt
├── training_curves_metric.png / .txt
├── experiment.yaml, data.yaml, model.yaml, train.yaml
├── profile.txt         # per-epoch time and steps/s
└── exp.log             # full training log
```

All parameters are plain YAML, validated through `@dataclass` objects with IDE type safety.

### 2. Feature Ablation

Each feature has an `enabled` flag. Disabled fields are excluded from both data pipeline and model — no code changes needed.

```yaml
fields:
  user_movie_rate:
    field_index: 101
    field_type: 1
    dim: 3579
    emb_size: 16
    enabled: false      # toggle off for ablation
```

### 3. Unified Feature Handling

Categorical (`field_type=1`) and continuous (`field_type=0`) features both go through `nn.EmbeddingBag`:

- **Categorical**: token index → embedding lookup
- **Continuous**: position index → embedding lookup with z-score normalized weights

Continuous features also support `concat_type: "direct"` to skip embedding and pass raw values directly into the deep network.

### 4. Pluggable Loss Functions

Multi-class models support three losses interchangeable via a single config line:

```yaml
loss:
  type: ce                      # ce | nce | sampled_softmax
  num_sampled: 100              # only used for nce / sampled_softmax
```

All three losses train the model's own `nn.Linear` head — no separate class embeddings, no weight copying.

### 5. Sample-Level Shuffle

`TFRecordDataset` is an `IterableDataset`. A shuffle buffer provides sample-level randomization:

```yaml
data:
  batch_size: 512
  shuffle_buffer: 8192         # ≈ 16× batch size
```

### 6. Clean Architecture

```
TFRecord → Dataset → Collator → Batch          [data pipeline]
                                  ↓
                    Model.forward()              [model]
                                  ↓
                 Loss Function                   [loss]
                                  ↓
                    Trainer.fit()                [training loop]
```

Each layer is independently testable, replaceable, and config-driven.

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Layout

Data must be pre-processed by `gerbil-data` into TFRecord format:

```
data_root/
├── pos_map.txt            # feature definitions
├── pos_map.json           # target mapping, vocab stats
├── train/tfrecord/        # training shards
├── val/tfrecord/          # validation shards
└── test/tfrecord/         # test shards
```

### Train a Model

```bash
# Train GwEN binary (CTR)
python -m gerbil_train.cli.gwen_binary_train \
  --config configs/2-gwen_ml1m_binary/experiment.yaml

# Train DeepFM (CTR)
python -m gerbil_train.cli.deepfm_train \
  --config configs/4-deepfm/experiment.yaml

# Train DIN (sequential)
python -m gerbil_train.cli.din_train \
  --config configs/3-din/experiment.yaml

# Train GwEN multiclass (recommendation)
python -m gerbil_train.cli.gwen_multiclass_train \
  --config configs/1-gwen_ml1m_multiclass/experiment.yaml
```

### Offline Inference

```bash
python -m gerbil_train.cli.inference \
  --config configs/2-gwen_ml1m_binary/experiment.yaml \
  --checkpoint checkpoints/gwen_ml1m_binary/20260624.../best_model.pth \
  --model-type gwen_binary \
  --split test \
  --output predictions.tsv
```

## Repository Structure

```bash
gerbil_train/
├── cli/                    # Training and inference entry points
│   ├── 1-gwen_multiclass_train.py
│   ├── 2-gwen_binary_train.py
│   ├── 3-din_train.py
│   ├── 4-deepfm_train.py
│   ├── 5-shared_bottom_two_tower_train.py
│   ├── 6-learning_to_rank_train.py
│   └── inference.py
├── config/                 # Dataclass configuration objects
│   ├── model_config.py     # BaseModelConfig, DINModelConfig, DeepFMModelConfig
│   └── train_config.py     # TrainConfig, TrainDataConfig, etc.
├── data/                   # TFRecord datasets and collators
│   └── tfrecord_dataset.py
├── inference/              # Offline predictor
│   ├── predictor.py
│   └── result_writer.py
├── losses/                 # Loss functions
│   ├── classification.py  # CE, NCE, SampledSoftmax
│   └── ranking.py         # LambdaRank, RankNet, ListNet, ListMLE
├── metrics/                # Evaluation metrics
│   ├── classification.py  # AUC, GAUC, MAP, MRR, HitRate
│   └── ranking.py         # NDCG@K
├── models/                 # Model architectures
│   ├── base_model.py      # Abstract base class
│   ├── gwen.py            # GwEN binary + multiclass
│   ├── din.py             # Deep Interest Network
│   ├── deepfm.py          # Deep Factorization Machine
│   ├── shared_bottom_two_tower.py
│   ├── learning_to_rank.py
│   └── layers.py          # Shared layers (FullyConnectedLayer, Dice, etc.)
├── trainer/                # Training loops
│   ├── base_trainer.py
│   ├── binary_trainer.py      # Shared binary trainer (GwEN/DIN/DeepFM)
│   ├── multi_trainer.py       # Shared multi-class trainer (GwEN)
│   ├── gwen_binary_trainer.py
│   ├── gwen_multiclass_trainer.py
│   ├── din_trainer.py
│   ├── deepfm_trainer.py
│   ├── shared_bottom_two_tower_trainer.py
│   └── learning_to_rank_trainer.py
└── utils/                  # Helpers
    ├── config.py           # YAML loading
    ├── run.py              # Run directory management
    ├── training.py         # Shared dataloader/model config builders
    ├── embedding.py        # Embedding helpers
    ├── nn.py               # Model summary, parameter counting
    ├── plot.py             # Training curve plotting
    └── inspect.py          # Batch inspector
```

## Configuration Layout

```bash
configs/
├── 0-data/                     # Shared data configs
│   └── ml1m_binary_tfrecord.yaml
├── 1-gwen_ml1m_multiclass/     # GwEN multiclass experiment
│   ├── experiment.yaml
│   ├── model.yaml
│   └── trainer.yaml
├── 2-gwen_ml1m_binary/         # GwEN binary (CTR) experiment
│   ├── experiment.yaml
│   ├── model.yaml
│   └── trainer.yaml
├── 3-din/                      # DIN experiment
│   ├── experiment.yaml
│   ├── model.yaml
│   └── trainer.yaml
├── 4-deepfm/                   # DeepFM experiment
│   ├── experiment.yaml
│   ├── model.yaml
│   └── trainer.yaml
├── 5-ltr/                      # Learning-to-rank experiment
│   └── learning_to_rank_*.yaml
└── build_model_config.py       # Helper to generate model YAML from pos_map.txt
```

## Dependencies

- Python 3.9+
- PyTorch 2.2+
- `tfrecord` — Python TFRecord reader
- Others: see `requirements.txt`

## Project Status & Quality

gerbil-train is in **active development** (~2 months, 15+ commits, single contributor). Code review score: **3.5 / 5.0**.

| Dimension | Score |
|-----------|:-----:|
| Architecture | 4/5 |
| Type Annotations | 4/5 |
| Code Complexity | 4/5 |
| Documentation | 3/5 |
| Error Handling | 3/5 |
| Testing | 3/5 |
| Engineering | 2/5 |

### What's solid

- Modular, config-driven architecture (data → model → loss → trainer)
- Fully reproducible experiment runs with timestamped artifacts
- Type-safe configuration via `@dataclass`
- Feature-level enabled/disabled toggle for ablation studies
- Pluggable loss functions with mathematical derivation
- Shared base trainers eliminate code duplication across models
- Complete offline inference pipeline

### What needs work

- **Testing**: More model coverage, CI/CD pipeline
- **Documentation**: API reference, contributing guide
- **Dependency management**: Lock version ranges, dev dependencies
- **Community infrastructure**: Issue/PR templates, GitHub Actions

## Related Projects (GERBIL Ecosystem)

- [`gerbil-data`](https://github.com/shardzhang/gerbil-data) — Spark-based feature engineering and data processing
- [`gerbil-serving`](https://github.com/shardzhang/gerbil-serving) — Online inference and model serving
