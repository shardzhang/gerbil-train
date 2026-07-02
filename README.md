# gerbil-train

**Offline training and evaluation for GERBIL recommender systems.**

**gerbil-train** is the offline training component of **GERBIL** (**G**eneral **E**fficient **R**ecommender for **B**enchmarking, **I**nference, and **L**earning). It provides config-driven, reproducible training and evaluation for multiple recommendation model families. Built with Python and PyTorch.

- **[gerbil-data](https://github.com/shardzhang/gerbil-data)** — Spark-based feature engineering and data processing
- **[gerbil-serving](https://github.com/shardzhang/gerbil-serving)** — Online inference and model serving

## Supported Models

| Model | Type | Description |
|-------|------|-------------|
| **FM** (Factorization Machine) | CTR | Linear (1st-order) + FM (2nd-order pair-wise) terms, no Deep MLP. |
| **FTRL** (Follow The Regularized Leader) | CTR | Linear model with FTRL-Proximal optimizer (per-coordinate LR + L1 sparsity). |
| **GwEN** (Group-wise Embedding Network) | Multiclass | EmbeddingBag per field + optional field-level attention + MLP for item recommendation. |
| **GwEN Binary** | CTR | Binary classification variant with sigmoid output. |
| **YouTubeDNN** | Multiclass | Behavior `mode="mean"`, example age, bias-free head, `encode()` for ANN serving. |
| **AFM** (Attentional FM) | CTR | FM with learned attention weights per feature pair via attention MLP. |
| **NFM** (Neural FM) | CTR | Bi-Interaction Pooling (k-dim vector) + Deep MLP. More parameter-efficient than DeepFM. |
| **PNN** (Product-based Neural Network) | CTR | Linear + Product Layer (pair-wise inner products) + MLP. |
| **Wide & Deep** | CTR | Linear (Wide) + MLP (Deep), per-field wide/deep control. |
| **DeepFM** | CTR | Linear + FM + Deep (MLP) sharing feature embeddings. Per-field wide/deep control. |
| **xDeepFM** | CTR | Linear + **CIN** (Compressed Interaction Network) + Deep. Explicit multi-order vector-wise interactions. |
| **DCN** (Deep & Cross Network) | CTR | Cross Network (explicit bounded-degree interactions) + Deep MLP. |
| **DCNv2** (Deep & Cross Network V2) | CTR | Full d×d matrix cross layers with optional low-rank approximation. |
| **FiBiNet** (Feature Importance & Bilinear Interaction) | CTR | SENET feature weighting + bilinear interaction + MLP. |
| **AutoInt** (Automatic Feature Interaction) | CTR | Multi-head self-attention (Transformer) over feature fields. Stacked interacting layers. |
| **DIEN** (Deep Interest Evolution Network) | Sequential | GRU interest extractor + AUGRU interest evolution. Auxiliary loss support. |
| **DSIN** (Deep Session Interest Network) | Sequential | Session division + Bi-LSTM + self-attention across sessions + attention pooling. |
| **MIMN** (Multi-channel Interest with Moment Network) | Sequential | Multi-slot memory network + Bi-LSTM + target-aware memory read. |
| **DIN** (Deep Interest Network) | Sequential | Behavior-sequence attention via LocalActivationUnit. Multi-behavior and multi-target support. |
| **Shared-Bottom Two-Tower** | Retrieval | Two-stage training (implicit pre-train + explicit fine-tune). |
| **Learning-to-Rank** | Ranking | Feed-forward network with configurable losses (LambdaRank, RankNet, ListNet, ListMLE). |

## Highlights

### 1. Config-Driven, Reproducible Runs

Every experiment produces a timestamped run directory:

```
checkpoints/gwen_ml1m_multiclass/20260615220526/
├── best_model.pth
├── training_curves_loss.png / .txt
├── training_curves_metric.png / .txt
├── experiment.yaml, data.yaml, model.yaml, train.yaml
├── profile.txt
└── exp.log
```

All parameters are plain YAML, validated through `@dataclass` objects with IDE type safety.

### 2. Feature Ablation

Each feature has an `enabled` flag. Disabled fields are excluded from both data pipeline and model.

### 3. Unified Feature Handling

Categorical (`field_type=1`) and continuous (`field_type=0`) features both go through `nn.EmbeddingBag`. Continuous features also support `concat_type: "direct"` to pass raw values directly.

### 4. Per-Field Tower Control (W&D / DeepFM)

Each field can independently be assigned to Wide (linear) or Deep (MLP) towers:

```yaml
fields:
  user_id:   {wide: true,  deep: false}   # ID → memorize
  user_rate: {wide: false, deep: true}    # stats → generalize
```

### 5. Pluggable Loss Functions

Multi-class models support three losses interchangeable via a single config line:

```yaml
loss:
  type: ce                      # ce | nce | sampled_softmax
  num_sampled: 100
```

### 6. FTRL Optimizer

FTRL-Proximal optimizer with per-coordinate learning rates and exact L1 sparsity.

### 7. Step-Level LR Scheduling (`warmup_exp_decay` / `warmup_cos_decay`)

Three scheduler types via a single config line:

```yaml
scheduler:
  type: warmup_exp_decay       # warmup_exp_decay | warmup_cos_decay | none
  warmup_steps: 5000
  decay_rate: -0.333            # for warmup_exp_decay
  total_steps: 100000           # for warmup_cos_decay (total training steps)
  learning_rate_min: 1e-7
```

| Type | LR Schedule |
|------|------------|
| `warmup_exp_decay` | Linear warmup → exponential decay: `lr = base × exp(decay_rate × (step - warmup) / warmup)` |
| `warmup_cos_decay` | Linear warmup → cosine decay: `lr = lr_min + 0.5 × (base - lr_min) × (1 + cos(π × progress))` |
| `none` | Fixed learning rate, no scheduling |

### 8. Clean Architecture

```
TFRecord → Dataset → Collator → Batch    → Model.forward() → Loss → Trainer.fit()
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
├── train/tfrecord/
├── val/tfrecord/
└── test/tfrecord/
```

### Train a Model

```bash
# CTR Models
python -m gerbil_train.cli.2-gwen_binary_train --config configs/2-gwen_ml1m_binary/experiment.yaml
python -m gerbil_train.cli.5-deepfm_train      --config configs/5-deepfm/experiment.yaml
python -m gerbil_train.cli.5-xdeepfm_train     --config configs/5-xdeepfm/experiment.yaml
python -m gerbil_train.cli.4-wide_and_deep_train --config configs/4-wide_and_deep/experiment.yaml

python -m gerbil_train.cli.1-fm_train            --config configs/1-fm/experiment.yaml
python -m gerbil_train.cli.3-afm_train           --config configs/3-afm/experiment.yaml
python -m gerbil_train.cli.3-nfm_train           --config configs/3-nfm/experiment.yaml
python -m gerbil_train.cli.3-pnn_train           --config configs/3-pnn/experiment.yaml
python -m gerbil_train.cli.6-autoint_train       --config configs/6-autoint/experiment.yaml
python -m gerbil_train.cli.6-fibinet_train       --config configs/6-fibinet/experiment.yaml
python -m gerbil_train.cli.6-dcn_train           --config configs/6-dcn/experiment.yaml
python -m gerbil_train.cli.6-dcnv2_train         --config configs/6-dcnv2/experiment.yaml
python -m gerbil_train.cli.1-ftrl_train         --config configs/1-ftrl/experiment.yaml

# Sequential Models
python -m gerbil_train.cli.7-din_train          --config configs/7-din/experiment.yaml
python -m gerbil_train.cli.7-dien_train         --config configs/7-dien/experiment.yaml
python -m gerbil_train.cli.7-dsin_train         --config configs/7-dsin/experiment.yaml

# Multi-class Models
python -m gerbil_train.cli.2-gwen_multiclass_train --config configs/2-gwen_ml1m_multiclass/experiment.yaml
python -m gerbil_train.cli.2-youtube_dnn_train     --config configs/2-youtube_dnn/experiment.yaml
```

### Offline Inference

```bash
python -m gerbil_train.cli.inference \
  --config configs/2-gwen_ml1m_binary/experiment.yaml \
  --checkpoint checkpoints/.../best_model.pth \
  --model-type gwen_binary \
  --split test \
  --output predictions.tsv
```

## Repository Structure

```bash
gerbil_train/
├── cli/                    # Training and inference entry points (numbered)
│   ├── 2-gwen_multiclass_train.py / 2-gwen_binary_train.py
│   ├── 4-wide_and_deep_train.py / 5-deepfm_train.py / 5-xdeepfm_train.py
│   ├── 7-din_train.py / 7-dien_train.py
│   ├── 2-youtube_dnn_train.py
│   ├── 1-fm_train.py / 1-ftrl_train.py
│   ├── 99-shared_bottom_two_tower_train.py / 99-learning_to_rank_train.py
│   └── inference.py
├── config/                 # Dataclass configuration objects
│   ├── model_config.py     # BaseModelConfig, DINModelConfig, DeepFMModelConfig, etc.
│   └── train_config.py     # TrainConfig, TrainDataConfig, etc.
├── data/                   # TFRecord datasets and collators
│   └── tfrecord_dataset.py
├── inference/              # Offline predictor
│   ├── predictor.py
│   └── result_writer.py
├── losses/                 # Loss functions
│   ├── classification.py  # CE, NCE, SampledSoftmax
│   └── ranking.py
├── metrics/                # Evaluation metrics
│   ├── classification.py  # AUC, GAUC, MAP, MRR, HitRate
│   └── ranking.py
├── models/                 # Model architectures
│   ├── base_model.py      # Abstract base class
│   ├── gwen.py            # GwEN binary + multiclass
│   ├── fm.py              # Factorization Machine
│   ├── deepfm.py          # Deep Factorization Machine
│   ├── xdeepfm.py         # eXtreme Deep Factorization Machine
│   ├── wide_and_deep.py   # Wide & Deep
│   ├── din.py             # Deep Interest Network
│   ├── dien.py            # Deep Interest Evolution Network
│   ├── youtube_dnn.py     # YouTube DNN
│   ├── ftrl.py            # FTRL linear model
│   ├── shared_bottom_two_tower.py
│   ├── learning_to_rank.py
│   └── layers.py          # Shared layers
├── optimizers/             # Custom optimizers
│   └── ftrl.py            # FTRL-Proximal optimizer
├── trainer/                # Training loops
│   ├── base_trainer.py
│   ├── binary_trainer.py  # Shared binary trainer
│   ├── multi_trainer.py   # Shared multi-class trainer
│   ├── gwen_binary_trainer.py / gwen_multiclass_trainer.py
│   ├── din_trainer.py / dien_trainer.py
│   ├── deepfm_trainer.py / fm_trainer.py / xdeepfm_trainer.py
│   ├── wide_and_deep_trainer.py / ftrl_trainer.py
│   ├── youtube_dnn_trainer.py
│   ├── shared_bottom_two_tower_trainer.py
│   └── learning_to_rank_trainer.py
└── utils/
    ├── config.py / run.py / training.py
    ├── embedding.py / nn.py / plot.py / inspect.py
```

## Documentation

| Document | Description |
|----------|------------|
| `docs/2-gwen.md` | GwEN architecture, formulas, configuration |
| `docs/5-deepfm.md` | DeepFM Linear + FM + Deep |
| `docs/5-xdeepfm.md` | xDeepFM with CIN (Compressed Interaction Network) |
| `docs/7-din.md` | DIN attention mechanism and interest pooling |
| `docs/7-dien.md` | DIEN GRU + AUGRU |
| `docs/1-fm.md` | Factorization Machine |
| `docs/4-wide_and_deep.md` | Wide & Deep per-field tower control |
| `docs/2-youtube_dnn.md` | YouTube DNN with encode() for ANN serving |
| `docs/1-ftrl.md` | FTRL online learning algorithm |
| `docs/99-shared_bottom_two_tower.md` | Two-stage retrieval training |
| `docs/feature_leakage_ctr.md` | CTR feature leakage analysis |
| `docs/feature_leakage_in_multiclass.md` | Multi-class feature leakage analysis |

## Configuration Layout

```bash
configs/
├── 0-data/                     # Shared data configs
├── 2-gwen_ml1m_{binary,multiclass}/
├── 4-wide_and_deep/ 5-deepfm/ 5-xdeepfm/
├── 7-din/ 7-dien/
├── 2-youtube_dnn/
├── 1-ftrl/
├── 99-eval/ 99-ltr/ 99-sbtt/
└── build_model_config.py       # Helper to generate model YAML from pos_map.txt
```

## Dependencies

- Python 3.9+
- PyTorch 2.2+
- `tfrecord` — Python TFRecord reader
- Others: see `requirements.txt`

## Project Status & Quality

gerbil-train is in **active development** (50+ commits, single contributor). **100 unit tests**, all passing.

| Dimension | Score |
|-----------|:-----:|
| Architecture | 4/5 |
| Type Annotations | 4/5 |
| Code Complexity | 4/5 |
| Documentation | 3/5 |
| Error Handling | 3/5 |
| Testing | 3/5 |
| Engineering | **3/5** |

### What's solid

- Modular, config-driven architecture (data → model → loss → trainer)
- Fully reproducible experiment runs with timestamped artifacts
- Type-safe configuration via `@dataclass`
- Feature-level enabled/disabled toggle for ablation
- Pluggable loss functions with mathematical derivation
- Shared base trainers eliminate code duplication
- FTRL-Proximal optimizer with per-coordinate LR
- Per-field wide/deep tower control
- Step-level LR scheduling (warmup + exp/cosine decay)
- Complete offline inference pipeline
- Custom exception hierarchy (8 classes)
- CI/CD via GitHub Actions (Python 3.10/3.11, push/PR)

### What needs work

- **Documentation**: API reference, contributing guide
- **Community infrastructure**: Issue/PR templates
- **Code formatting**: .editorconfig, black/ruff config

## Related Projects (GERBIL Ecosystem)

- [`gerbil-data`](https://github.com/shardzhang/gerbil-data) — Spark-based feature engineering
- [`gerbil-serving`](https://github.com/shardzhang/gerbil-serving) — Online inference and model serving
