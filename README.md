# gerbil-train

**Offline training and evaluation for GERBIL recommender systems.**

`gerbil-train` is the offline training component of GERBIL. Its primary model is **GwEN (Group-wise Embedding Network)**, a multi-class classification architecture for item recommendation. The project is designed with production ML principles: config-driven, fully reproducible runs, pluggable losses, and first-class feature management.

---

## Highlights

### 1. Config-Driven, Reproducible Runs

Every experiment produces a timestamped directory containing the model checkpoint, training curves, and a full snapshot of all configuration files:

```
checkpoints/gwen_ml1m_tfrecord/20260615220526/
├── best_model.pth              # checkpoint with best monitored metric
├── training_curves.png         # loss + metric plots
├── training_curves_loss.txt    # per-epoch loss values
├── training_curves_metric.txt  # per-epoch metric values
├── experiment.yaml             # experiment assembly config
├── data.yaml                   # data pipeline config
├── model.yaml                  # model architecture config
└── train.yaml                  # training hyper-parameters
```

Configurations are plain YAML – no hardcoded paths, no magic strings. All parameters are validated through `@dataclass` objects, giving IDE autocompletion and type safety.

### 2. Seamless Feature Ablation

Each feature has an `enabled` flag. Disabled features are excluded from both the data pipeline (TFRecord parsing) and the model (EmbeddingBag construction). No code changes needed.

```yaml
fields:
  user_movie_rate:
    f_index: 301
    f_type: 1
    vocab_size: 3569
    emb_dim: 16
    enabled: false   # ← toggle off for ablation
```

This design makes it trivial to test hypotheses about feature importance, detect label leakage, and evaluate minimal feature sets.

### 3. Unified Continuous & Categorical Feature Handling

All feature types go through the same `nn.EmbeddingBag` mechanism:

- **Categorical** (`field_type=1`): token index → embedding lookup, weight=1.0
- **Continuous** (`field_type=0`): position index → embedding lookup, weight = `(raw_value - mean) / std` (z-score normalized)

The normalization uses per-bucket `mean`/`std` from `pos_map.json`, making the continuous value embedding equivalent to `Linear(1, emb_dim)` projection with learned scale.

### 4. Pluggable Loss Functions

Three loss types interchangeable via a single config line:

```yaml
loss:
  type: ce                    # ce | nce | sampled_softmax
  num_sampled: 100            # only used for nce / sampled_softmax
```

All three losses train the model's own `nn.Linear` head directly – no separate class embeddings, no weight copying, no architectural changes. This means switching loss during evaluation is seamless: the same `model.forward()` produces correct full softmax logits regardless of which loss was used during training.

| Loss | Computation | Best for |
|------|-------------|----------|
| Cross-Entropy | logits over all `target_size` classes | Small-to-medium vocabularies |
| NCE | binary classification: signal vs noise | Large vocabularies, fast convergence |
| Sampled Softmax | multi-class over `1 + num_sampled` classes | Large vocabularies, stable training |

At initialization (random weights), the NCE loss can be estimated analytically:

| Variable | Value | Derivation |
|----------|-------|------------|
| `scores` | `≈ N(0, σ²)` | random Xavier init |
| `log(K / C)` | `≈ -3.61` | `K=100, C=3706` |
| BCE(signal) | `≈ 0.03` | `log(1 + exp(3.61))` for label=1 |
| BCE(noise) | `≈ 3.61` | `log(1 + exp(-3.61))` for label=0 |
| **Initial loss** | **≈ 3.57** | `(0.03 + 100 × 3.61) / 101` |

This matches the observed initial loss values and confirms the implementation is numerically correct.

### 5. Field-Level Attention (Optional)

Each field gets a learned `Linear(emb_dim, 1)` score. Scores are softmax-normalized across fields and used to reweight embeddings before concatenation. This lets the model dynamically emphasize informative fields and suppress noise – though in practice, with well-engineered features, uniform weighting often performs equally well.

### 6. Clean Architecture Separation

```
TFRecord → Dataset → Collator → Batch          [data pipeline]
                                  ↓
                          GwEN.forward()         [model]
                                  ↓
            CE / NCE / SampledSoftmax Loss       [loss function]
                                  ↓
                          GwENTrainer.fit()      [training loop]
```

Each layer is independently testable, replaceable, and config-driven.

---

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

### Train GwEN

```bash
python -m gerbil_train.cli.gwen_train \
  --config configs/experiment/gwen_ml1m_multiclass.yaml
```

### Switch Loss

```bash
# Edit configs/train/gwen_multiclass_trainer.yaml
loss:
  type: sampled_softmax     # ← change here
  num_sampled: 50

# Run (no code changes)
python -m gerbil_train.cli.gwen_train \
  --config configs/experiment/gwen_ml1m_multiclass.yaml
```

---

## Repository Structure

```bash
gerbil_train/
├── cli/            # training entry points
│   └── gwen_train.py
├── config.py       # dataclass configuration
├── data/           # TFRecord datasets and collators
│   └── gwen_tfrecord_dataset.py
├── losses/         # loss functions
│   ├── classification.py  # CE, NCE, SampledSoftmax
│   └── ranking.py
├── metrics/        # evaluation metrics
├── models/         # model architectures
│   └── gwen.py
├── trainer/        # training loops
│   ├── base_trainer.py
│   └── gwen_trainer.py
└── utils/          # helpers
```

---

## Configuration Layout

```bash
configs/
├── data/
│   └── ml1m_multiclass_tfrecord.yaml
├── model/
│   └── gwen_multiclass_model.yaml
├── train/
│   └── gwen_multiclass_trainer.yaml
└── experiment/
    └── gwen_ml1m_multiclass.yaml
```

---

## Dependencies

- Python 3.9+
- PyTorch 2.2+
- `tfrecord` — Python TFRecord reader
- Others: see `requirements.txt`

---

## Related Projects

- `gerbil-data` — data processing and sample generation
- `gerbil-serving` — online inference and model serving
