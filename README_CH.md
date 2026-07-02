# gerbil-train

**GERBIL 推荐系统的离线训练与评估组件。**

**gerbil-train** 是 **GERBIL**（General Efficient Recommender for Benchmarking, Inference, and Learning）系统的离线训练与评估组件，提供配置驱动的、可复现的推荐模型训练与评估。基于 Python 和 PyTorch。

- **[gerbil-data](https://github.com/shardzhang/gerbil-data)** — 基于 Spark 的特征工程数据处理
- **[gerbil-serving](https://github.com/shardzhang/gerbil-serving)** — 在线推理与模型服务

## 支持的模型

| 模型 | 类型 | 说明 |
|------|------|------|
| **GwEN**（分组嵌入网络） | 多分类 | 基础推荐架构，EmbeddingBag + 可选 field 级注意力 + MLP |
| **GwEN 二分类** | CTR | 二分类变体，sigmoid 输出 |
| **FM**（因子分解机） | CTR | Linear(一阶) + FM(二阶交叉)，无 Deep MLP |
| **DeepFM**（深度因子分解机） | CTR | Linear + FM + Deep 共享特征嵌入，per-field wide/deep 控制 |
| **xDeepFM**（极深因子分解机） | CTR | Linear + **CIN**（压缩交互网络）+ Deep，显式多阶向量级特征交叉 |
| **Wide & Deep** | CTR | Linear(Wide) + MLP(Deep)，per-field wide/deep 控制 |
| **DIN**（深度兴趣网络） | 序列推荐 | LocalActivationUnit 行为序列注意力 |
| **DIEN**（深度兴趣演化网络） | 序列推荐 | GRU + AUGRU 行为演化建模 |
| **YouTubeDNN** | 多分类 | Behavior `mode="mean"`，example age，`encode()` 推理 |
| **FTRL** | CTR | FTRL-Proximal 在线学习优化器 |
| **双塔模型**（Two-Tower） | 检索 | 两阶段训练（隐式 + 显式） |
| **Learning-to-Rank** | 排序 | 多种排序损失（LambdaRank, RankNet 等） |

## 项目亮点

### 1. 配置驱动，完全可复现

每次实验自动生成时间戳目录，包含检查点、训练曲线和完整配置快照。

### 2. Per-Field Tower 控制（W&D / DeepFM）

每个字段可独立配置进入 Wide（线性）还是 Deep（MLP）塔：

```yaml
user_id:   {wide: true,  deep: false}   # ID → 记忆
user_rate: {wide: false, deep: true}    # 统计 → 泛化
```

### 3. 特征级启用/禁用开关

每个字段带 `enabled` 开关，零代码消融实验。

### 4. 可插拔损失函数

```yaml
loss:
  type: sampled_softmax     # ce | nce | sampled_softmax
```

### 5. Step 级 LR Scheduling（`warmup_exp_decay` / `warmup_cos_decay`）

```yaml
scheduler:
  type: warmup_exp_decay       # warmup_exp_decay | warmup_cos_decay | none
  warmup_steps: 5000
  decay_rate: -0.333            # warmup_exp_decay 使用
  total_steps: 100000           # warmup_cos_decay 使用（总训练步数）
  learning_rate_min: 1e-7
```

| 类型 | LR 变化曲线 |
|------|------------|
| `warmup_exp_decay` | 线性 warmup → 指数衰减 |
| `warmup_cos_decay` | 线性 warmup → cosine 衰减（更平滑） |
| `none` | 固定 LR |

### 6. FTRL 优化器

支持 per-coordinate 学习率和精确 L1 稀疏的 FTRL-Proximal 优化器。

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
# CTR 模型
python -m gerbil_train.cli.5-deepfm_train        --config configs/5-deepfm/experiment.yaml
python -m gerbil_train.cli.5-xdeepfm_train       --config configs/5-xdeepfm/experiment.yaml
python -m gerbil_train.cli.4-wide_and_deep_train --config configs/4-wide_and_deep/experiment.yaml
python -m gerbil_train.cli.7-ftrl_train          --config configs/7-ftrl/experiment.yaml

# 序列模型
python -m gerbil_train.cli.7-din_train           --config configs/7-din/experiment.yaml
python -m gerbil_train.cli.7-dien_train          --config configs/7-dien/experiment.yaml

# 多分类模型
python -m gerbil_train.cli.8-youtube_dnn_train   --config configs/8-youtube_dnn/experiment.yaml
```

### 离线推理

```bash
python -m gerbil_train.cli.inference \
  --config configs/2-gwen_ml1m_binary/experiment.yaml \
  --checkpoint checkpoints/.../best_model.pth \
  --model-type gwen_binary \
  --split test \
  --output predictions.tsv
```

## 仓库结构

```bash
gerbil_train/
├── cli/           # 训练入口（12 个文件）
├── config/        # 配置数据类
├── data/          # TFRecord 数据管道
├── inference/     # 离线推理
├── losses/        # 损失函数（CE/NCE/SampledSoftmax + 排序损失）
├── metrics/       # 评估指标（AUC/GAUC/MAP/MRR/NDCG）
├── models/        # 12 个模型（GwEN/FM/DeepFM/xDeepFM/W&D/DIN/DIEN/YouTubeDNN/FTRL/双塔/LTR）
├── optimizers/    # FTRL-Proximal 优化器
├── trainer/       # 12 个训练器（共享 binary/multi 基类）
└── utils/         # 辅助工具
```

## 文档

各模型的详细文档在 `docs/` 目录：

| 文档 | 说明 |
|------|------|
| `docs/gwen.md` | GwEN 架构、公式、配置 |
| `docs/deepfm.md` | DeepFM Linear + FM + Deep |
| `docs/xdeepfm.md` | xDeepFM Linear + CIN + Deep |
| `docs/din.md` | DIN 注意力机制和兴趣池化 |
| `docs/dien.md` | DIEN GRU + AUGRU |
| `docs/ftrl.md` | FTRL 在线学习算法 |

## 项目状态

**活跃开发中**，66 个单元测试全部通过。

## 关联项目（GERBIL 生态）

- [`gerbil-data`](https://github.com/shardzhang/gerbil-data) — 基于 Spark 的特征工程
- [`gerbil-serving`](https://github.com/shardzhang/gerbil-serving) — 在线推理与模型服务
