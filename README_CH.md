# gerbil-train

**GERBIL 推荐系统的离线训练与评估组件。**

**gerbil-train** 是 **GERBIL**（General Efficient Recommender for Benchmarking, Inference, and Learning）系统的离线训练与评估组件，提供配置驱动的、可复现的推荐模型训练与评估。基于 Python 和 PyTorch。

- **[gerbil-data](https://github.com/shardzhang/gerbil-data)** — 基于 Spark 的特征工程数据处理
- **[gerbil-serving](https://github.com/shardzhang/gerbil-serving)** — 在线推理与模型服务

## 支持的模型

| 模型 | 类型 | 说明 |
|------|------|------|
| **GwEN**（分组嵌入网络） | 多分类 | 基础推荐架构。每个 field 使用 EmbeddingBag + 可选 field 级注意力 + MLP |
| **GwEN 二分类** | CTR | 二分类变体，sigmoid 输出 |
| **DIN**（深度兴趣网络） | 序列推荐 | LocalActivationUnit 实现行为序列注意力，支持多行为+多目标字段 |
| **DeepFM**（深度因子分解机） | CTR | Linear + FM（二阶交叉）+ Deep（MLP）共享特征嵌入 |
| **双塔模型**（Shared-Bottom Two-Tower） | 检索 | 两阶段训练（隐式预训练 + 显式精调） |
| **Learning-to-Rank** | 排序 | 简单全连接网络，支持多种排序损失 |

## 项目亮点

### 1. 配置驱动，完全可复现

每次实验自动生成时间戳目录，保存模型检查点、训练曲线和完整的配置文件快照：

```
checkpoints/gwen_ml1m_multiclass/20260615220526/
├── best_model.pth
├── training_curves_loss.png / .txt
├── training_curves_metric.png / .txt
├── experiment.yaml, data.yaml, model.yaml, train.yaml
├── profile.txt         # 每 epoch 耗时和 steps/s
└── exp.log             # 完整训练日志
```

所有参数通过 `@dataclass` 验证，IDE 自动补全和类型安全。

### 2. 特征级启用/禁用开关

每个字段带 `enabled` 开关，禁用后数据管道和模型同时排除，零代码改动做消融实验：

```yaml
fields:
  user_movie_rate:
    field_index: 101
    field_type: 1
    dim: 3579
    emb_size: 16
    enabled: false      # 关闭以验证特征重要性
```

### 3. 连续值与分类值统一建模

分类特征（`field_type=1`）和连续特征（`field_type=0`）走同一套 `nn.EmbeddingBag`：

- **分类特征**：token ID → embedding 查表
- **连续特征**：桶 ID → embedding 查表 + z-score 标准化权重

连续特征还支持 `concat_type: "direct"`，跳过 embedding 直接将原始值拼入深度网络。

### 4. 可插拔损失函数

三种损失通过一行配置切换：

```yaml
loss:
  type: ce                      # ce | nce | sampled_softmax
  num_sampled: 100              # 仅 nce / sampled_softmax 时生效
```

所有损失直接训练模型的 `nn.Linear` 头，无需单独维护类别嵌入。

### 5. 清晰的架构分层

```
TFRecord → Dataset → Collator → Batch          [数据管道]
                                  ↓
                    Model.forward()              [模型]
                                  ↓
                 Loss Function                   [损失函数]
                                  ↓
                    Trainer.fit()                [训练循环]
```

每层独立可测试、可替换、受配置驱动。

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

### 数据目录结构

数据需由 `gerbil-data` 预处理好为 TFRecord 格式：

```
data_root/
├── pos_map.txt            # 特征定义
├── pos_map.json           # target 映射、词汇统计
├── train/tfrecord/        # 训练集分片
├── val/tfrecord/          # 验证集分片
└── test/tfrecord/         # 测试集分片
```

### 训练模型

```bash
# 训练 GwEN 二分类（CTR）
python -m gerbil_train.cli.gwen_binary_train \
  --config configs/2-gwen_ml1m_binary/experiment.yaml

# 训练 DeepFM（CTR）
python -m gerbil_train.cli.deepfm_train \
  --config configs/4-deepfm/experiment.yaml

# 训练 DIN（序列推荐）
python -m gerbil_train.cli.din_train \
  --config configs/3-din/experiment.yaml

# 训练 GwEN 多分类（推荐）
python -m gerbil_train.cli.gwen_multiclass_train \
  --config configs/1-gwen_ml1m_multiclass/experiment.yaml
```

### 离线推理

```bash
python -m gerbil_train.cli.inference \
  --config configs/2-gwen_ml1m_binary/experiment.yaml \
  --checkpoint checkpoints/gwen_ml1m_binary/20260624.../best_model.pth \
  --model-type gwen_binary \
  --split test \
  --output predictions.tsv
```

## 仓库结构

```bash
gerbil_train/
├── cli/                    # 训练和推理入口
│   ├── 1-gwen_multiclass_train.py
│   ├── 2-gwen_binary_train.py
│   ├── 3-din_train.py
│   ├── 4-deepfm_train.py
│   ├── 5-shared_bottom_two_tower_train.py
│   ├── 6-learning_to_rank_train.py
│   └── inference.py
├── config/                 # 配置数据类
│   ├── model_config.py     # BaseModelConfig, DINModelConfig, 等
│   └── train_config.py     # TrainConfig, TrainDataConfig, 等
├── data/                   # TFRecord 数据集和 collator
│   └── tfrecord_dataset.py
├── inference/              # 离线预测
│   ├── predictor.py
│   └── result_writer.py
├── losses/                 # 损失函数
│   ├── classification.py  # CE, NCE, SampledSoftmax
│   └── ranking.py         # LambdaRank, RankNet, ListNet, ListMLE
├── metrics/                # 评估指标
│   ├── classification.py  # AUC, GAUC, MAP, MRR, HitRate
│   └── ranking.py         # NDCG@K
├── models/                 # 模型结构
│   ├── base_model.py      # 抽象基类
│   ├── gwen.py            # GwEN 二分类 + 多分类
│   ├── din.py             # 深度兴趣网络
│   ├── deepfm.py          # 深度因子分解机
│   ├── shared_bottom_two_tower.py
│   ├── learning_to_rank.py
│   └── layers.py          # 共享层（FullyConnectedLayer, Dice 等）
├── trainer/                # 训练循环
│   ├── base_trainer.py
│   ├── binary_trainer.py      # 共享二分类训练器 (GwEN/DIN/DeepFM)
│   ├── multi_trainer.py       # 共享多分类训练器 (GwEN)
│   ├── gwen_binary_trainer.py
│   ├── gwen_multiclass_trainer.py
│   ├── din_trainer.py
│   ├── deepfm_trainer.py
│   ├── shared_bottom_two_tower_trainer.py
│   └── learning_to_rank_trainer.py
└── utils/                  # 辅助工具
    ├── config.py           # YAML 加载和解析
    ├── run.py              # 运行目录管理
    ├── training.py         # 共享数据加载器和模型配置构建
    ├── embedding.py        # Embedding 辅助函数
    ├── nn.py               # 模型摘要和参数统计
    ├── plot.py             # 训练曲线绘制
    └── inspect.py          # Batch 检查器
```

## 配置目录结构

```bash
configs/
├── 0-data/                     # 共享数据配置
│   └── ml1m_binary_tfrecord.yaml
├── 1-gwen_ml1m_multiclass/     # GwEN 多分类实验
│   ├── experiment.yaml
│   ├── model.yaml
│   └── trainer.yaml
├── 2-gwen_ml1m_binary/         # GwEN 二分类实验
│   ├── experiment.yaml
│   ├── model.yaml
│   └── trainer.yaml
├── 3-din/                      # DIN 实验
│   ├── experiment.yaml
│   ├── model.yaml
│   └── trainer.yaml
├── 4-deepfm/                   # DeepFM 实验
│   ├── experiment.yaml
│   ├── model.yaml
│   └── trainer.yaml
├── 5-ltr/                      # Learning-to-rank 实验
│   └── learning_to_rank_*.yaml
└── build_model_config.py       # 从 pos_map.txt 生成模型 YAML 的辅助工具
```

## 依赖

- Python 3.9+
- PyTorch 2.2+
- `tfrecord` — Python TFRecord 读取器
- 其他：见 `requirements.txt`

## 项目状态与质量

gerbil-train 处于**活跃开发阶段**（约 2 个月，15+ 次提交，单人维护）。代码审查评分 **3.5 / 5.0**。

| 维度 | 评分 |
|------|:----:|
| 架构设计 | 4/5 |
| 类型注解 | 4/5 |
| 代码复杂度 | 4/5 |
| 文档 | 3/5 |
| 错误处理 | 3/5 |
| 测试 | 3/5 |
| 工程实践 | 2/5 |

### 优势

- 模块化配置驱动架构，四层解耦清晰
- 时间戳实验产物 + 完整配置快照，运行完全可复现
- `@dataclass` 配置，IDE 自动补全
- 特征级 enabled/disabled 开关
- 可插拔损失函数，数学推导已记录
- 共享基类训练器，消除模型间代码重复
- 完整离线推理管线

### 待改进

- **测试体系**：更多模型覆盖，CI/CD 自动化
- **文档**：API 参考、贡献指南
- **依赖管理**：锁定版本范围
- **社区基础设施**：Issue/PR 模板、GitHub Actions

## 关联项目（GERBIL 生态）

- [`gerbil-data`](https://github.com/shardzhang/gerbil-data) — 基于 Spark 的特征工程数据处理
- [`gerbil-serving`](https://github.com/shardzhang/gerbil-serving) — 在线推理与模型服务
