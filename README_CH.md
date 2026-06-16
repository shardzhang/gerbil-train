# gerbil-train

**GERBIL 推荐系统的离线训练与评估组件。**

**gerbil-train** 是 **GERBIL**（General Efficient Recommender for Benchmarking, Inference, and Learning）系统的离线训练与评估组件，专注于高效、模块化地训练、评估和导出推荐模型。项目采用 Python 开发，与以下组件形成完整链路：

- **[gerbil-data](https://github.com/shardzhang/gerbil-data)** — 基于 Spark 的特征工程数据处理
- **[gerbil-serving](https://github.com/shardzhang/gerbil-serving)** — 在线推理与模型服务

核心模型为 **GwEN（Group-wise Embedding Network，分组嵌入网络）**，一种用于物品推荐的多分类架构。项目同时支持 CTR 预测、Top-K 推荐、排序、检索预训练、序列推荐等任务。

**支持能力：**

- **任务类型：** CTR 预测、Top-K 推荐、排序、检索预训练、序列推荐
- **模型覆盖：** MF、FM、DeepFM、Wide & Deep、DIN、SASRec、双塔模型、GwEN 等
- **评估指标：** AUC、LogLoss、Recall@K、HitRate@K、NDCG@K、MRR

---

## 项目亮点

### 1. 配置驱动，完全可复现

每次实验自动生成时间戳目录，保存模型检查点、训练曲线和完整的配置文件快照：

```bash
checkpoints/gwen_ml1m_tfrecord/20260615220526/
├── best_model.pth              # 监控指标最优的检查点
├── training_curves.png         # 损失 + 指标曲线图
├── training_curves_loss.txt    # 逐 epoch 损失值
├── training_curves_metric.txt  # 逐 epoch 指标值
├── experiment.yaml             # 实验组装配置
├── data.yaml                   # 数据管道配置
├── model.yaml                  # 模型架构配置
└── train.yaml                  # 训练超参数配置
```

所有配置为纯 YAML，无硬编码路径，无魔数字段。通过 `@dataclass` 做参数校验，IDE 可自动补全和类型检查。

### 2. 特征级启用/禁用开关

每个特征都带 `enabled` 开关。禁用的特征会被数据管道（TFRecord 解析）和模型（EmbeddingBag 构造）同时排除，零代码改动即可做消融实验。

```yaml
fields:
  user_movie_rate:
    f_index: 301
    f_type: 1
    vocab_size: 3569
    emb_dim: 16
    enabled: false   # ← 关闭以验证特征重要性
```

这个设计使得检验特征重要性、排查标签泄漏、评估最小特征集变得极其简便。

### 3. 连续值与分类值统一建模

所有特征类型走同一套 `nn.EmbeddingBag` 机制：

- **分类特征**（`field_type=1`）：token ID → embedding 查表，权重=1.0
- **连续特征**（`field_type=0`）：桶 ID → embedding 查表，权重 = `(raw - mean) / std`（z-score 标准化）

标准化使用的 `mean`/`std` 来自 `pos_map.json` 中每个桶的统计量。这使得连续特征嵌入等价于 `Linear(1, emb_dim)` 投影 + 可学习缩放。

### 4. 可插拔损失函数

三种损失通过一行配置切换：

```yaml
loss:
  type: ce                    # ce | nce | sampled_softmax
  num_sampled: 100            # 仅 nce / sampled_softmax 时生效
```

所有损失直接训练模型的 `nn.Linear` 分类头——不需要单独维护类别嵌入，不需要权重拷贝，不需要改动模型结构。评估时 `model.forward()` 输出的完整 softmax logits 与训练方式无关，始终正确。

| 损失 | 计算方式 | 适用场景 |
|------|----------|----------|
| Cross-Entropy | 对所有 `target_size` 类计算 logits | 中小词汇量 |
| NCE | 二分类：区分信号 vs 噪声 | 大词汇量，收敛快 |
| Sampled Softmax | 在 `1 + num_sampled` 类上做 softmax | 大词汇量，训练稳定 |

初始化时（随机权重），NCE loss 的理论估算：

| 变量 | 值 | 推导 |
|------|----|------|
| `scores` | `≈ N(0, σ²)` | Xavier 随机初始化 |
| `log(K / C)` | `≈ -3.61` | `K=100, C=3706` |
| BCE(信号) | `≈ 0.03` | `log(1 + exp(3.61))`，label=1 |
| BCE(噪声) | `≈ 3.61` | `log(1 + exp(-3.61))`，label=0 |
| **初始 loss** | **≈ 3.57** | `(0.03 + 100 × 3.61) / 101` |

该估算与实际运行时的初始 loss 一致，验证了 NCE 实现的数值正确性。

### 5. 特征级注意力（可选）

每个 field 有一个可学习的 `Linear(emb_dim, 1)` 评分头，分数经 softmax 归一化后作为权重，拼接前对每个 field 的 embedding 做加权。这让模型能动态放大信息量大的特征、抑制噪声。在特征工程成熟的场景下，均匀加权的效果通常也不差。

### 6. 清晰的架构分层

```
TFRecord → Dataset → Collator → Batch          [数据管道]
                                  ↓
                          GwEN.forward()         [模型]
                                  ↓
            CE / NCE / SampledSoftmax Loss       [损失函数]
                                  ↓
                          GwENTrainer.fit()      [训练循环]
```

每层独立可测试、可替换、受配置驱动。

---

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

### 训练 GwEN

```bash
python -m gerbil_train.cli.gwen_train \
  --config configs/experiment/gwen_ml1m_multiclass.yaml
```

### 切换损失函数

```bash
# 编辑 configs/train/gwen_multiclass_trainer.yaml
loss:
  type: sampled_softmax     # ← 改这里
  num_sampled: 50

# 直接跑（不改代码）
python -m gerbil_train.cli.gwen_train \
  --config configs/experiment/gwen_ml1m_multiclass.yaml
```

---

## 仓库结构

```bash
gerbil_train/
├── cli/            # 训练入口
│   └── gwen_train.py
├── config.py       # dataclass 配置
├── data/           # TFRecord 数据集和 collator
│   └── gwen_tfrecord_dataset.py
├── losses/         # 损失函数
│   ├── classification.py  # CE, NCE, SampledSoftmax
│   └── ranking.py
├── metrics/        # 评估指标
├── models/         # 模型结构
│   └── gwen.py
├── trainer/        # 训练循环
│   ├── base_trainer.py
│   └── gwen_trainer.py
└── utils/          # 辅助工具
```

---

## 配置目录结构

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

## 依赖

- Python 3.9+
- PyTorch 2.2+
- `tfrecord` — Python TFRecord 读取器
- 其他：见 `requirements.txt`

---

## 项目状态与质量

gerbil-train 目前处于**早期原型阶段**（约 1 个月开发，11+ 次提交，单人维护）。独立代码审查评分为 **3.2 / 5.0**，按维度分解如下：

| 维度 | 评分 | 说明 |
|------|:----:|------|
| 架构设计 | 4/5 | 清晰分层、模板方法模式、`@dataclass` 配置 |
| 类型注解 | 4/5 | 使用 Python 3.10+ 现代类型注解 |
| 代码复杂度 | 4/5 | 职责分离清晰，设计模式得当 |
| 文档 | 3/5 | 核心模块有文档，缺少 API 文档 |
| 错误处理 | 3/5 | 输入校验到位，缺少自定义异常和日志机制 |
| 测试 | 2/5 | 已补 41 个 GwEN 测试，无 CI/CD，无覆盖率追踪 |
| 工程实践 | 2/5 | CI/CD、格式化配置、Issue/PR 模板尚未配置 |

### 优势

- 模块化配置驱动架构，数据/模型/损失/训练四层解耦清晰
- 时间戳实验产物 + 完整配置快照，运行完全可复现
- `@dataclass` 配置，IDE 自动补全和类型安全
- 特征级 enabled/disabled 开关，零代码消融实验
- 可插拔损失函数（CE / NCE / Sampled Softmax），数学推导已记录

### 待改进

- **测试体系**：补充 metrics/losses/utils 单元测试，配置 CI/CD 自动化测试
- **文档**：API 参考、架构文档、贡献指南（CONTRIBUTING.md）
- **依赖管理**：区分开发/运行时依赖，锁定版本范围
- **社区基础设施**：Issue/PR 模板、GitHub Actions、语义化提交

### 关联项目（GERBIL 生态）

- [`gerbil-data`](https://github.com/shardzhang/gerbil-data) — 基于 Spark 的特征工程数据处理
- [`gerbil-serving`](https://github.com/shardzhang/gerbil-serving) — 在线推理与模型服务

---

## 关联项目

- `gerbil-data` — 数据处理与样本生成
- `gerbil-serving` — 在线推理与模型服务
