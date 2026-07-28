# gerbil-train

**GERBIL 推荐系统的离线训练与评估组件。**

**gerbil-train** 是 **GERBIL**（General Efficient Recommender for Benchmarking, Inference, and Learning）系统的离线训练与评估组件，提供配置驱动的、可复现的推荐模型训练与评估。基于 Python 和 PyTorch。

- **[gerbil-data](https://github.com/shardzhang/gerbil-data)** — 基于 Spark 的特征工程数据处理
- **[gerbil-serving](https://github.com/shardzhang/gerbil-serving)** — 在线推理与模型服务

## 支持的模型

| 模型 | 类型 | 说明 | 论文 |
|------|------|------|------|
| **FM** | CTR | Linear(一阶) + FM(二阶交叉)，无 Deep MLP | [Rendle 2010](https://doi.org/10.1109/ICDM.2010.127) |
| **FTRL** | CTR | FTRL-Proximal 在线学习优化器 | [McMahan 2013](https://doi.org/10.1145/2487575.2488200) |
| **FFM** | CTR | 域感知FM: 每对field独立embedding, Σ_{i<j} ⟨v_{i,f_j}, v_{j,f_i}⟩ | [Juan 2016](https://doi.org/10.1145/2959100.2959137) |
| **GwEN** | 多分类 | EmbeddingBag + 可选 field 级注意力 + MLP | — |
| **GwEN 二分类** | CTR | 二分类变体，sigmoid 输出 | — |
| **YouTubeDNN** | 多分类 | Behavior `mode="mean"`，example age，`encode()` 推理 | [Covington 2016](https://doi.org/10.1145/2959100.2959190) |
| **EGES** | 召回 | 增强图嵌入+边信息, 注意力加权物品+边信息嵌入 | [Wang 2018](https://doi.org/10.1145/3178876.3186070) |
| **Word2Vec** | 召回 | 行为序列Skip-gram负采样, target+context物品嵌入 | [Mikolov 2013](https://arxiv.org/abs/1310.4546) |
| **SSL** | 召回 | 自监督对比学习, 数据增强(mask/crop) + InfoNCE损失 | [Xie 2021](https://arxiv.org/abs/2104.06879) |
| **NCF** | CTR | GMF(逐元素乘积) + MLP + NeuMF融合, 学习非线性用户-物品交互 | [He 2017](https://doi.org/10.1145/3038912.3052569) |
| **AFM** | CTR | FM + 每对特征交叉可学习的注意力权重 | [Xiao 2017](https://doi.org/10.24963/ijcai.2017/435) |
| **NFM** | CTR | Bi-Interaction 池化 → MLP | [He 2017](https://doi.org/10.1145/3038912.3052569) |
| **PNN** | CTR | Linear + Product Layer（内积交叉）+ MLP | [Qu 2016](https://doi.org/10.1109/icdm.2016.0151) |
| **Wide & Deep** | CTR | Linear(Wide) + MLP(Deep)，per-field 控制 | [Cheng 2016](https://doi.org/10.1145/2988450.2988454) |
| **DeepFM** | CTR | Linear + FM + Deep 共享特征嵌入 | [Guo 2017](https://doi.org/10.24963/ijcai.2017/239) |
| **xDeepFM** | CTR | Linear + **CIN**（压缩交互网络）+ Deep | [Lian 2018](https://doi.org/10.1145/3219819.3220023) |
| **DCN** | CTR | Cross Network（有界交叉）+ Deep MLP | [Wang 2017](https://doi.org/10.1145/3124749.3124754) |
| **DCNv2** | CTR | 全 d×d 矩阵交叉层，支持低秩近似 | [Wang 2021](https://doi.org/10.1145/3459637.3481951) |
| **FiBiNet** | CTR | SENET 特征加权 + 双线性交互 + MLP | [Huang 2019](https://arxiv.org/abs/1905.09433) |
| **STAR** | CTR | 星形拓扑自适应网络, 每场景参数 = 共享 ⊙ 专用 | [Sheng 2022](https://dl.acm.org/doi/10.1145/3459637.3482412) |
| **MaskNet** | CTR | 实例引导掩码块, 样本级特征掩码 + FC | [Wang 2021](https://arxiv.org/abs/2102.07619) |
| **AutoInt** | CTR | Multi-head self-attention 建模特征交互 | [Song 2019](https://doi.org/10.1145/3357384.3357925) |
| **DLRM** | CTR | Bottom MLP(连续) + 稀疏嵌入 + 全配对点积 + Top MLP | [Naumov 2019](https://arxiv.org/abs/1906.00091) |
| **DIN** | 序列推荐 | LocalActivationUnit 行为序列注意力 | [Zhou 2018](https://doi.org/10.1145/3219819.3219823) |
| **DIEN** | 序列推荐 | GRU + AUGRU 行为演化建模 | [Zhou 2019](https://doi.org/10.1609/aaai.v33i01.33015941) |
| **DSIN** | 序列推荐 | 会话分割 + Bi-LSTM + 自注意力 | [Feng 2019](https://doi.org/10.24963/ijcai.2019/319) |
| **MIMN** | 序列推荐 | 多槽记忆网络 + Bi-LSTM | [Pi 2019](https://doi.org/10.1145/3292500.3330666) |
| **SIM** | 序列推荐 | GSU 检索 + ESU 多头交叉注意力 | [Pi 2020](https://doi.org/10.1145/3340531.3412744) |
| **MIND** | 序列推荐 | 动态路由(CapsNet)提取K个兴趣胶囊, 标签感知注意力选择 | [Li 2019](https://doi.org/10.1145/3357384.3357814) |
| **BST** | 序列推荐 | Transformer编码器(自注意力)建模成对物品交互 | [Chen 2019](https://doi.org/10.1145/3326937.3341261) |
| **ETA** | 序列推荐 | LSH哈希约束注意力, 哈希匹配降低长序列复杂度 | [Li 2021](https://dl.acm.org/doi/10.1145/3459637.3482270) |
| **SDIM** | 序列推荐 | Gumbel-Sigmoid概率语义掩码, 过滤无关行为 | [Fan 2022](https://doi.org/10.1145/3511808.3557082) |
| **SASRec** | 序列推荐 | 因果Transformer(从左到右), 仅从最后行为提取兴趣 | [Kang 2018](https://arxiv.org/abs/1808.09781) |
| **BERT4Rec** | 序列推荐 | [CLS]标记 + 双向Transformer, 无因果掩码 | [Sun 2019](https://doi.org/10.1145/3357384.3357895) |
| **TWIN** | 序列推荐 | 两阶段兴趣发现, 原型分配 + 目标感知注意力 | [TWIN 2023](https://arxiv.org/abs/2302.09894) |
| **ESMM** | 多任务 | 全空间CVR多任务模型, pCTCVR = pCTR × pCVR, 全空间损失 | [Ma 2018](https://doi.org/10.1145/3209978.3210104) |
| **双塔模型** | 检索 | 两阶段训练（隐式 + 显式） | [Yi 2019](https://doi.org/10.1145/3298689.3346996) |
| **MV-DNN** | 检索 | 多视角DNN, 独立用户/物品DNN + 余弦相似度 + BPR损失 | [Elkahky 2015](https://www.microsoft.com/en-us/research/publication/multi-view-deep-neural-network-for-cross-view-learning/) |
| **MMoE** | 多任务 | 多门控专家混合, K个共享专家 + T个任务门控, 软路由 | [Ma 2018](https://dl.acm.org/doi/10.1145/3219819.3220007) |
| **PLE** | 多任务 | 渐进分层提取, 多层共享+任务专用专家, 渐进分离 | [Tang 2020](https://dl.acm.org/doi/10.1145/3383313.3412236) |
| **PEPNet** | 多任务 | 参数高效个性化网络, EPNet按域/用户生成scale&bias | [Chang 2023](https://dl.acm.org/doi/10.1145/3543507.3583206) |
| **GateNet** | 多任务 | 逐field sigmoid门控特征选择 + MMoE 骨干 | [Dai 2021](https://dl.acm.org/doi/10.1145/3459637.3481952) |
| **AdaTT** | 多任务 | 自适应任务间融合, 每任务独立专家 + 跨任务门控融合层 | [Li 2022](https://dl.acm.org/doi/10.1145/3485447.3512040) |
| **LTR** | 排序 | 多种排序损失（LambdaRank, RankNet 等） | [Burges 2005](https://doi.org/10.1145/1102351.1102363) |
| **BPR** | 排序 | 负采样配对排序损失, 用户-物品点积 | [Rendle 2009](https://arxiv.org/abs/1205.2618) |

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
python -m gerbil_train.cli.3-afm_train           --config configs/3-afm/experiment.yaml
python -m gerbil_train.cli.3-nfm_train           --config configs/3-nfm/experiment.yaml
python -m gerbil_train.cli.3-pnn_train           --config configs/3-pnn/experiment.yaml
python -m gerbil_train.cli.6-autoint_train       --config configs/6-autoint/experiment.yaml
python -m gerbil_train.cli.6-fibinet_train       --config configs/6-fibinet/experiment.yaml
python -m gerbil_train.cli.6-dcn_train           --config configs/6-dcn/experiment.yaml
python -m gerbil_train.cli.6-dcnv2_train         --config configs/6-dcnv2/experiment.yaml
python -m gerbil_train.cli.5-xdeepfm_train       --config configs/5-xdeepfm/experiment.yaml
python -m gerbil_train.cli.4-wide_and_deep_train --config configs/4-wide_and_deep/experiment.yaml
python -m gerbil_train.cli.1-lr_train          --config configs/1-lr/experiment.yaml

# 序列模型
python -m gerbil_train.cli.7-din_train           --config configs/7-din/experiment.yaml
python -m gerbil_train.cli.7-dien_train          --config configs/7-dien/experiment.yaml
python -m gerbil_train.cli.7-dsin_train          --config configs/7-dsin/experiment.yaml

# 多分类模型
python -m gerbil_train.cli.2-youtube_dnn_train   --config configs/2-youtube_dnn/experiment.yaml
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
├── models/        # 16 个模型（FM/FTRL/GwEN/GwEN Binary/YouTubeDNN/AFM/NFM/PNN/W&D/DeepFM/xDeepFM/AutoInt/DIEN/DIN/双塔/LTR）
├── optimizers/    # FTRL-Proximal 优化器
├── trainer/       # 12 个训练器（共享 binary/multi 基类）
└── utils/         # 辅助工具
```

## 文档

各模型的详细文档在 `docs/` 目录：

| 文档 | 说明 |
|------|------|
| `docs/1-fm.md` | FM 因子分解机 |
| `docs/1-ftrl.md` | FTRL 在线学习算法 |
| `docs/2-gwen.md` | GwEN 架构、公式、配置 |
| `docs/2-youtube_dnn.md` | YouTubeDNN encode() ANN 推理 |
| `docs/3-afm.md` | AFM 注意力因子分解机 |
| `docs/3-nfm.md` | NFM Bi-Interaction + Deep MLP |
| `docs/3-pnn.md` | PNN Product Layer + MLP |
| `docs/4-wide_and_deep.md` | W&D per-field tower 控制 |
| `docs/5-deepfm.md` | DeepFM Linear + FM + Deep |
| `docs/5-xdeepfm.md` | xDeepFM Linear + CIN + Deep |
| `docs/6-dcn.md` | DCN Cross Network + Deep MLP |
| `docs/6-dcnv2.md` | DCNv2 全矩阵交叉 + 低秩近似 |
| `docs/6-autoint.md` | AutoInt Transformer 自注意力 |
| `docs/7-dien.md` | DIEN GRU + AUGRU |
| `docs/7-dsin.md` | DSIN 会话分割 + Bi-LSTM + 自注意力 |
| `docs/7-din.md` | DIN 注意力机制和兴趣池化 |
| `docs/99-shared_bottom_two_tower.md` | 双塔检索模型 |

## 项目状态

**活跃开发中**，**100 个**单元测试全部通过。CI/CD 已配置（GitHub Actions），自定义异常层次（8 类）。

## 关联项目（GERBIL 生态）

## 关联项目（GERBIL 生态）

- [`gerbil-data`](https://github.com/shardzhang/gerbil-data) — 基于 Spark 的特征工程
- [`gerbil-serving`](https://github.com/shardzhang/gerbil-serving) — 在线推理与模型服务
