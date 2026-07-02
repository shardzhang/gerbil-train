# FTRL (Follow The Regularized Leader)

**FTRL 是一种在线学习算法**，与传统的 SGD/Adam 有本质区别。

## 核心思想

FTRL-Proximal 算法（McMahan et al., 2013）每步更新每个权重时都解一个带 L1 正则的优化问题：

$$ w_{t+1} = \arg\min_w \left( \sum_{s=1}^{t} g_s \cdot w + \frac{1}{2} \sum_{s=1}^{t} \sigma_s \|w - w_s\|^2 + \lambda_1 \|w\|_1 + \frac{\lambda_2}{2} \|w\|^2 \right) $$

解析解：

```
w = 0                                    if |z| ≤ λ1
w = -( (β + √n) / α + λ₂ )⁻¹ · (z - sign(z)·λ1)   otherwise
```

其中每个权重有独立的**学习率**（per-coordinate learning rate）。

## 与 Adam 的关键对比

| 维度 | Adam | FTRL |
|------|------|------|
| 学习率 | 全局，自适应衰减 | **per-coordinate**，每个权重独立 |
| L1 正则 | weight_decay 近似 | **精确 L1**，产生稀疏解 |
| 适用场景 | 深度神经网络 | **高维稀疏线性模型**（广告 CTR） |
| 参数初始化 | Xavier / He | **全零** |
| 训练方式 | epoch + mini-batch | 也可 online（逐样本更新） |

## 模型架构

FTRL 在当前框架中实现为一个**线性模型**：

```
EmbeddingBag(vocab, 1) for each field → sum → sigmoid
```

每个 field 的 EmbeddingBag 输出一个标量，所有 field 累加后过 sigmoid。没有 MLP，没有激活函数，是最简单的 LR。

## 参数说明

```yaml
optimizer:
  type: ftrl
  lr: 0.1        # α (alpha) — per-coordinate learning rate base
  beta: 1.0      # β — smoothing term
  lambda1: 1.0   # λ₁ — L1 regularization (越大越稀疏)
  lambda2: 1.0   # λ₂ — L2 regularization
```

**调参建议**：

| 参数 | 增大效果 | 减小效果 |
|------|---------|---------|
| `alpha` | 学习更慢，更稳定 | 学习更快，可能震荡 |
| `lambda1` | 更稀疏（更多权重归零） | 更稠密 |
| `beta` | 影响衰减曲线形状 | — |

## 与 DeepFM / W&D 的区别

- **DeepFM** = Linear + FM + Deep（三个 term）
- **Wide & Deep** = Linear + Deep（两个 term）
- **FTRL** = Linear only（一个 term，但用 FTRL 优化器）

FTRL 和 DeepFM/W&D 的 linear 部分在**模型结构上完全一致**，区别仅在优化器。

## 使用场景

FTRL 适合：
1. 高维稀疏特征（vocab 很大，每个样本只有少数非零特征）
2. 需要模型稀疏化（部署时剪枝掉零权重）
3. 在线学习（逐样本更新，无需 replay buffer）

不适合：
1. 深度模型（需要 MLP 层）
2. 需要特征交互（FM 或 Deep 部分）
3. 小数据集（FTRL 的 per-coordinate LR 优势在大量稀疏特征下才明显）

## 启动命令

```bash
python -m gerbil_train.cli.ftrl_train \
  --config configs/1-ftrl/experiment.yaml
```

## 参考

- McMahan, H. B., et al. "Ad click prediction: a view from the trenches." KDD 2013.
- [FTRL 算法详解（知乎）](https://zhuanlan.zhihu.com/p/584923052)
