# FTRL (Follow The Regularized Leader)

**FTRL 是一种在线学习算法**，与传统的 SGD/Adam 有本质区别。



## 核心思想

FTRL-Proximal 算法（McMahan et al., 2013）每步更新每个权重时都解一个带 L1 正则的优化问题：

$$ w_{t+1} = \arg\min_w \left( \sum_{s=1}^{t} g_s \cdot w + \frac{1}{2} \sum_{s=1}^{t} \sigma_s \|w - w_s\|^2 + \lambda_1 \|w\|_1 + \frac{\lambda_2}{2} \|w\|^2 \right) $$

解析解：

$$
w_{t+1,i} =
\begin{cases}
0 & \text{if } |z_{t,i}| \leq \lambda_1 \\
-\eta_t (z_{t,i} - \operatorname{sgn}(z_{t,i}) \lambda_1) & \text{otherwise.}
\end{cases}
$$

其中每个权重有独立的**学习率**（per-coordinate learning rate）。



## 与 Adam 的关键对比

| 维度 | Adam | FTRL |
|------|------|------|
| 学习率 | per-parameter，指数移动平均自适应 | **per-coordinate**，每个权重独立 |
| L1 正则 | weight_decay 近似 | **精确 L1**，产生稀疏解 |
| 适用场景 | 深度神经网络 | **高维稀疏线性模型**（广告 CTR） |
| 参数初始化 | Xavier / He | **全零** |
| 训练方式 | epoch + mini-batch | 也可 online（逐样本更新） |


> Adam 更新：$\theta_{t+1,i} = \theta_{t,i} - \eta \cdot \frac{m_{t,i}}{\sqrt{v_{t,i}} + \epsilon}$
>
> 每个参数 $\theta_i$ 的有效学习率是 $\eta / (\sqrt{v_{t,i}} + \epsilon)$，其中 $v_{t,i}$ 是梯度平方的指数移动平均——**每个参数独立调整**。
>
> 事实上 Adam 和 FTRL 在自适应学习率上**形式非常相似**：
>
> | Adam     | $\eta / (\sqrt{v_{t,i}} + \epsilon)$ | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$（近期加权） |
> | -------- | ------------------------------------ | ------------------------------------------------------- |
> | **FTRL** | $\alpha / (\beta + \sqrt{n_{t,i}})$  | $n_i = \sum g_{t,i}^2$（全历史累积）                    |
>
> 区别在于**衰减策略**（指数移动平均 vs 全累积），而非是否"全局"。



Adam的per-parameter和FTRL的per-coordinate，是一回事么？

**本质上是同一个概念，但实现机制不同。**

相同点：
- 都继承了 AdaGrad 的思想：每个维度/参数独立维护梯度统计量，频繁更新的维度学习率小，稀疏维度学习率大
- 公式形式一致：$\text{base} / (\beta + \sqrt{\text{统计量}})$

不同点：

|                | Adam per-parameter                                      | FTRL per-coordinate                 |
| -------------- | ------------------------------------------------------- | ----------------------------------- |
| **梯度统计**   | 指数移动平均 $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ | 全累积 $\sum g_{s,i}^2$（永不衰减） |
| **动量**       | 有 $m_t$（一阶矩）                                      | 无                                  |
| **学习率衰减** | EMA 不会衰减到 0                                        | 全累积 → 训练够久会趋近 0           |
| **设计目标**   | 非凸 DNN 快速收敛                                       | 凸优化 + **L1 稀疏截断**            |

**类比**：都是"根据频率调音量"——Adam 像最近一段时间平均音量（EMA），新声音来了旧的就逐渐忘掉；FTRL 像从出生到现在的总累积音量，永远不会忘记，所以越往后反应越迟钝。两者各有各的用途场景。



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
python -m gerbil_train.cli.1-ftrl_train --config configs/1-ftrl/experiment.yaml
```



## 参考

- McMahan, H. B., et al. "Ad click prediction: a view from the trenches." KDD 2013.





小数据集上 FTRL 的优势不明显——per-coordinate LR 在大量稀疏特征下收益最大；小数据可直接用 batch L1 LR（如 sklearn），且 online/streaming 的优势也无从发挥。 为什么小数据集上 online/streaming 的优势也无从发挥



因为 online/streaming 的两个核心价值在小数据下都不存在：

1. **数据太大装不进内存** → 只能流式读一遍，处理完就丢。小数据完全装得进内存，不需要流式，可以反复读。
2. **数据是连续到达的实时流** → 模型必须一边服务一边学习，无法存储所有历史。小数据集是固定的静态数据，不是流，直接 batch 训练多轮即可。

小数据上 batch L-BFGS 一类的二阶优化器收敛更快（用更少步数），且可以做交叉验证、网格搜索等线上不适用的事。FTRL 的"只过一遍"在数据装得下时是劣势而非优势。
