# FM (Factorization Machine)

核心思想：LR 只能线性加权特征，无法建模二阶交互。FM 用因子分解的方式，为每个 field 学习一个 embedding，用两两 dot product 表示特征交互。

## 模型架构

```
FM = Linear + FM 二阶交互

Linear:     w_0 + Σ w_i · x_i          （一阶，每个 field 输出标量）
FM:         0.5 · ((Σ v)² - Σ(v²))     （二阶，所有 field embedding 两两 dot product）
Output:     sigmoid(Linear + FM)
```

## 核心公式

$$ \text{FM} = w_0 + \sum_{i=1}^{n} w_i x_i + \frac{1}{2} \sum_{k=1}^{K} \left( \left(\sum_{i=1}^{n} v_{i,k}\right)^2 - \sum_{i=1}^{n} v_{i,k}^2 \right) $$

第一个求和项是一阶线性（参数 = 词表大小），第二个求和项是二阶 FM 交互（参数 = 词表大小 × emb_size）。

## 与 DeepFM 的区别

| 对比 | FM | DeepFM |
|------|----|--------|
| Linear | ✅ | ✅ |
| FM | ✅ | ✅ |
| Deep (MLP) | ❌ | ✅ |
| 参数量 | 少 | 多（MLP 部分） |
| 适合场景 | 线性交互为主的数据 | 高维非线性交互 |

## 配置文件

```yaml
# configs/model/fm.yaml
task: binary
embedding:
  default_emb_size: 4
  fields: {}

mlp:   # FM 不使用 MLP，此配置项仅占位
  hidden_dims: [128, 64]
```

## 启动命令

```bash
python -m gerbil_train.cli.fm_train --config configs/1-fm/experiment.yaml
```

## 前提条件

所有 field 的 `emb_size` 必须相同（FM 公式要求所有 embedding 同维度），否则 `_validate_fields` 会抛异常。
