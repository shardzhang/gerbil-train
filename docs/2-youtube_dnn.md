# YouTubeDNN

YouTube 推荐的深度神经网络（Covington et al., 2016）。

## 模型架构

```
feature_bags → EmbeddingBag(mode=mean for behavior, sum for others) → concat
    → MLP → user_embedding → Linear(vocab, bias=False) → logits → softmax
```

## 核心设计

### 1. Behavior `mode="mean"`

行为序列字段（如 `watch_history`）用 `mode="mean"`，使输出与序列长度无关。

```yaml
behavior_fields:
  - watch_history     # 这个 field 使用 mode="mean"
```

### 2. Example Age

```python
embedding = log(age + 1)    # 训练时取 log
embedding = log(0 + 1) = 0  # 推理时固定为 0（最近的行为）
```

YAML 配置：

```yaml
example_age_field: age
```

### 3. Bias-Free Head

`Linear(hidden_size, vocab, bias=False)` — `head.weight` 就是 item embedding 矩阵，可直接导出给 ANN 搜索。

### 4. Encode 分离推理

```python
user_emb = model.encode(feature_bags)   # → [batch_size, hidden_dim]
# 在 gerbil-serving 中用 faiss 搜索 item_emb = head.weight.T
```

## 与 GwEN 的区别

| 对比 | GwEN multiclass | YouTubeDNN |
|------|----------------|------------|
| Behavior mode | mode="sum" | mode="mean" |
| Head bias | bias=True | bias=False |
| Example age | 无 | log(age+1) |
| 推理 | 完整 forward | encode() → ANN 搜索 |

## 启动命令

```bash
python -m gerbil_train.cli.youtube_dnn_train --config configs/2-youtube_dnn/experiment.yaml
```

## 参考

- Covington, P., et al. "Deep Neural Networks for YouTube Recommendations." RecSys 2016.
