# Shared-Bottom Two-Tower（共享底部双塔模型）

用于检索场景的两阶段训练模型：隐式反馈预训练 → 显式反馈精调。

## 模型架构

```
                ┌────────┐     ┌───────────┐
                │ Implicit│    │  Explicit  │
                │  Tower  │    │   Tower    │
                └───┬────┘     └─────┬─────┘
                    │                │
                    └───────┬────────┘
                            │
                      ┌─────┴─────┐
                      │   Shared   │
                      │  Bottom    │
                      │  Encoder   │
                      └─────┬─────┘
                            │
                 ┌──────────┴──────────┐
                 │                     │
           ┌─────┴─────┐        ┌─────┴─────┐
           │  Query    │        │   Item    │
           │ Features  │        │ Features  │
           └───────────┘        └───────────┘
```

## 两阶段训练

### Stage 1: Implicit Pre-Train

用隐式反馈（点击、播放）训练 query encoder + item encoder，产出 query 和 item 的 embedding。

### Stage 2: Explicit Fine-Tune

用显式反馈（评分、点赞）精调，共享 bottom encoder，只更新 explicit tower 的参数。

## 推理

训练完成后，query embedding 和 item embedding 均可导出给 ANN 搜索：

```python
# 从 checkpoint 中提取
query_emb = model.query_encoder(query_features)
item_emb = model.item_encoder(item_features)
```

## 启动命令

```bash
python -m gerbil_train.cli.shared_bottom_two_tower_train \
  --config configs/99-sbtt/experiment.yaml
```

## 参考

- Ma, J., et al. "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts." KDD 2018.
