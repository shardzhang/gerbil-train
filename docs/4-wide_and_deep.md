# Wide & Deep

核心思想：同时利用 Wide（线性层）的**记忆能力**和 Deep（MLP）的**泛化能力**。

## 模型架构

```
Wide:   w_0 + Σ w_i · x_i           （仅 wide=True 的字段参与）
Deep:   MLP(concat(v_1, ..., v_n))  （仅 deep=True 的字段参与）
Output: sigmoid(Wide + Deep)
```

## Per-Field Tower 控制

每个 field 可通过 YAML 配置进入 Wide、Deep 或 Both：

| wide | deep | 参与 Wide | 参与 Deep | 典型用例 |
|:----:|:----:|:---------:|:---------:|---------|
| true | true | ✅ | ✅ | ID 类特征，需要同时记忆+泛化 |
| true | false | ✅ | ❌ | 需要精确记忆的字段（user_id） |
| false | true | ❌ | ✅ | 统计值、上下文，需 MLP 隐式交互 |

## 推荐配置

```yaml
# Wide only — 线性 Memorization
user_id:         {wide: true,  deep: false}
item_id:         {wide: true,  deep: false}

# Deep only — 非线性 Generalization
user_rate_std:          {wide: false, deep: true}
context_time_hour:      {wide: false, deep: true}
user_watch_same_genre:  {wide: false, deep: true}
```

## 与 DeepFM 的区别

| 对比 | Wide & Deep | DeepFM |
|------|------------|--------|
| Linear (Wide) | ✅ | ✅ |
| Deep (MLP) | ✅ | ✅ |
| FM 二阶交互 | ❌ | ✅ |
| 训练速度 | 略快（无 FM） | 略慢（有 FM） |

## 启动命令

```bash
python -m gerbil_train.cli.wide_and_deep_train --config configs/4-wide_and_deep/experiment.yaml
```

## 参考

- Cheng, H. T., et al. "Wide & Deep Learning for Recommender Systems." DLRS 2016.
