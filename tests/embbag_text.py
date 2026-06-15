import torch

# 模拟 3 条样本，movie_genres 的 vocab_size = 19
# token ID 必须在 [0, 18] 范围内
samples = [
    {"movie_genres_index": [5, 12, 3], "movie_genres_value": [1.0, 1.0, 1.0]},  # 3 个 genre
    {"movie_genres_index": [9],        "movie_genres_value": [1.0]},              # 1 个 genre
    {"movie_genres_index": [18, 7],    "movie_genres_value": [0.8, 1.2]},        # 2 个 genre，权重不同
]

torch.manual_seed(42)
embedding = torch.nn.EmbeddingBag(num_embeddings=19, embedding_dim=4, mode="sum")
torch.nn.init.xavier_uniform_(embedding.weight)

print("=" * 70)
print("step 0: movie_genres 的 embedding 表 (vocab_size=19, embedding_dim=4)")
print("=" * 70)
for i in range(19):
    print(f"  token {i:>2d}: {[round(v, 4) for v in embedding.weight[i].detach().tolist()]}")
print()

"""
======================================================================
step 0: movie_genres 的 embedding 表 (vocab_size=19, embedding_dim=4)
======================================================================
  token  0: [0.3044, 0.3472, -0.3704, -0.2727]
  token  1: [0.4677, -0.1723, -0.1811, -0.4942]
  token  2: [-0.2925, 0.1276, -0.0674, -0.3707]
  token  3: [0.012, -0.3489, -0.4333, -0.2813]
  token  4: [-0.447, -0.3252, 0.5106, 0.0965]
  token  5: [0.1574, -0.4764, -0.3354, -0.17]
  token  6: [0.0799, -0.4494, -0.2201, -0.3058]
  token  7: [0.0014, -0.1901, -0.0354, -0.3461]
  token  8: [-0.3506, -0.298, -0.1748, -0.4031]
  token  9: [0.4283, -0.1014, 0.4395, 0.1591]
  token 10: [-0.4325, 0.3535, -0.1405, -0.1958]
  token 11: [-0.424, -0.5078, 0.1461, -0.1116]
  token 12: [0.1988, -0.4192, 0.3792, -0.3749]
  token 13: [-0.0882, 0.1067, 0.2637, 0.4123]
  token 14: [0.4653, -0.405, 0.1285, -0.2197]
  token 15: [-0.056, -0.3823, 0.4652, -0.3749]
  token 16: [0.273, 0.1795, 0.166, -0.2761]
  token 17: [0.4643, 0.1122, 0.0657, -0.4501]
  token 18: [0.2144, -0.0766, -0.234, 0.4387]
"""

print("=" * 70)
print("step 1: collator 输出 (indices, offsets, weights)")
print("=" * 70)
indices_list, offsets_list, weights_list, cursor = [], [], [], 0
for i, s in enumerate(samples):
    offsets_list.append(cursor)
    idx, val = s["movie_genres_index"], s["movie_genres_value"]
    size = min(len(idx), len(val))
    print(f"  样本 {i}: indices={idx}, weights={val}, {size} 个 token")
    if size > 0:
        indices_list.extend(idx[:size])
        weights_list.extend(val[:size])
        cursor += size

"""
======================================================================
step 1: collator 输出 (indices, offsets, weights)
======================================================================
  样本 0: indices=[5, 12, 3], weights=[1.0, 1.0, 1.0], 3 个 token
  样本 1: indices=[9], weights=[1.0], 1 个 token
  样本 2: indices=[18, 7], weights=[0.8, 1.2], 2 个 token

  indices  = [5, 12, 3, 9, 18, 7]
  offsets  = [0, 3, 4]
  weights  = [1.0, 1.0, 1.0, 1.0, 0.800000011920929, 1.2000000476837158]
    样本 0 → indices[0:3]=[5, 12, 3], weights=[1.0, 1.0, 1.0]
    样本 1 → indices[3:4]=[9], weights=[1.0]
    样本 2 → indices[4:6]=[18, 7], weights=[0.800000011920929, 1.2000000476837158]
"""

indices = torch.tensor(indices_list, dtype=torch.long)
offsets = torch.tensor(offsets_list, dtype=torch.long)
# shape:  [total_tokens]。
weights = torch.tensor(weights_list, dtype=torch.float32)
print(f"\n  indices  = {indices.tolist()}")
print(f"  offsets  = {offsets.tolist()}")
print(f"  weights  = {weights.tolist()}")
for i in range(len(samples)):
    start = offsets[i].item()
    end = offsets[i+1].item() if i+1 < len(offsets) else cursor
    print(f"    样本 {i} → indices[{start}:{end}]={indices[start:end].tolist()}, "
          f"weights={weights[start:end].tolist()}")
print()

print("=" * 70)
print("step 2: EmbeddingBag 前向")
print("=" * 70)

# shape: [total_tokens_across_batch, embedding_dim]
raw_embeds = embedding.weight[indices]

print("  查表结果:")
for i in range(indices.size(0)):
    e = [round(v, 4) for v in raw_embeds[i].detach().tolist()]
    print(f"    token {indices[i].item()} × weight {weights[i].item()} → {e}")

"""
======================================================================
step 2: EmbeddingBag 前向
======================================================================
  查表结果:
    token 5 × weight 1.0 → [0.1574, -0.4764, -0.3354, -0.17]
    token 12 × weight 1.0 → [0.1988, -0.4192, 0.3792, -0.3749]
    token 3 × weight 1.0 → [0.012, -0.3489, -0.4333, -0.2813]
    token 9 × weight 1.0 → [0.4283, -0.1014, 0.4395, 0.1591]
    token 18 × weight 0.800000011920929 → [0.2144, -0.0766, -0.234, 0.4387]
    token 7 × weight 1.2000000476837158 → [0.0014, -0.1901, -0.0354, -0.3461]
"""

output = embedding(indices, offsets, per_sample_weights=weights)
print()
print("  手动加权求和验证:")
for i in range(len(samples)):
    start = offsets[i].item()
    end = offsets[i+1].item() if i+1 < len(offsets) else cursor
    seg = raw_embeds[start:end]
    seg_w = weights[start:end].unsqueeze(-1)
    weighted = seg * seg_w
    summed = weighted.sum(dim=0)
    print(f"    样本 {i}:")
    print(f"      token embeds × weights:")
    for j in range(end - start):
        e = [round(v,4) for v in seg[j].detach().tolist()]
        w = round(seg_w[j].item(), 4)
        print(f"        {e} × {w}")
    print(f"      = 加权求和: {[round(v,4) for v in summed.detach().tolist()]}")
    print(f"      EmbeddingBag 输出: {[round(v,4) for v in output[i].detach().tolist()]}")
    print()

"""
  手动加权求和验证:
    样本 0:
      token embeds × weights:
        [0.1574, -0.4764, -0.3354, -0.17] × 1.0
        [0.1988, -0.4192, 0.3792, -0.3749] × 1.0
        [0.012, -0.3489, -0.4333, -0.2813] × 1.0
      = 加权求和: [0.3682, -1.2444, -0.3896, -0.8262]
      EmbeddingBag 输出: [0.3682, -1.2444, -0.3896, -0.8262]

    样本 1:
      token embeds × weights:
        [0.4283, -0.1014, 0.4395, 0.1591] × 1.0
      = 加权求和: [0.4283, -0.1014, 0.4395, 0.1591]
      EmbeddingBag 输出: [0.4283, -0.1014, 0.4395, 0.1591]

    样本 2:
      token embeds × weights:
        [0.2144, -0.0766, -0.234, 0.4387] × 0.8
        [0.0014, -0.1901, -0.0354, -0.3461] × 1.2
      = 加权求和: [0.1732, -0.2894, -0.2297, -0.0644]
      EmbeddingBag 输出: [0.1732, -0.2894, -0.2297, -0.0644]
"""

print("=" * 70)
print("最终: 每个样本得到一个定长 4 维向量")
print("=" * 70)
for i in range(len(samples)):
    print(f"  样本 {i} → {[round(v,4) for v in output[i].detach().tolist()]}")

"""
======================================================================
最终: 每个样本得到一个定长 4 维向量
======================================================================
  样本 0 → [0.3682, -1.2444, -0.3896, -0.8262]
  样本 1 → [0.4283, -0.1014, 0.4395, 0.1591]
  样本 2 → [0.1732, -0.2894, -0.2297, -0.0644]
"""


"""
上面就是完整过程，核心是：
变长的 token 序列先压平成一根 indices（绳子）+ weights（系数），再用 offsets 标记每个样本的起止位置，EmbeddingBag 收到后：查表 → 乘权重 → 按 offsets 求和 → 输出定长向量。
最后 3 个样本的 movie_genres 特征被转换成了 3 个等长的 4 维向量（样本 0 长度 3 → 4 维，样本 1 长度 1 → 4 维，样本 2 长度 2 → 4 维），后续和其他 field 的向量 concat 后进 MLP。
"""