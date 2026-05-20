"""ListMLE

输入: 136 维特征
输出: 每个文档的相关性得分
评价: NDCG@k
"""

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

warnings.filterwarnings('ignore')

def normalize_query_features(features, eps=1e-6):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < eps] = 1.0
    return (features - mean) / std

def load_mslrweb10k_groups(file_path):
    dataset = torch.load(file_path, map_location="cpu", weights_only=False)

    def process_split(split):
        groups = []
        for qid, (features, labels) in split.items():
            features = normalize_query_features(np.asarray(features, dtype=np.float32))
            labels = np.asarray(labels, dtype=np.float32)
            groups.append({
                'qid': qid,
                'X': torch.from_numpy(features),
                'y': torch.from_numpy(labels),
            })
        return groups

    train_groups = process_split(dataset["train"])
    val_groups = process_split(dataset["vali"])
    test_groups = process_split(dataset["test"])
    return train_groups, val_groups, test_groups

def load_letor_data(file_path):
    """读取 LETOR4.0"""
    labels = []
    qids = []
    features = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = int(parts[0])
            qid = int(parts[1].split(':')[1])

            feat = np.zeros(46)
            for item in parts[2:]:
                if ':' not in item:
                    continue
                fid, val = item.split(':')
                fid = int(fid) - 1
                feat[fid] = float(val)
            labels.append(label)
            qids.append(qid)
            features.append(feat)
    return np.array(features), np.array(labels), np.array(qids)


def load_mslr_data(file_path):
    """读取 MSLR-WEB10K """
    labels = []
    qids = []
    features = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = int(parts[0])
            qid = int(parts[1].split(':')[1])
            feat = np.zeros(136)
            for item in parts[2:]:
                fid, val = item.split(':')
                feat[int(fid)-1] = float(val)
            labels.append(label)
            qids.append(qid)
            features.append(feat)
    return np.array(features), np.array(labels), np.array(qids)

def normalize_by_query(X, qids, eps=1e-6):
    """query-wise 归一化
    X = (X - mean) / std
    """
    unique_qids = np.unique(qids)
    X_norm = X.copy()
    
    for qid in unique_qids:
        mask = (qids == qid)
        x_q = X[mask]
        
        mean = x_q.mean(axis=0, keepdims=True)
        std = x_q.std(axis=0, keepdims=True)
        std[std < eps] = 1.0
        
        X_norm[mask] = (x_q - mean) / std
    return X_norm


dataset_path = "/tmp/MSLR-WEB10K.pt"
print(f"Loading dataset from {dataset_path}")
train_groups, val_groups, test_groups = load_mslrweb10k_groups(dataset_path)
print(f"Loaded {len(train_groups)} train / {len(val_groups)} val / {len(test_groups)} test queries")

def group_by_qid(X, y, qids):
    """按 query ID 分组
     - X: 特征矩阵 (N, D)
     - y: 标签向量 (N,)
     - qids: query ID 向量 (N,)
    """
    groups = defaultdict(lambda: {'X': [], 'y': []})
    for i, qid in enumerate(qids):
        groups[qid]['X'].append(X[i])
        groups[qid]['y'].append(y[i])
    
    grouped = []
    for g in groups.values():
        grouped.append({
            'X': torch.stack(g['X']),
            'y': torch.stack(g['y'])
        })
    return grouped

class DeepRankNet(nn.Module):
    def __init__(self, input_dim=136, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)


def listmle_loss(scores, labels):
    sorted_idx = torch.argsort(labels, descending=True)
    scores_sorted = scores[sorted_idx]
    cumsum = torch.logcumsumexp(torch.flip(scores_sorted, dims=[0]), dim=0)
    cumsum = torch.flip(cumsum, dims=[0])
    loss = -(scores_sorted - cumsum).mean()
    return loss

def ranknet_loss(s1, s2, y1, y2):
    S = 1.0 if y1 > y2 else (-1.0 if y1 < y2 else 0.0)
    C = torch.log(1 + torch.exp(s2 - s1)) - S * (s2 - s1)
    return C.mean()

def lambdarank_loss(scores, labels):
    sorted_idx = torch.argsort(labels, descending=True)
    s = scores[sorted_idx]
    y = labels[sorted_idx]

    valid_pairs = torch.triu(y[:, None] > y[None, :], diagonal=1)
    if not valid_pairs.any():
        return scores.new_tensor(0.0)

    pairwise_score_diff = s[None, :] - s[:, None]
    pairwise_loss = torch.logaddexp(
        pairwise_score_diff[valid_pairs],
        torch.zeros_like(pairwise_score_diff[valid_pairs]),
    )
    return pairwise_loss.mean()


def ndcg_score(y_true, y_score, k=5):
    k = min(k, len(y_true))
    order = y_score.argsort(descending=True)[:k]
    ranked_y_true = y_true[order]
    gain = 2 ** ranked_y_true - 1
    discount = torch.log2(torch.arange(2, 2+k, dtype=torch.float32, device=y_true.device))
    dcg = (gain / discount).sum()
    ideal = torch.sort(y_true, descending=True)[0][:k]
    ideal_gain = 2 ** ideal - 1
    idcg = (ideal_gain / discount).sum()
    return dcg / idcg if idcg.item() > 0 else y_true.new_tensor(0.0)


MODEL_TYPE = "LambdaRank"  # 可选：RankNet / LambdaRank / ListMLE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepRankNet(input_dim=136).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
writer = SummaryWriter()

# 早停
patience = 5
wait = 0
best_ndcg = 0.0

# 绘图日志
train_loss_history = []
val_ndcg_history = []

epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for group in train_groups:
        X = group['X'].to(device)
        y = group['y'].to(device)
        
        optimizer.zero_grad()
        scores = model(X)
        
        if MODEL_TYPE == "RankNet":
            loss = 0
            cnt = 0
            for i in range(len(scores)):
                for j in range(i+1, len(scores)):
                    loss += ranknet_loss(scores[i], scores[j], y[i], y[j])
                    cnt += 1
            loss = loss / cnt if cnt > 0 else 0
        elif MODEL_TYPE == "LambdaRank":
            loss = lambdarank_loss(scores, y)

        elif MODEL_TYPE == "ListMLE":
            loss = listmle_loss(scores, y)
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

        loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        if loss_value > 0:
            loss.backward()
            optimizer.step()
            total_loss += loss_value
        
    # 验证
    model.eval()
    ndcg_sum = 0.0
    with torch.no_grad():
        for group in val_groups:
            X = group['X'].to(device)
            y = group['y'].to(device)
            scores = model(X)
            ndcg_sum += ndcg_score(y, scores, k=5).item()
    ndcg_val = ndcg_sum / len(val_groups)

    train_loss_history.append(total_loss)
    val_ndcg_history.append(ndcg_val)
    writer.add_scalar("Loss/train", total_loss, epoch)
    writer.add_scalar("NDCG/val", ndcg_val, epoch)
    scheduler.step(ndcg_val)

    print(f"Epoch {epoch+1:2d} | loss: {total_loss:.4f} | NDCG@5 val: {ndcg_val:.4f}")

    if ndcg_val > best_ndcg:
        best_ndcg = ndcg_val
        wait = 0
        torch.save(model.state_dict(), "best_ltr_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print("✅ 早停触发！")
            break

writer.close()
# tensorboard --logdir=runs

# 绘图
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_loss_history)
plt.title("Train Loss")
plt.subplot(1,2,2)
plt.plot(val_ndcg_history)
plt.title("Val NDCG@5")
plt.tight_layout()
plot_path = Path("listmle_training_curves.png")
plt.savefig(plot_path)
plt.close()
print(f"Saved training curves to {plot_path.resolve()}")


# 测试集评估
model.load_state_dict(torch.load("best_ltr_model.pth", map_location=device))
model.eval()

ndcg1 = ndcg3 = ndcg5 = ndcg10 = 0.0
with torch.no_grad():
    for group in test_groups:
        X = group['X'].to(device)
        y = group['y'].to(device)
        scores = model(X)
        ndcg1 += ndcg_score(y, scores, k=1).item()
        ndcg3 += ndcg_score(y, scores, k=3).item()
        ndcg5 += ndcg_score(y, scores, k=5).item()
        ndcg10 += ndcg_score(y, scores, k=10).item()

n = len(test_groups)
print(f"Test NDCG@1  = {ndcg1/n:.4f}")
print(f"Test NDCG@3  = {ndcg3/n:.4f}")
print(f"Test NDCG@5  = {ndcg5/n:.4f}")
print(f"Test NDCG@10 = {ndcg10/n:.4f}")