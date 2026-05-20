"""ListMLE

输入: 46 维特征
输出: 每个文档的相关性得分
评价: NDCG@k
"""

import warnings
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"

warnings.filterwarnings('ignore')

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

def process_split(split):
    X = np.array(split["features"])     # (N, 46)
    y = np.array(split["label"])        # (N,)
    qids = np.array(split["query_id"])  # (N,)
    return X, y, qids

# dataset = load_dataset("irds/letor40", "mq2007")
# dataset = load_dataset("philipphager/MSLR-WEB10k")
dataset = load_dataset("aletovv/MSLRWEB10K")

X_train, y_train, q_train = process_split(dataset["train"])
X_val, y_val, q_val = process_split(dataset["validation"])
X_test, y_test, q_test = process_split(dataset["test"])

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

def group_by_qid(X, y, qids):
    """按qid分组
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

train_groups = group_by_qid(X_train, y_train, q_train)
val_groups = group_by_qid(X_val, y_val, q_val)
test_groups = group_by_qid(X_test, y_test, q_test)

class DeepRankNet(nn.Module):
    def __init__(self, input_dim=46, hidden_dims=[128, 64, 32]):
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
    # 按标签从大到小排序
    sorted_idx = torch.argsort(labels, descending=True)
    scores_sorted = scores[sorted_idx]
    
    # ListMLE 公式
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
    n = len(s)
    total_loss = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if y[i] > y[j]:
                total_loss += torch.log(1 + torch.exp(s[j] - s[i]))
                count += 1
    return total_loss / count if count > 0 else 0.0

def ndcg_score(y_true, y_score, k=5):
    k = min(k, len(y_true))
    order = y_score.argsort(descending=True)[:k]
    y_true = y_true[order]

    gain = 2 ** y_true - 1
    discount = torch.log2(torch.arange(2, 2+k, dtype=torch.float32))
    
    dcg = (gain / discount).sum()
    ideal = torch.sort(y_true, descending=True)[0]
    idcg = (ideal / discount).sum()
    return dcg / idcg if idcg > 0 else 0.0

MODEL_TYPE = "LambdaRank"  # 可选：RankNet / LambdaRank
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepRankNet(input_dim=46).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
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

        if loss > 0:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    model.eval()
    ndcg_sum = 0.0
    with torch.no_grad():
        for group in val_groups:
            X = group['X'].to(device)
            y = group['y'].to(device)
            scores = model(X)
            ndcg_sum += ndcg_score(y, scores, k=5)
    ndcg_val = ndcg_sum / len(val_groups)

    train_loss_history.append(total_loss)
    val_ndcg_history.append(ndcg_val)
    writer.add_scalar("Loss/train", total_loss, epoch)
    writer.add_scalar("NDCG/val", ndcg_val, epoch)
    scheduler.step(ndcg_val)

    print(f"Epoch {epoch+1:2d} | loss: {total_loss:.4f} | NDCG@5 val: {ndcg_val:.4f}")

    # 保存最优 & 早停
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

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_loss_history)
plt.title("Train Loss")
plt.subplot(1,2,2)
plt.plot(val_ndcg_history)
plt.title("Val NDCG@5")
plt.tight_layout()
plt.show()

# 测试集评估
model.load_state_dict(torch.load("best_ltr_model.pth"))
model.eval()

ndcg1 = ndcg3 = ndcg5 = ndcg10 = 0.0
with torch.no_grad():
    for group in test_groups:
        X = group['X'].to(device)
        y = group['y'].to(device)
        scores = model(X)
        ndcg1 += ndcg_score(y, scores, k=1)
        ndcg3 += ndcg_score(y, scores, k=3)
        ndcg5 += ndcg_score(y, scores, k=5)
        ndcg10 += ndcg_score(y, scores, k=10)

n = len(test_groups)
print(f"Test NDCG@1  = {ndcg1/n:.4f}")
print(f"Test NDCG@3  = {ndcg3/n:.4f}")
print(f"Test NDCG@5  = {ndcg5/n:.4f}")
print(f"Test NDCG@10 = {ndcg10/n:.4f}")