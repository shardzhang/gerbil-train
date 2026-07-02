import numpy as np
import matplotlib.pyplot as plt

# 超参数配置
base_lr = 0.01
warmup_steps = 1000
decay_rate = -1.2
learning_rate_min = 1e-6
total_steps = 8000

# 生成step序列
steps = np.arange(0, total_steps)
lr_list = []

for step in steps:
    if step < warmup_steps:
        # 线性warmup上升
        lr = base_lr * (step + 1) / warmup_steps
    else:
        # 指数衰减
        exponent = decay_rate * (step + 1 - warmup_steps) / warmup_steps
        lr = base_lr * np.exp(exponent)
    # 限制最小学习率
    lr = max(lr, learning_rate_min)
    lr_list.append(lr)

# 绘图
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(steps, lr_list, color='#2E86AB', linewidth=2.5, label='Learning Rate Schedule')

# 标记warmup分界点
# 垂直线
plt.axvline(x=warmup_steps, color='#A23B72', linestyle='--', alpha=0.7, label=f'Warmup End (step={warmup_steps})')
# 水平线
plt.axhline(y=learning_rate_min, color='#F18F01', linestyle=':', alpha=0.8, label=f'Min LR = {learning_rate_min}')

plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('LR Schedule: Linear Warmup + Exponential Decay with Floor', fontsize=14, pad=15)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()