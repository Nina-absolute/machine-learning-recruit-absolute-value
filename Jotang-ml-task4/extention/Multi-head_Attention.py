import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 超参数
SEED = 16
BATCH_SIZE = 1
SEQ_LEN = 4
D_MODEL = 8
NUM_HEADS = 2
D_K = D_MODEL // NUM_HEADS

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

# 随机生成输入数据
X = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

# 初始化Q, K, V, W_o对应的权重矩阵
W_q = torch.randn(D_MODEL, D_MODEL)
W_k = torch.randn(D_MODEL, D_MODEL)
W_v = torch.randn(D_MODEL, D_MODEL)
W_o = torch.randn(D_MODEL, D_MODEL)

# 计算Q, K, V并拆分为多头
Q = torch.matmul(X, W_q).view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_K).transpose(1, 2)
K = torch.matmul(X, W_k).view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_K).transpose(1, 2)
V = torch.matmul(X, W_v).view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, D_K).transpose(1, 2)

# 计算注意力分数（每个头并行计算）
scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D_K)

# 计算注意力权重
attn_weights = F.softmax(scores, dim=-1)

# 计算加权求和并拼接
context = torch.matmul(attn_weights, V)
context = context.transpose(1, 2).contiguous().view(BATCH_SIZE, SEQ_LEN, D_MODEL)

# 最终线性变换输出
output = torch.matmul(context, W_o)

# 输出注意力分数和权重矩阵的形状
print('Attention Scores Shape:', scores.shape)
print('Attention Weights Shape:', attn_weights.shape)

# 输出每个头的权重矩阵（打印）
for h in range(NUM_HEADS):
    print(f'\n--- Head {h+1} 权重矩阵 ---')
    print(attn_weights[0, h])

# 绘制热力图：每个头一行，两列（左：分数，右：权重）
fig, axes = plt.subplots(NUM_HEADS, 2, figsize=(10, 4 * NUM_HEADS))
if NUM_HEADS == 1:
    axes = axes.reshape(1, -1)  # 兼容单头情况

for h in range(NUM_HEADS):
    # 左图：注意力分数
    sns.heatmap(scores[0, h].detach().numpy(), annot=True, cmap='viridis', fmt='.2f', ax=axes[h, 0])
    axes[h, 0].set_title(f'Head {h+1} Scores')
    axes[h, 0].set_xlabel('Key')
    axes[h, 0].set_ylabel('Query')

    # 右图：注意力权重
    sns.heatmap(attn_weights[0, h].detach().numpy(), annot=True, cmap='viridis', fmt='.2f', ax=axes[h, 1])
    axes[h, 1].set_title(f'Head {h+1} Weights')
    axes[h, 1].set_xlabel('Key')
    axes[h, 1].set_ylabel('Query')

plt.tight_layout()
plt.savefig('multi_head_attention_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()