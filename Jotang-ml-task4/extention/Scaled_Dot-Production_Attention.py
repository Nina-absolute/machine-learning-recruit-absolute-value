# 单头缩放点积注意力

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
D_K = 8

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

# 随机生成输入数据
X = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

# 初始化Q, K, V对应的权重矩阵
W_q = torch.randn(D_MODEL, D_K)
W_k = torch.randn(D_MODEL, D_K)
W_v = torch.randn(D_MODEL, D_K)

# 计算Q, K, V（矩阵乘法）
Q = torch.matmul(X, W_q)
K = torch.matmul(X, W_k)
V = torch.matmul(X, W_v)

# 计算注意力分数（Q乘以K的转置，除以根号D_K缩放）
scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D_K)

# 计算注意力权重（对分数进行Softmax归一化）
attn_weights = F.softmax(scores, dim=-1)

# 输出注意力分数和权重矩阵
print('Attention Scores Shape:', scores.shape)
print('Attention Weights Shape:', attn_weights.shape)
print('Attention Weights Matrix:')
print(attn_weights[0])

# 绘制注意力分数和权重矩阵的热力图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(scores[0].detach().numpy(), annot=True, cmap='viridis', fmt='.2f')
plt.title('Attention Scores')

plt.subplot(1, 2, 2)
sns.heatmap(attn_weights[0].detach().numpy(), annot=True, cmap='viridis', fmt='.2f')
plt.title('Attention Weights')

plt.tight_layout()
plt.savefig('single_head_attention_heatmap.png')
plt.show()