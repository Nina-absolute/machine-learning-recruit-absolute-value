# 单样本计算

import numpy as np
import torch
from sklearn.datasets import make_moons

# ==================== 0. 超参数 ====================
SEED = 12
BATCH_SIZE = 1
N_SAMPLES = 10
NOISE = 0.1
RANDOM_STATE = 12
HIDDEN_UNITS = 4

np.random.seed(SEED)
torch.manual_seed(SEED)


# ==================== 1. 初始化数据与参数 ====================
# 生成 make_moons 数据，取第 1 个样本
X_data, y_data = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=RANDOM_STATE)
X = X_data[0:BATCH_SIZE].astype(np.float32)                  # (1, 2)
y = y_data[0:BATCH_SIZE].reshape(BATCH_SIZE, 1).astype(np.float32)  # (1, 1)

W1 = np.random.randn(2, HIDDEN_UNITS).astype(np.float32)     # (2, 4)
b1 = np.random.randn(1, HIDDEN_UNITS).astype(np.float32)     # (1, 4)
W2 = np.random.randn(HIDDEN_UNITS, 1).astype(np.float32)     # (4, 1)
b2 = np.random.randn(1, 1).astype(np.float32)                # (1, 1)

# ==================== 2. NumPy 前向传播 ====================
Z1 = X @ W1 + b1                   # (1, 4)
A1 = np.maximum(0, Z1)             # (1, 4)
Z2 = A1 @ W2 + b2                  # (1, 1)
p = 1 / (1 + np.exp(-Z2))          # (1, 1)
L = - (y * np.log(p) + (1 - y) * np.log(1 - p)).mean()  # ()

# ==================== 3. NumPy 手动反向传播 ====================
dL_dp = (p - y) / (p * (1 - p))           # (1, 1)
dL_dZ2 = dL_dp * p * (1 - p)              # (1, 1)

dL_dW2 = A1.T @ dL_dZ2                    # (4, 1)
dL_db2 = dL_dZ2                           # (1, 1)

dL_dA1 = dL_dZ2 @ W2.T                    # (1, 4)
dL_dZ1 = dL_dA1 * (Z1 > 0)                # (1, 4)

dL_dW1 = X.T @ dL_dZ1                     # (2, 4)
dL_db1 = dL_dZ1                           # (1, 4)

print("NumPy 前向传播 Loss:", L)

# ==================== 4. PyTorch 前向与自动求导 ====================
X_pt = torch.tensor(X, dtype=torch.float32)
y_pt = torch.tensor(y, dtype=torch.float32)
W1_pt = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
b1_pt = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
W2_pt = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
b2_pt = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

Z1_pt = X_pt @ W1_pt + b1_pt
A1_pt = torch.relu(Z1_pt)
Z2_pt = A1_pt @ W2_pt + b2_pt
p_pt = torch.sigmoid(Z2_pt)

loss_pt = torch.nn.functional.binary_cross_entropy(p_pt, y_pt)
loss_pt.backward()

dL_dW1_pt = W1_pt.grad.numpy()
dL_db1_pt = b1_pt.grad.numpy()
dL_dW2_pt = W2_pt.grad.numpy()
dL_db2_pt = b2_pt.grad.numpy()

# ==================== 5. 结果对比 ====================
print("\n" + "="*40)
print("NumPy 手算与 PyTorch 自动求导对比 (单样本)")
print("="*40)

def compare(name, np_grad, pt_grad):
    max_err = np.max(np.abs(np_grad - pt_grad))
    print(f"\n--- {name} ---")
    print(f"形状: NumPy {np_grad.shape}, PyTorch {pt_grad.shape}")
    print(f"最大绝对误差: {max_err:.8e}")
    print("✅ 结果一致" if max_err < 1e-6 else "❌ 结果不一致")

compare("dL_dW1", dL_dW1, dL_dW1_pt)
compare("dL_db1", dL_db1, dL_db1_pt)
compare("dL_dW2", dL_dW2, dL_dW2_pt)
compare("dL_db2", dL_db2, dL_db2_pt)