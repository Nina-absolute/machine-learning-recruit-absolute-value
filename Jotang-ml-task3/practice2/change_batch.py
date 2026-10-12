"""
Task 3：前向传播与反向传播实践（小批量 Batch）
目标：观察换成小批量（N=4）后，计算公式和形状的变化。
数据源：sklearn.datasets.make_moons
随机种子：12
隐藏层神经元数量：4
输入特征数：2
Batch Size: 4

变量形状（Batch N=4）：
X:   (4, 2)    输入
W1:  (2, 4)    第一层权重
b1:  (1, 4)    第一层偏置
Z1:  (4, 4)    第一层线性输出
A1:  (4, 4)    ReLU 输出
W2:  (4, 1)    第二层权重
b2:  (1, 1)    第二层偏置
Z2:  (4, 1)    第二层线性输出
p:   (4, 1)    Sigmoid 输出
y:   (4, 1)    二分类标签
L:   ()        标量损失

梯度形状：
dL_dW1:  (2, 4)
dL_db1:  (1, 4)
dL_dW2:  (4, 1)
dL_db2:  (1, 1)
"""

import numpy as np
import torch
from sklearn.datasets import make_moons

np.random.seed(12)
torch.manual_seed(12)

N = 4  # Batch Size

# ==================== 1. 初始化数据与参数 ====================
X_data, y_data = make_moons(n_samples=10, noise=0.1, random_state=12)
X = X_data[0:N].astype(np.float32)                  # (4, 2)
y = y_data[0:N].reshape(N, 1).astype(np.float32)    # (4, 1)

W1 = np.random.randn(2, 4).astype(np.float32)       # (2, 4)
b1 = np.random.randn(1, 4).astype(np.float32)       # (1, 4)
W2 = np.random.randn(4, 1).astype(np.float32)       # (4, 1)
b2 = np.random.randn(1, 1).astype(np.float32)       # (1, 1)

# ==================== 2. NumPy 前向传播 (Batch) ====================
Z1 = X @ W1 + b1                   # (4, 4)
A1 = np.maximum(0, Z1)             # (4, 4)
Z2 = A1 @ W2 + b2                  # (4, 1)
p = 1 / (1 + np.exp(-Z2))          # (4, 1)
L = - (y * np.log(p) + (1 - y) * np.log(1 - p)).mean()  # ()

# ==================== 3. NumPy 手动反向传播 (Batch) ====================
# 核心修改：除以 N
dL_dZ2 = (p - y) / N                      # (4, 1)

dL_dW2 = A1.T @ dL_dZ2                    # (4, 1)
dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)  # (1, 1)

dL_dA1 = dL_dZ2 @ W2.T                    # (4, 4)
dL_dZ1 = dL_dA1 * (Z1 > 0)                # (4, 4)

dL_dW1 = X.T @ dL_dZ1                     # (2, 4)
dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)  # (1, 4)

print("NumPy Batch Loss:", L)

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
print("NumPy 手算与 PyTorch 自动求导对比 (Batch N=4)")
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