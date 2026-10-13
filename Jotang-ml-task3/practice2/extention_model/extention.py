import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from pathlib import Path

# ==================== 0. 超参数 ====================
SEED = 12
N_SAMPLES = 1000
NOISE = 0.2
RANDOM_STATE = 12
BATCH_SIZE = 4          # 沿用原文件的 BATCH_SIZE
HIDDEN_UNITS = 4        # 沿用原文件的 HIDDEN_UNITS
NUM_EPOCHS = 500
LR = 0.01

OUTPUTS_DIR = Path(__file__).resolve().parent

np.random.seed(SEED)

# ==================== 1. 初始化数据与参数 ====================
X_data, y_data = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=RANDOM_STATE)
X = X_data.astype(np.float32)                       # (N, 2)
y = y_data.reshape(N_SAMPLES, 1).astype(np.float32) # (N, 1)

# 划分训练集和测试集（8:2）
n_train = int(N_SAMPLES * 0.8)
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

# 沿用原文件的参数初始化方式
W1 = np.random.randn(2, HIDDEN_UNITS).astype(np.float32)     # (2, HIDDEN_UNITS)
b1 = np.random.randn(1, HIDDEN_UNITS).astype(np.float32)     # (1, HIDDEN_UNITS)
W2 = np.random.randn(HIDDEN_UNITS, 1).astype(np.float32)     # (HIDDEN_UNITS, 1)
b2 = np.random.randn(1, 1).astype(np.float32)                # (1, 1)

# ==================== 2. 训练循环 ====================
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(NUM_EPOCHS):
    # 每个 epoch 打乱训练集
    idx = np.random.permutation(n_train)
    X_train_shuffled = X_train[idx]
    y_train_shuffled = y_train[idx]

    epoch_loss = 0.0
    epoch_acc = 0.0
    n_batches = 0

    for i in range(0, n_train, BATCH_SIZE):
        X_batch = X_train_shuffled[i:i+BATCH_SIZE]   # (batch, 2)
        y_batch = y_train_shuffled[i:i+BATCH_SIZE]   # (batch, 1)

        # ---- 严格沿用原文件的前向传播 ----
        Z1 = X_batch @ W1 + b1                   # (batch, HIDDEN_UNITS)
        A1 = np.maximum(0, Z1)                   # (batch, HIDDEN_UNITS)
        Z2 = A1 @ W2 + b2                        # (batch, 1)
        p = 1 / (1 + np.exp(-Z2))                # (batch, 1)
        L = - (y_batch * np.log(p) + (1 - y_batch) * np.log(1 - p)).mean()  # ()

        # ---- 严格沿用原文件的反向传播 ----
        dL_dZ2 = (p - y_batch) / BATCH_SIZE      # (batch, 1) —— 注意：沿用原文件固定除以 BATCH_SIZE
        dL_dW2 = A1.T @ dL_dZ2                   # (HIDDEN_UNITS, 1)
        dL_db2 = np.sum(dL_dZ2, axis=0, keepdims=True)  # (1, 1)
        dL_dA1 = dL_dZ2 @ W2.T                   # (batch, HIDDEN_UNITS)
        dL_dZ1 = dL_dA1 * (Z1 > 0)               # (batch, HIDDEN_UNITS)
        dL_dW1 = X_batch.T @ dL_dZ1              # (2, HIDDEN_UNITS)
        dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)  # (1, HIDDEN_UNITS)

        # ---- 最基础的 SGD 更新 ----
        W1 -= LR * dL_dW1
        b1 -= LR * dL_db1
        W2 -= LR * dL_dW2
        b2 -= LR * dL_db2

        # 统计本 batch
        epoch_loss += L
        pred = (p > 0.5).astype(np.float32)
        epoch_acc += (pred == y_batch).mean()
        n_batches += 1

    # 本 epoch 平均训练指标
    train_loss = epoch_loss / n_batches
    train_acc = epoch_acc / n_batches

    # ---- 验证集前向传播（沿用相同公式）----
    Z1_val = X_test @ W1 + b1
    A1_val = np.maximum(0, Z1_val)
    Z2_val = A1_val @ W2 + b2
    p_val = 1 / (1 + np.exp(-Z2_val))
    val_loss = - (y_test * np.log(p_val) + (1 - y_test) * np.log(1 - p_val)).mean()
    val_pred = (p_val > 0.5).astype(np.float32)
    val_acc = (val_pred == y_test).mean()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f'epoch {epoch+1:03d}, '
              f'train_loss {train_loss:.4f} train_acc {train_acc:.4f}, '
              f'val_loss {val_loss:.4f} val_acc {val_acc:.4f}')

# ==================== 3. 绘制 loss / accuracy 曲线 ====================
epochs = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_loss'], label='train')
plt.plot(epochs, history['val_loss'], label='val')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_acc'], label='train')
plt.plot(epochs, history['val_acc'], label='val')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / 'numpy_curves.png', dpi=200)
plt.close()

# ==================== 4. 绘制二维决策边界 ====================
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

# 沿用相同的前向传播
Z1_grid = grid @ W1 + b1
A1_grid = np.maximum(0, Z1_grid)
Z2_grid = A1_grid @ W2 + b2
p_grid = 1 / (1 + np.exp(-Z2_grid))
pred = (p_grid > 0.5).astype(np.float32).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, pred, alpha=0.3, levels=[-0.5, 0.5, 1.5])
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', s=20)
plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Decision Boundary (NumPy)')
plt.savefig(OUTPUTS_DIR / 'numpy_decision_boundary.png', dpi=200)
plt.close()

print("\n图像已保存：numpy_curves.png, numpy_decision_boundary.png")