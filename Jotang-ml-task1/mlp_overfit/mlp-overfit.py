import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# ---------- 超参数（刻意制造过拟合） ----------
N_SAMPLES = 2000
NOISE = 0.20
RANDOM_STATE = 30
BATCH_SIZE = 64
HIDDEN_UNITS = 512          # 原来 16，现在 512，容量大增
NUM_EPOCHS = 2000           # 原来 100，现在 2000，训练更久
LR = 0.01
SEED = 30

# ---------- 数据处理 ----------
X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=RANDOM_STATE)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)


# 转张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.long)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

# ---------- 定义更大的模型（两层隐藏层） ----------
class MLP(nn.Module):
    def __init__(self, hidden_units=512, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),   # 额外一层
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes))

    def forward(self, x):
        return self.net(x)

# ---------- 训练准备 ----------
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = MLP(HIDDEN_UNITS).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = {'train_loss': [], 'train_acc': [],
           'val_loss': [], 'val_acc': []}

# 注意：这里我们不再保存最佳模型，也不加载最佳模型
# 目的是让过拟合保留下来

for epoch in range(NUM_EPOCHS):
    # ---- 训练 ----
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        train_correct += (logits.argmax(1) == y_batch).sum().item()
        train_total += X_batch.size(0)
    train_loss /= train_total
    train_acc = train_correct / train_total

    # ---- 验证 ----
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            val_correct += (logits.argmax(1) == y_batch).sum().item()
            val_total += X_batch.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f'epoch {epoch+1:04d}, '
              f'train_loss {train_loss:.4f} train_acc {train_acc:.4f}, '
              f'val_loss {val_loss:.4f} val_acc {val_acc:.4f}')

# ---------- 绘制 loss / accuracy 曲线 ----------
epochs = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_loss'], label='train')
plt.plot(epochs, history['val_loss'], label='val')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
plt.title('Loss (Overfitting)')

plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_acc'], label='train')
plt.plot(epochs, history['val_acc'], label='val')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()
plt.title('Accuracy (Overfitting)')

plt.tight_layout()
plt.savefig('overfit_curves.png', dpi=200)
plt.show()

# ---------- 最终测试（使用过拟合的模型） ----------
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        test_correct += (logits.argmax(1) == y_batch).sum().item()
        test_total += X_batch.size(0)
test_loss /= test_total
test_acc = test_correct / test_total
print(f'\nFinal test loss {test_loss:.4f}, test acc {test_acc:.4f}')