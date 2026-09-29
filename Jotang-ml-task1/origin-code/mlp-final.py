import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader


# 1. 生成数据，打包数据
X, y = make_moons(n_samples=2000, noise=0.2, random_state=20)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=20, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=20, stratify=y_temp)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.long)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=64)


# 2. MLP 主体部分
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

net = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

best_val_loss = float('inf')
save_path = 'best_model.pt'
history = {'train_loss': [], 'train_acc': [],
           'val_loss': [], 'val_acc': []}

for epoch in range(100)
for epoch in range(100):
    # ---- 4.1 训练一个 epoch ----
    net.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = net(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        train_correct += (logits.argmax(1) == y_batch).sum().item()
        train_total += X_batch.size(0)
    train_loss /= train_total
    train_acc = train_correct / train_total

    # ---- 4.2 在验证集上评估 ----
    net.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = net(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            val_correct += (logits.argmax(1) == y_batch).sum().item()
            val_total += X_batch.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total


# 5

net.load_state_dict()