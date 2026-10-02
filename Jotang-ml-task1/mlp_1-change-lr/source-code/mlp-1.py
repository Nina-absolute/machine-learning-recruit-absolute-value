# 对照实验1：
# 改变学习率：将 LR = 0.01 变为 LR = 0.1

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


N_SAMPLES = 2000
NOISE = 0.20
RANDOM_STATE = 30
BATCH_SIZE = 32
HIDDEN_UNITS = 16
NUM_EPOCHS = 100
LR = 0.1
SEED = 30


X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=RANDOM_STATE)


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE,
    stratify=y_temp)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.long)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)


class MLP(nn.Module):

    def __init__(self, hidden_units=HIDDEN_UNITS, num_classes=2):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes))

    def forward(self, x):
        return self.net(x)




torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


output_dir = Path(__file__).parent.parent / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / 'best_model.pt'


model = MLP(HIDDEN_UNITS).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)



history = {'train_loss': [], 'train_acc': [],
           'val_loss': [], 'val_acc': []}

best_val_loss = float('inf')


for epoch in range(NUM_EPOCHS):

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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'epoch {epoch+1:03d}, '
              f'train_loss {train_loss:.4f} train_acc {train_acc:.4f}, '
              f'val_loss {val_loss:.4f} val_acc {val_acc:.4f}')


model.load_state_dict(torch.load(save_path))
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
test_preds, test_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        test_correct += (logits.argmax(1) == y_batch).sum().item()
        test_total += X_batch.size(0)
        test_preds.append(logits.argmax(1).cpu())
        test_labels.append(y_batch.cpu())
test_loss /= test_total
test_acc = test_correct / test_total
test_preds = torch.cat(test_preds).numpy()
test_labels = torch.cat(test_labels).numpy()
print(f'\ntest loss {test_loss:.4f}, test acc {test_acc:.4f}')


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
plt.savefig(output_dir / 'curves.png', dpi=200)
plt.close()


X_np = X_test.numpy()
y_np = y_test.numpy()

x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                    dtype=torch.float32).to(device)
with torch.no_grad():
    pred = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, pred, alpha=0.3, levels=[-0.5, 0.5, 1.5])
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, edgecolors='k', s=20)
plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Decision Boundary')
plt.savefig(output_dir / 'decision_boundary.png', dpi=200)
plt.close()


cm = confusion_matrix(test_labels, test_preds)
ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(values_format='d')
plt.title('Confusion Matrix')
plt.savefig(output_dir / 'confusion_matrix.png', dpi=200)
plt.close()


wrong = np.where(test_labels != test_preds)[0][:5]

plt.figure(figsize=(6, 5))
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, alpha=0.25)
if len(wrong) > 0:
    plt.scatter(X_np[wrong, 0], X_np[wrong, 1],
                facecolors='none', edgecolors='k', s=120)
plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Misclassified Samples')
plt.savefig(output_dir / 'misclassified.png', dpi=200)
plt.close()

print('\nMisclassified samples:')
for i in wrong:
    print(f'  x=({X_np[i,0]:.3f}, {X_np[i,1]:.3f}), '
          f'true={test_labels[i]}, pred={test_preds[i]}')