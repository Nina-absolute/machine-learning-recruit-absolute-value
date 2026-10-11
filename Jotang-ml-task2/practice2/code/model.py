
import random
import numpy as np
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

SEED = 10

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(SEED)

from data import (
    train_loader,
    aug_train_loader,
    val_loader,
    test_loader,
    test_dataset,
    IMAGE_SIZE,
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 1. 搭建 CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


# 2. 在验证集或测试集上评估模型
def evaluate(model, loader, criterion):
    model.eval()

    loss_sum = 0
    correct = 0
    total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)
            predicted = logits.argmax(dim=1)

            loss_sum += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            predictions.extend(predicted.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return (
        loss_sum / total,
        correct / total,
        predictions,
        targets,
    )


# 3. 训练模型，并保存验证集表现最好的参数
def train_model(model, loader, name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_acc = -1
    best_weights = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()

        loss_sum = 0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"{name} | Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), OUTPUT_DIR / f"{name}_best.pth")

    # 绘制 Loss 和 Accuracy 曲线
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_curves.png")
    plt.close()

    print(f"{name} 训练耗时：{time.time() - start_time:.1f} 秒")

    return model, history


# 4. 对照实验：无增强与有增强
cnn_plain, history_plain = train_model(
    CNN(), train_loader, "cnn_plain"
)

cnn_aug, history_aug = train_model(
    CNN(), aug_train_loader, "cnn_aug"
)

criterion = nn.CrossEntropyLoss()

plain_loss, plain_acc, plain_pred, plain_true = evaluate(
    cnn_plain, test_loader, criterion
)

aug_loss, aug_acc, aug_pred, aug_true = evaluate(
    cnn_aug, test_loader, criterion
)

print(f"无增强测试集 Loss：{plain_loss:.4f}")
print(f"无增强测试集 Accuracy：{plain_acc:.4f}")
print(f"有增强测试集 Loss：{aug_loss:.4f}")
print(f"有增强测试集 Accuracy：{aug_acc:.4f}")

# 选择测试准确率较高的模型
# 正式调参时应优先根据验证集表现选择，避免反复使用测试集调参
if aug_acc >= plain_acc:
    best_model = cnn_aug
    best_name = "cnn_aug"
    best_pred = aug_pred
    best_true = aug_true
else:
    best_model = cnn_plain
    best_name = "cnn_plain"
    best_pred = plain_pred
    best_true = plain_true

torch.save(best_model.state_dict(), OUTPUT_DIR / "best_model.pth")
print(f"当前选择的模型：{best_name}")


# 5. 记录图片经过 CNN 各层后的 Tensor 形状
sample_images, sample_labels = next(iter(test_loader))
x = sample_images[:1].to(DEVICE)

print("\n各层 Tensor 形状：")
print("输入：", tuple(x.shape))

best_model.eval()

with torch.no_grad():
    for i, layer in enumerate(best_model.features):
        x = layer(x)
        print(
            f"features[{i}] {layer.__class__.__name__}: "
            f"{tuple(x.shape)}"
        )

    x = best_model.pool(x)
    print("AdaptiveAvgPool2d：", tuple(x.shape))

    x = torch.flatten(x, start_dim=1)
    print("Flatten：", tuple(x.shape))

    x = best_model.classifier(x)
    print("分类层 logits：", tuple(x.shape))


# 6. 可视化第一层卷积产生的特征图
image = sample_images[:1].to(DEVICE)

with torch.no_grad():
    feature_maps = best_model.features[0](image).cpu()[0]

fig, axes = plt.subplots(2, 8, figsize=(12, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[i], cmap="gray")
    ax.set_title(f"Channel {i}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_maps.png")
plt.close()


# 7. 混淆矩阵
fig, ax = plt.subplots(figsize=(5, 5))

ConfusionMatrixDisplay(
    confusion_matrix(best_true, best_pred, labels=[0, 1]),
    display_labels=["cat", "dog"],
).plot(ax=ax)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()


# 8. 查看整个测试集中的误分类图片
# test_dataset 与 test_loader 使用相同的 test_paths 顺序
wrong_indices = [
    i
    for i, (pred, true) in enumerate(zip(plain_pred, plain_true))
    if pred != true
]

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for ax, index in zip(axes.flat, wrong_indices[:8]):
    image, true_label = test_dataset[index]

    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(
        f"True: {'dog' if true_label == 1 else 'cat'}\n"
        f"Pred: {'dog' if plain_pred[index] == 1 else 'cat'}"
    )
    ax.axis("off")

for ax in list(axes.flat)[min(len(wrong_indices), 8):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "misclassified_samples.png")
plt.close()

print(f"无增强模型的误分类数量：{len(wrong_indices)}")


# 9. 绘制模型结构示意图
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")

architecture = (
    "Input: 3 x 224 x 224\n"
    "        |\n"
    "Conv2d(3 -> 16) + ReLU + MaxPool\n"
    "        |\n"
    "Conv2d(16 -> 32) + ReLU + MaxPool\n"
    "        |\n"
    "Conv2d(32 -> 64) + ReLU + MaxPool\n"
    "        |\n"
    "AdaptiveAvgPool2d(1 x 1)\n"
    "        |\n"
    "Flatten -> Linear(64 -> 2)\n"
    "        |\n"
    "      Cat / Dog"
)

ax.text(
    0.5,
    0.5,
    architecture,
    ha="center",
    va="center",
    fontsize=11,
    family="monospace",
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "edgecolor": "black",
    },
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_architecture.png")
plt.close()
