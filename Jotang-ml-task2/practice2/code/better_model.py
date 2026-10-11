from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from torch.optim import Adam
import torch


CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"
OUTPUT_DIR = CODE_DIR.parent / "outputs" / "better_model"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test1"

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 10
EPOCHS = 10
LR = 1e-3

random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 统一尺寸、裁剪、转成 Tensor（用于验证 / 测试）
basic_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

# 训练时使用的数据增强：随机裁剪 + 翻转 + 旋转 + 颜色扰动 + 随机遮挡
augment_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

# 收集训练图片及标签：猫为 0，狗为 1
image_paths, labels = [], []

for path in TRAIN_DIR.iterdir():
    if not path.is_file():
        continue

    name = path.name.lower()
    if name.startswith("cat."):
        image_paths.append(path)
        labels.append(0)
    elif name.startswith("dog."):
        image_paths.append(path)
        labels.append(1)

print(f"猫：{labels.count(0)} 张")
print(f"狗：{labels.count(1)} 张")
print(f"训练图片总数：{len(image_paths)} 张")

# 先划分测试集，再从剩余数据中划分验证集
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.1, random_state=SEED,
    stratify=labels
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=1/9, random_state=SEED,
    stratify=train_labels
)


class CatDogDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.labels is None:
            return image, self.paths[index].name

        return image, self.labels[index]


# 训练集用增强 transform，验证 / 测试集用 basic transform
aug_train_dataset = CatDogDataset(
    train_paths, train_labels, augment_transform
)
val_dataset = CatDogDataset(val_paths, val_labels, basic_transform)
test_dataset = CatDogDataset(test_paths, test_labels, basic_transform)

aug_train_loader = DataLoader(
    aug_train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 官方测试集没有标签
official_paths = sorted(
    [p for p in TEST_DIR.iterdir() if p.is_file()],
    key=lambda p: int(p.stem) if p.stem.isdigit() else p.name
)
official_dataset = CatDogDataset(
    official_paths, transform=basic_transform
)
official_loader = DataLoader(
    official_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# 展示随机裁剪、翻转、旋转、颜色扰动、随机遮挡后的增强样本
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax, path in zip(axes.flat, random.choices(train_paths, k=8)):
    image = Image.open(path).convert("RGB")
    image = augment_transform(image).permute(1, 2, 0).numpy()
    ax.imshow(image)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "better_augmented_samples.png")
plt.close()

# 检查一个 batch 的形状
images, batch_labels = next(iter(aug_train_loader))
print("一个 batch 的图片形状：", images.shape)
print("一个 batch 的标签形状：", batch_labels.shape)


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


# 2. 评估
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    predictions, targets = [], []

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

    return loss_sum / total, correct / total, predictions, targets


# 3. 训练
def train_model(model, loader, name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    best_acc = -1
    best_weights = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        loss_sum, correct, total = 0, 0, 0

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

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

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

    if best_weights is not None:
        model.load_state_dict(best_weights)

    torch.save(model.state_dict(), OUTPUT_DIR / f"{name}_best.pth")

    # 绘制 Loss / Accuracy 曲线
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


# 4. 训练模型
model, history = train_model(CNN(), aug_train_loader, "cnn_aug")

criterion = nn.CrossEntropyLoss()
test_loss, test_acc, test_pred, test_true = evaluate(
    model, test_loader, criterion)

print(f"测试集 Loss：{test_loss:.4f}")
print(f"测试集 Accuracy：{test_acc:.4f}")


# 5. 混淆矩阵
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(
    confusion_matrix(test_true, test_pred, labels=[0, 1]),
    display_labels=["cat", "dog"],
).plot(ax=ax)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()


# 6. 误分类样本
wrong_indices = [
    i for i, (pred, true) in enumerate(zip(test_pred, test_true))
    if pred != true
]

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for ax, index in zip(axes.flat, wrong_indices[:8]):
    image, true_label = test_dataset[index]
    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(
        f"True: {'dog' if true_label == 1 else 'cat'}\n"
        f"Pred: {'dog' if test_pred[index] == 1 else 'cat'}"
    )
    ax.axis("off")

for ax in list(axes.flat)[min(len(wrong_indices), 8):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "misclassified_samples.png")
plt.close()

print(f"误分类数量：{len(wrong_indices)}")