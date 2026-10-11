
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"
OUTPUT_DIR = CODE_DIR.parent / "outputs"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test1"

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 10

random.seed(SEED)
torch.manual_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 统一尺寸、裁剪、转成 Tensor
basic_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

# 训练时使用的数据增强
augment_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
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


train_dataset = CatDogDataset(train_paths, train_labels, basic_transform)
aug_train_dataset = CatDogDataset(
    train_paths, train_labels, augment_transform
)
val_dataset = CatDogDataset(val_paths, val_labels, basic_transform)
test_dataset = CatDogDataset(test_paths, test_labels, basic_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
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

# 展示预处理前后的图片
sample_paths = random.sample(
    train_paths, min(6, len(train_paths))
)
fig, axes = plt.subplots(len(sample_paths), 2, figsize=(7, 3 * len(sample_paths)))
if len(sample_paths) == 1:
    axes = [axes]

for i, path in enumerate(sample_paths):
    original = Image.open(path).convert("RGB")
    processed = basic_transform(original).permute(1, 2, 0).numpy()

    axes[i][0].imshow(original)
    axes[i][0].set_title(f"Original: {original.size}")
    axes[i][1].imshow(processed)
    axes[i][1].set_title(f"Processed: {IMAGE_SIZE}x{IMAGE_SIZE}")

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "preprocessing_comparison.png")
plt.close()

# 展示随机水平翻转、随机裁剪后的增强样本
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax, path in zip(axes.flat, random.choices(train_paths, k=8)):
    image = Image.open(path).convert("RGB")
    image = augment_transform(image).permute(1, 2, 0).numpy()
    ax.imshow(image)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "augmented_samples.png")
plt.close()

# 检查一个 batch 的形状
images, batch_labels = next(iter(train_loader))
print("一个 batch 的图片形状：", images.shape)
print("一个 batch 的标签形状：", batch_labels.shape)
