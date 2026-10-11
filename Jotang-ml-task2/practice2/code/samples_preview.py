from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# 1. 设置数据集路径
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "train"

# 固定随机种子，保证每次抽样结果一致
random.seed(10)

# 2. 获取猫和狗的图片路径
cat_images = sorted(DATA_DIR.glob("cat.*.jpg"))
dog_images = sorted(DATA_DIR.glob("dog.*.jpg"))

print(f"猫图片数量：{len(cat_images)}")
print(f"狗图片数量：{len(dog_images)}")
print(f"训练集总图片数量：{len(cat_images) + len(dog_images)}")


# 3. 分别随机抽取 4 张猫图片和 4 张狗图片
sample_images = random.sample(cat_images, 4) + random.sample(dog_images, 4)
random.shuffle(sample_images)


# 4. 显示并保存抽样图片
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax, image_path in zip(axes.flat, sample_images):
    with Image.open(image_path) as image:
        width, height = image.size

        ax.imshow(image)
        ax.set_title(
            f"{image_path.name}\n{width} × {height}"
        )
        ax.axis("off")

plt.tight_layout()

# 创建图片保存目录
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 保存抽样结果
save_path = OUTPUT_DIR / "sample_images.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()


# 5. 统计图片尺寸
image_sizes = {}

for image_path in cat_images + dog_images:
    with Image.open(image_path) as image:
        size = image.size
        image_sizes[size] = image_sizes.get(size, 0) + 1

print("\n图片尺寸统计（宽 × 高）：")

for size, count in sorted(
    image_sizes.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]:
    print(f"{size[0]} × {size[1]}：{count} 张")

print(f"训练集中不同尺寸的数量：{len(image_sizes)}")