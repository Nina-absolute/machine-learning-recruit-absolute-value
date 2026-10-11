
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


# ==================== 1. 基本配置 ====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 当前文件位于 practice2/code/
ROOT_DIR = Path(__file__).resolve().parent

# 模型文件位于 practice2/outputs/model/best_model.pth
MODEL_PATH = ROOT_DIR.parent / "outputs" / "model" / "best_model.pth"

# 推理结果保存目录
OUTPUT_DIR = ROOT_DIR.parent / "outputs" / "predict_processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
CLASS_NAMES = ["猫", "狗"]

# 设置中文字体正常显示
from matplotlib.font_manager import FontProperties

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
chinese_font = FontProperties(fname=FONT_PATH)

plt.rcParams["axes.unicode_minus"] = False

# ==================== 2. 定义 CNN 结构 ====================
# 必须与训练 best_model.pth 时的 CNN 完全一致。

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# ==================== 3. 检查输入文件 ====================

if len(sys.argv) != 2:
    print("用法：python best_predict.py 图片路径")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print(f"错误：图片不存在：{image_path}")
    sys.exit(1)

if not MODEL_PATH.is_file():
    print(f"错误：找不到模型文件：{MODEL_PATH}")
    print("请先确认已经训练并保存 best_model.pth。")
    sys.exit(1)


# ==================== 4. 加载已保存的模型 ====================

model = CNN().to(DEVICE)

try:
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state_dict)
except (RuntimeError, KeyError) as error:
    print("模型权重加载失败。")
    print("请检查 best_predict.py 中的 CNN 是否与训练时完全一致。")
    print(f"详细错误：{error}")
    sys.exit(1)

model.eval()

print(f"已加载模型：{MODEL_PATH}")
print(f"推理设备：{DEVICE}")


# ==================== 5. 读取并预处理图片 ====================

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

try:
    original_image = Image.open(image_path).convert("RGB")
except Exception as error:
    print(f"图片读取失败：{error}")
    sys.exit(1)

processed_image = preprocess(original_image)


# ==================== 6. 执行推理 ====================

with torch.no_grad():
    image_tensor = processed_image.unsqueeze(0).to(DEVICE)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

predicted_class = int(probabilities.argmax().item())

cat_probability = probabilities[0].item()
dog_probability = probabilities[1].item()

print("\n========== 推理结果 ==========")
print(f"图片路径：{image_path}")
print(f"预测类别：{CLASS_NAMES[predicted_class]}")
print(f"猫的预测概率：{cat_probability:.4%}")
print(f"狗的预测概率：{dog_probability:.4%}")


# ==================== 7. 保存预处理后的图片 ====================

plt.figure(figsize=(5, 5))
plt.imshow(processed_image.permute(1, 2, 0))
plt.title(
    f"预测：{CLASS_NAMES[predicted_class]}\n"
    f"猫：{cat_probability:.2%}，狗：{dog_probability:.2%}",
    fontproperties=chinese_font
)
plt.axis("off")
plt.tight_layout()

image_name = image_path.stem
# 根据输入图片名称生成输出文件名
result_path = OUTPUT_DIR / f"{image_name}_processed.png"

plt.savefig(result_path, dpi=150, bbox_inches="tight")
plt.close()