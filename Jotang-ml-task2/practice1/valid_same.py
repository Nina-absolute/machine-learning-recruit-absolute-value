# TO DO LIST:
# 尝试不填充（valid）和适当填充（same）图片边缘，并比较输出尺寸与边缘像素的变化。

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


# 1. 二维相关运算
def corr2d(X, K, padding=0):
    kh, kw = K.shape

    # padding：在图片上下左右分别填充 0
    if padding > 0:
        X = np.pad(
            X,
            ((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0
        )

    out_h = X.shape[0] - kh + 1
    out_w = X.shape[1] - kw + 1

    Y = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            region = X[i:i + kh, j:j + kw]
            Y[i, j] = np.sum(region * K)

    return Y


# 2. RGB 图片：三个通道分别计算
def corr2d_rgb(X, K, padding=0):
    return np.stack(
        [corr2d(X[:, :, c], K, padding) for c in range(3)],
        axis=2
    )


# 3. 读取原来的图片 outer1.png
image_path = (
    Path(__file__).resolve().parent.parent / "images" / "outer1.png"
)

X = np.array(
    Image.open(image_path).convert("RGB"),
    dtype=np.float32
)

# 4. 定义 3×3 均值核
K = np.ones((3, 3), dtype=np.float32) / 9

# 5. valid：不填充
valid_result = corr2d_rgb(X, K, padding=0)

# 6. same：四周各填充 1 层 0
same_result = corr2d_rgb(X, K, padding=1)

# 7. padding=10：四周各填充 10 层 0
padding10_result = corr2d_rgb(X, K, padding=10)

# 8. 显示原图与三种填充方式的结果
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(X.astype(np.uint8))
axes[0].set_title(f"Original {X.shape[1]}x{X.shape[0]}")

axes[1].imshow(
    np.clip(valid_result, 0, 255).astype(np.uint8)
)
axes[1].set_title(
    f"valid {valid_result.shape[1]}x{valid_result.shape[0]}"
)

axes[2].imshow(
    np.clip(same_result, 0, 255).astype(np.uint8)
)
axes[2].set_title(
    f"same {same_result.shape[1]}x{same_result.shape[0]}"
)

axes[3].imshow(
    np.clip(padding10_result, 0, 255).astype(np.uint8)
)
axes[3].set_title(
    f"padding=10 {padding10_result.shape[1]}x{padding10_result.shape[0]}"
)

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("valid_same_result.png", dpi=150)
plt.show()

# 9. 输出结果信息
print("原图尺寸（高, 宽, 通道）：", X.shape)
print("valid 输出尺寸：", valid_result.shape)
print("same 输出尺寸：", same_result.shape)
print("padding=10 输出尺寸：", padding10_result.shape)

print("\nvalid 左上角 RGB 像素值：", valid_result[0, 0])
print("same 左上角 RGB 像素值：", same_result[0, 0])
print("padding=10 左上角 RGB 像素值：", padding10_result[0, 0])