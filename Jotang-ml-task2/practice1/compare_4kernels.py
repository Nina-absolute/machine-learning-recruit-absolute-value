import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


# 1. RGB 图片：三个通道分别计算
def corr2d(X, K):
# 输入图片 X 和卷积核 K
    kh, kw = K.shape

    out_h = X.shape[0] - kh + 1
    out_w = X.shape[1] - kw + 1

    Y = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = X[i:i + kh, j:j + kw]
            Y[i, j] = np.sum(region * K)


    return Y

def corr2d_rgb(X, K):
    return np.stack(
        [corr2d(X[:, :, c], K[c]) for c in range(3)],
        axis=2
    )


# 2. 读取图片
image_path = (
    Path(__file__).resolve().parent.parent / "images/outer1.png"
)
X = np.array(
    Image.open(image_path).convert("RGB"),
    dtype=np.float32
)

# 3. 定义四种卷积核
# 均值模糊：全为 1 的 3×3 核
mean_kernel = np.ones((3, 3), dtype=np.float32) / 9

# 高斯模糊：根据高斯函数生成 5×5 核
x = np.arange(-2, 3)
xx, yy = np.meshgrid(x, x)
gaussian_kernel = np.exp(-(xx**2 + yy**2) / 2)
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

# 锐化：
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)


# 4. 同一个核应用于 RGB 三个通道
def apply_rgb_kernel(X, kernel):
    K = np.stack([kernel] * 3)
    return corr2d_rgb(X, K)


mean_result = apply_rgb_kernel(X, mean_kernel)
gaussian_result = apply_rgb_kernel(X, gaussian_kernel)
sharpen_result = apply_rgb_kernel(X, sharpen_kernel)


# 5. Sobel 边缘检测：先转灰度，再计算两个方向
gray = (
    0.299 * X[:, :, 0]
    + 0.587 * X[:, :, 1]
    + 0.114 * X[:, :, 2]
)

gx = corr2d(gray, sobel_x)
gy = corr2d(gray, sobel_y)
edge = np.sqrt(gx**2 + gy**2)

# 为保证图像卷积处理后，所有结果仍然在原来的像素范围内，遂引入归一化
# 6. 归一化到 0~255，仅用于显示
def show_gray(A):
    A = np.abs(A)
    if A.max() == 0:
        return np.zeros_like(A)
    return A / A.max() * 255


# 7. 对比原图和四种处理结果
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

images = [
    (X.astype(np.uint8), "Original"),
    (np.clip(mean_result, 0, 255).astype(np.uint8), "Mean Blur"),
    (np.clip(gaussian_result, 0, 255).astype(np.uint8), "Gaussian Blur"),
    (np.clip(sharpen_result, 0, 255).astype(np.uint8), "Sharpen"),
    (show_gray(gx), "Sobel-X"),
    (show_gray(edge), "Combined Sobel")
]

for ax, (img, title) in zip(axes.flat, images):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("compare_result.png", dpi=150)
plt.show()