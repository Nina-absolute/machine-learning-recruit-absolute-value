# 使用 outer1.png 实测手写函数读取结果
from pathlib import Path
import numpy as np
from PIL import Image

# 定义卷积函数
# 仍然是卷积核 K（这里是三个二维卷积核）与图片 X（这里是 outer1.png）
def corr2d(X, K):
    kh, kw = K.shape

    out_h = X.shape[0] - kh + 1
    out_w = X.shape[1] - kw + 1

    Y = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = X[i:i + kh, j:j + kw]

            Y[i, j] = np.sum(region * K)

    return Y

# 定义用于处理 RGB 图像通道的函数
def conv2d_rgb(X, K):
    Y = np.zeros((
        X.shape[0] - K.shape[1] + 1,
        X.shape[1] - K.shape[2] + 1
    ))

    for c in range(X.shape[2]):
        Y += corr2d(X[:, :, c], K[c])
        # 依次处理每一个颜色通道，对应每一个卷积核
        # 此处三个通道加和到一张特征图里
        
    return Y

image_path = (
    Path(__file__).resolve().parent.parent / "images/outer1.png"
)
image = Image.open(image_path).convert("RGB")
X = np.array(image, dtype=np.float32)
# 读取图片，转数组

print("图片数组形状：", X.shape)
print("图片数据类型：", X.dtype)

K = np.array([
    [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ],

    [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ],

    [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]
], dtype=np.float32)
# 写出具体的卷积核

Y = conv2d_rgb(X, K)
# 得到最终特征图

print("卷积核形状：", K.shape)
print("特征图形状：", Y.shape)
print("特征图最大值：", Y.max())
print("特征图最小值：", Y.min())