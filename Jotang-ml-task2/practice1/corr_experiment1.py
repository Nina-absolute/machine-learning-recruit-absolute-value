import numpy as np
from PIL import Image

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


def conv2d_rgb(X, K):
    Y = np.zeros((
        X.shape[0] - K.shape[1] + 1,
        X.shape[1] - K.shape[2] + 1
    ))

    for c in range(X.shape[2]):
        Y += corr2d(X[:, :, c], K[c])

    return Y

image_path = "./../images/outer1.png"

image = Image.open(image_path).convert("RGB")

X = np.array(image, dtype=np.float32)


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

Y = conv2d_rgb(X, K)

print("卷积核形状：", K.shape)
print("特征图形状：", Y.shape)
print("特征图最大值：", Y.max())
print("特征图最小值：", Y.min())