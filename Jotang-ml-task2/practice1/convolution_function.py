# TO DO LIST：手写可处理灰度图的函数
import numpy as np

def corr2d(X, K):
# 输入图片 X 和卷积核 K
    kh, kw = K.shape
    # 得到卷积核的高度和宽度

    out_h = X.shape[0] - kh + 1
    out_w = X.shape[1] - kw + 1
    # 输出卷积核在图片中的实际滑动范围

    Y = np.zeros((out_h, out_w))
    # 创建一个形状为 (out_h, out_w) 的零矩阵

    for i in range(out_h):
        for j in range(out_w):
        # 实现先从左向右，再上到下滑动
            region = X[i:i + kh, j:j + kw]
            # 滑动（此处为 stride = 1）取 X 中卷积核要覆盖的局部区域：从第 i 行到第 (i + kh - 1 ) 行共 kh 行，列数同理
            Y[i, j] = np.sum(region * K)
            # 此处是取出的矩阵和卷积核对应位置相乘再相加，得到输出值加到 Y 对应位置上去

    return Y
    # 最终得到 X 关于 K 的特征图 Y
