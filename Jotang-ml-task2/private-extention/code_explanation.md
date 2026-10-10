# 图像卷积处理代码解析

说明：本文假设 `corr2d(X, K)` 已经实现，因此不重复讲解其内部循环。

## 1. 导入库

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
```

- `numpy`：创建卷积核、操作数组和计算边缘强度。
- `matplotlib.pyplot`：展示并保存对比图。
- `PIL.Image`：读取图片并转换为 RGB 格式。

## 2. 处理 RGB 图片

```python
def corr2d_rgb(X, K):
    return np.stack(
        [corr2d(X[:, :, c], K[c]) for c in range(3)],
        axis=2
    )
```

彩色图片的形状为 `(H, W, 3)`，三个通道分别对应 R、G、B。

- `X[:, :, c]`：提取第 `c` 个颜色通道。
- `K[c]`：取出对应通道的二维卷积核。
- `corr2d(...)`：计算该通道的输出。
- `np.stack(..., axis=2)`：沿第三个维度重新组合三个结果。

## 3. 读取图片

```python
X = np.array(
    Image.open("test.jpg").convert("RGB"),
    dtype=np.float32
)
```

- `Image.open(...)`：打开图片文件。
- `convert("RGB")`：确保图片有三个颜色通道。
- `np.array(...)`：把图片转换成 NumPy 数组。
- `dtype=np.float32`：使用浮点数保存像素，方便后续运算。

## 4. 创建卷积核

### 均值核

```python
mean_kernel = np.ones((3, 3)) / 9
```

创建一个所有元素均为 1 的矩阵，再除以 9，使权重和等于 1。

### 高斯核

```python
x = np.arange(-2, 3)
xx, yy = np.meshgrid(x, x)
gaussian_kernel = np.exp(-(xx**2 + yy**2) / 2)
gaussian_kernel /= gaussian_kernel.sum()
```

- `np.arange(-2, 3)`：生成 `-2, -1, 0, 1, 2`。
- `np.meshgrid(...)`：生成二维坐标网格。
- `np.exp(...)`：根据高斯函数计算每个位置的权重。
- 除以 `sum()`：让全部权重之和等于 1。

这里对应 \(\sigma=1\)。

### 锐化核与 Sobel 核

直接填写矩阵即可。矩阵中的正负权重决定不同的局部变化如何影响输出。

## 5. 应用模糊和锐化

```python
def apply_rgb_kernel(X, kernel):
    K = np.stack([kernel] * 3)
    return corr2d_rgb(X, K)
```

`[kernel] * 3` 创建三个相同卷积核的列表，`np.stack(...)` 将它们组合成形状为 `(3, kh, kw)` 的数组。

这样，同一个二维核就能分别应用到 RGB 三个通道。

`mean_result`、`gaussian_result` 和 `sharpen_result` 分别保存三种处理结果。

## 6. Sobel 边缘检测

```python
gray = (
    0.299 * X[:, :, 0]
    + 0.587 * X[:, :, 1]
    + 0.114 * X[:, :, 2]
)

gx = corr2d(gray, sobel_x)
gy = corr2d(gray, sobel_y)
edge = np.sqrt(gx**2 + gy**2)
```

先将 RGB 转换成灰度图，再分别计算两个方向的灰度变化。

- `gx`：水平方向的梯度响应。
- `gy`：垂直方向的梯度响应。
- `gx**2 + gy**2`：两个方向响应的平方和。
- `np.sqrt(...)`：开平方，得到合成边缘强度。

## 7. 归一化显示

```python
def show_gray(A):
    A = np.abs(A)
    if A.max() == 0:
        return np.zeros_like(A)
    return A / A.max() * 255
```

Sobel 输出可能为负数，也可能超出 0～255。

- `np.abs(A)`：取绝对值，显示边缘强度。
- `A.max()`：找到最大强度。
- 除以最大值再乘 255：将数值缩放到 0～255。
- 最大值为 0 时返回全零数组，避免除以 0。

这一步仅用于显示，不改变前面的卷积计算。

## 8. 展示与保存结果

```python
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
```

创建 2 行 3 列的子图，分别展示原图、三种处理结果、Sobel-X 和合成边缘图。

```python
ax.imshow(img)
ax.set_title(title)
ax.axis("off")
```

- `imshow(...)`：显示图片。
- `set_title(...)`：设置标题。
- `axis("off")`：隐藏坐标轴。

最后使用 `plt.savefig(...)` 保存结果，并通过 `plt.show()` 显示窗口。


## 关于四种卷积核
# 四种卷积核的分析

## 1. 均值模糊（Mean Blur）

卷积核：

\[
K=\frac{1}{9}
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}
\]

### 原理

将局部 \(3\times3\) 区域内的 9 个像素相加，再除以 9，作为输出像素的值。

### 预期结果

- 图像变得平滑。
- 细小纹理和部分噪声减弱。
- 物体边缘也会变模糊。

### 原因

局部像素被平均后，较大的亮度差异会缩小。

---

## 2. 高斯模糊（Gaussian Blur）

使用二维高斯函数生成 \(5\times5\) 卷积核：

\[
G(x,y)=e^{-\frac{x^2+y^2}{2\sigma^2}}
\]

代码使用 \(\sigma=1\)，并将核归一化，使全部权重之和为 1。

### 原理

距离中心越近的像素通常权重越大，距离越远的像素权重越小。

### 预期结果

- 图像变得平滑。
- 细节和噪声减弱。
- 与均值模糊相比，通常能产生更自然的平滑效果。

### 原因

高斯核不是对所有邻近像素一视同仁，而是根据距离分配不同权重。

---

## 3. 锐化（Sharpen）

卷积核：

\[
K=
\begin{bmatrix}
0&-1&0\\
-1&5&-1\\
0&-1&0
\end{bmatrix}
\]

### 原理

中心像素的权重为 5，上下左右邻居的权重为 -1。

### 预期结果

- 边缘和纹理更突出。
- 图像看起来更清晰。
- 噪声可能被放大，强边缘处可能出现过亮或过暗的区域。

### 原因

计算结果会加强中心像素与周围像素之间的差异。

---

## 4. Sobel 边缘检测

### Sobel-X

\[
K_x=
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix}
\]

它估计图像在水平方向上的灰度变化，因此通常突出垂直边缘。

### Sobel-Y

\[
K_y=
\begin{bmatrix}
-1&-2&-1\\
0&0&0\\
1&2&1
\end{bmatrix}
\]

它估计图像在垂直方向上的灰度变化，因此通常突出水平边缘。

### 合成边缘图

将两个方向的响应合并：

\[
M=\sqrt{G_x^2+G_y^2}
\]

### 预期结果

- 物体轮廓和明显的灰度变化区域变亮。
- 平坦区域通常较暗。
- 合成结果能同时反映两个方向的边缘。

### 注意

显示前对边缘强度进行归一化只是为了便于观察，并不保留原始响应的绝对数值。