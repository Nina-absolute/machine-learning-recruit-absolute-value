from PIL import Image
import numpy as np
import torch

outer1 = Image.open("./../images/outer1.png").convert("RGB")
# 使用 Pillow 读取图片，转成 RGB 模式

arr = np.array(outer1)
tensor = torch.from_numpy(arr).permute(2, 0, 1)
# 转为 NumPy 数组和 PyTorch Tensor, 并且将 HWC 转为 PyTorch 常用的 CHW

print("Numpy shape:", arr.shape)
print("Numpy dtype:", arr.dtype)
print("Numpy min:", arr.min())
print("Numpy max:", arr.max())
# 打印 Numpy 的 shape、dtype、最大值和最小值

print("Tensor shape:", tensor.shape)
print("Tensor dtype:", tensor.dtype)
print("Tensor min:", tensor.min().item())
print("Tensor max:", tensor.max().item())
# 打印 Tensor 的 shape、dtype、最大值和最小值

y, x = 0, 0
print("Numpy [0, 0] 像素 RGB 数值：", arr[y, x])
print("Tensor [0, 0] 像素 RGB 数值：", tensor[:, y, x])
# 打印 [0, 0] 这个像素的 RGB 值