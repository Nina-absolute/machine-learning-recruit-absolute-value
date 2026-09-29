
# 编写函数计算平均分，并找出最高分；
def average(scores):
    return sum(scores.values()) / len(scores)

def find_highest(scores):
    return max(scores.values())
# 使用列表或字典记录几组姓名与成绩
names = ["甲","乙","丙","丁"]

scores = {
    "甲": 99,
    "乙": 88,
    "丙": 77,
    "丁": 66
}
    
print("NAMES:", names)
print("SCORES:", scores)
print("AVERAGE:", average(scores))
print("HIGHEST:", find_highest(scores))

# 使用 NumPy 创建两个形状合适的矩阵，完成矩阵乘法
import numpy as np
mat1 = np.array([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])
mat2 = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])

result = mat1 @ mat2
# 输出计算结果以及输入、输出矩阵的形状
print("mat1:", mat1)
print("mat2:", mat2)
print("Result:", result)
print("Shape of mat1:", mat1.shape)
print("Shape of mat2:", mat2.shape)
print("Shape of result:", result.shape)

# 做一次张量计算
import torch
mat3 = torch.tensor([
    [1., 2.],
    [3., 4.]
])
mat4 = torch.tensor([
    [1., 2.],
    [3., 4.]
])

result_tensor = mat3 @ mat4
print("Result of tensor:", result_tensor)


