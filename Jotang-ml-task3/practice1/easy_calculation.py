import numpy as np

# 前向计算
x = np.float64(2.0)
y = np.float64(3.0)

a = x * y
b = x ** 2
c = a + b
f = c ** 2

# 反向传播：从 f 开始，沿计算图反向计算梯度
df_dc = 2 * c

df_da = df_dc
df_db = df_dc

df_dx = df_da * y + df_db * (2 * x)
df_dy = df_da * x

print("f =", f)
print("df/dx =", df_dx)
print("df/dy =", df_dy)