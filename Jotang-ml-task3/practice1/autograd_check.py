import torch

# 定义需要计算梯度的输入
x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
y = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

# 前向计算
a = x * y
b = x ** 2
c = a + b
f = c ** 2

# 自动反向传播
f.backward()

print("f =", f.item())
print("df/dx =", x.grad.item())
print("df/dy =", y.grad.item())