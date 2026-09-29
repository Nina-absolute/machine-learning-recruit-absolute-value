import torch
torch.device("cuda")
# PyTorch: 我要使用 CUDA 这个 GPU 计算后端。
mat1 = torch.tensor([
    [1., 2.],
    [3., 4.]
]
)
mat2 = torch.tensor([
    [1., 2.],
    [3., 4.]
])

print("Result is: ", mat1 @ mat2)