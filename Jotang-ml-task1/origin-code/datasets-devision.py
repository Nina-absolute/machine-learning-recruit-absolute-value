# 本模块完成的任务：
# 1. 用 make_moons 生成数据
# 2. 用 train_test_split 划分数据
# 3. 把 make_moons 生成的 Numpy 数组转换成更适合计算机计算的 Tensor
# 4. 用 TensorDataset, DataLoader 打包数据，准备发给模型


import torch
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def create_data():

# 1. 用 make_moons 生成数据
X, y = make_moons(
    n_samples=2000,
    noise=0.2,
    random_state=20
)

# 2. 用 train_test_split 划分数据
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=20
    stratify=y
)
# 先分出30%作为最终测试集

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, 
    y_temp,
    test_size=0.5,
    random_state=20,
    stratify=y_temp
)
# 分出剩下70%中的50%作为验证集


# 3. 把 make_moons 生成的 Numpy 数组转换成更适合计算机计算的 Tensor

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 4. 用 TensorDataset, DataLoader 打包数据，准备发给模型

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

return train_loader, val_loader, test_loader