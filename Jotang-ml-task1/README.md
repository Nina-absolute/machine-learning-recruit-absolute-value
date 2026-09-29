## model.py


### 关于数据
make_moons
    ↓
生成 NumPy 数据
    ↓
train_test_split
    ↓
train / validation / test
    ↓
torch.tensor
    ↓
Tensor
    ↓
TensorDataset
    ↓
DataLoader
    ↓
batch

### 关于 MLP
make_net
    ↓
Sequential
    ↓
Linear
    ↓
ReLU
    ↓
Linear
    ↓
logits

### 初始化
init_weights
    ↓
Linear
    ↓
weight 初始化

## train.py
