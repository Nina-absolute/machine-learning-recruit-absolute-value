import torch
from torch import nn

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. 数据
# ============================================================

def load_data(batch_size=64):

    X, y = make_moons(
        n_samples=2000,
        noise=0.2,
        random_state=20
    )

    # 80% train, 20% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=20
    )

    # 10% validation, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=20
    )

    # NumPy → PyTorch Tensor
    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )
    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )
    y_val = torch.tensor(
        y_val,
        dtype=torch.long
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )
    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    train_iter = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    val_iter = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size
    )

    test_iter = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size
    )

    return (
        X,
        y,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_iter,
        val_iter,
        test_iter
    )


# ============================================================
# 2. MLP
# ============================================================

def make_net(hidden_units=16):

    net = nn.Sequential(
        nn.Linear(2, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 2)
    )

    return net


def init_weights(m):

    if type(m) == nn.Linear:
        nn.init.normal_(
            m.weight,
            std=0.01
        )