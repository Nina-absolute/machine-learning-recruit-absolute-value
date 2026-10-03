import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import 导入模块/库

# #0 超参数

N_SAMPLES = 2000
NOISE = 0.20
RANDOM_STATE = 30
BATCH_SIZE = 32
HIDDEN_UNITS = 16
NUM_EPOCHS = 100
LR = 0.01
SEED = 30

# #1 数据处理

# 1-1 生成数据，打包数据
X, y = make_classification(
    n_samples=N_SAMPLES,        # 总样本数
    n_features=2,          # 两个特征
    n_informative=2,       # 两个特征都有信息量
    n_redundant=0,         # 无冗余特征
    n_clusters_per_class=1,# 每个类别一个簇
    weights=[0.99, 0.01],  # 类别 0 占 99%，类别 1 占 1%
    random_state=42        # 随机种子
)
# 1-2 划分数据
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
# 分出 70% 训练集和 30% 临时集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE,
    stratify=y_temp)
# 从临时集里分出 50% 验证集和 50% 测试集

# 1-3 数组变张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.long)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.long)
# 从 make_moons 生成的默认 Numpy 数组转化为更适合矩阵计算 Tensor

# 1-4 打包数据
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
# TensorDataset 把输入张量和标签张量按第一维配对，打包成一个数据集对象
# DataLoader: 把数据集取出，经过对于每个 epoch 打乱索引等处理，按照 batch size 小批量取出
# ps: DataLoader 不直接接受两个分开的张量，而是接受一个数据集对象.

# =====================================
# #2 定义模型
class MLP(nn.Module):
# 类似于声明这个新的自定义类要继承 nn.Module 的能力
    def __init__(self, hidden_units=HIDDEN_UNITS, num_classes=2):
    # 在 MLP 中定义一种新的初始化方法，也就是：1. 调用父类的初始化方法；2. 在创建 net 这个属性的基础上，创建层
        super().__init__()
        # 先调用父类的初始化方法（大概类似冷启动:/
        self.net = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes))

    def forward(self, x):
        return self.net(x)
    # 在 MLP 中定义 forward 方法：调用 self.net(x)，把 x 传给每个层，最后返回 self.net 得到的数值。

# =============================

# #3 训练

# 3-1 准备阶段：调好设备/调用函数/准备存储位置

torch.manual_seed(SEED)
# 设置 Pytorch 的随机种子：保证每次运行代码时，模型权重初始化、DataLoader 打乱顺序等随机过程的结果一样
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# 如果可以，把模型搬到 GPU 上跑。

output_dir = Path(__file__).parent.parent / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / 'best_model.pt'
# 设置保存最好模型/实验表格/各种图线的位置

model = MLP(HIDDEN_UNITS).to(device)
# 把模型参数等数据也挪到同一设备上去。
loss_fn = nn.CrossEntropyLoss()
# 设置使用的损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# 设置使用的优化器和学习率，调用 parameter() 告诉优化器要更新什么参数

# 3-2 训练循环（训练 + 验证）

history = {'train_loss': [], 'train_acc': [],
           'val_loss': [], 'val_acc': []}
# 创建一个叫 history 的新字典，用来保存每个 epoch 的训练结果
best_val_loss = float('inf')
# 创建 best_val_loss 变量，用历史最低损失值

for epoch in range(NUM_EPOCHS):
# 执行 NUM_EPOCHS 个 epoch
    # ---- 训练 ----
    model.train()
    # 调用 nn.Module 自带的 train 方法，切换训练模式
    train_loss, train_correct, train_total = 0.0, 0, 0
    # 初始化累计变量
    for X_batch, y_batch in train_loader:
    # 从训练集里取 batch 来训练
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # 把样本数据移到设备上去
        optimizer.zero_grad()
        # 清空上一轮累积的梯度
        logits = model(X_batch)
        # 经过前向传播，得到预测分数
        loss = loss_fn(logits, y_batch)
        # 得到损失函数算出的损失
        loss.backward()
        # 反向传播，从 loss 出发，沿着计算图反向计算每个参数的梯度，并将其存到每个参数的 .grad 属性里。
        optimizer.step()
        # 读取每个参数的 .grad 中的梯度，根据优化器的规则调整参数值。
        train_loss += loss.item() * X_batch.size(0)
        # 累加当前批次的总损失
        train_correct += (logits.argmax(1) == y_batch).sum().item()
        # 统计预测正确的样本数
        train_total += X_batch.size(0)
        # 利用 X_batch 的第零维长度（也就是一个 batch 的样本数），统计样本总数
    train_loss /= train_total
    train_acc = train_correct / train_total
# train_loss 除以总样本数，得到整个 epoch 的平均训练损失。
# train_correct 除以总样本数，得到整个 epoch 的训练准确率。
# ==============
    # ---- 验证 ----
    model.eval()
    # 切换评估模式
    val_loss, val_correct, val_total = 0.0, 0, 0
    # 初始化累计变量
    with torch.no_grad():
    # 关闭梯度计算
        for X_batch, y_batch in val_loader:
        # 从验证集里取 batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # 把数据搬到模型所在的设备上去算
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            val_correct += (logits.argmax(1) == y_batch).sum().item()
            val_total += X_batch.size(0)
            # 得到预测分数、损失、准确度等等，同训练集
            # 区别在于此处无反向传播、优化器更新参数，验证集的数据不参与模型学习。
    val_loss /= val_total
    val_acc = val_correct / val_total

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
# 把 loss/accuracy 写在字典里

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'epoch {epoch+1:03d}, '
              f'train_loss {train_loss:.4f} train_acc {train_acc:.4f}, '
              f'val_loss {val_loss:.4f} val_acc {val_acc:.4f}')


# #4 测试
model.load_state_dict(torch.load(save_path))
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
test_preds, test_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        test_correct += (logits.argmax(1) == y_batch).sum().item()
        test_total += X_batch.size(0)
        test_preds.append(logits.argmax(1).cpu())
        test_labels.append(y_batch.cpu())
test_loss /= test_total
test_acc = test_correct / test_total
test_preds = torch.cat(test_preds).numpy()
test_labels = torch.cat(test_labels).numpy()
print(f'\ntest loss {test_loss:.4f}, test acc {test_acc:.4f}')

# #5 绘制loss / accuracy 曲线
epochs = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_loss'], label='train')
plt.plot(epochs, history['val_loss'], label='val')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_acc'], label='train')
plt.plot(epochs, history['val_acc'], label='val')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()

plt.tight_layout()
plt.savefig(output_dir / 'curves.png', dpi=200)
plt.close()

# #6 绘制决策边界
X_np = X_test.numpy()
y_np = y_test.numpy()

x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                    dtype=torch.float32).to(device)
with torch.no_grad():
    pred = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, pred, alpha=0.3, levels=[-0.5, 0.5, 1.5])
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, edgecolors='k', s=20)
plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Decision Boundary')
plt.savefig(output_dir / 'decision_boundary.png', dpi=200)
plt.close()

# #7 输出混淆矩阵
cm = confusion_matrix(test_labels, test_preds)
ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(values_format='d')
plt.title('Confusion Matrix')
plt.savefig(output_dir / 'confusion_matrix.png', dpi=200)
plt.close()

# #8 打印错误样本
wrong = np.where(test_labels != test_preds)[0][:5]

plt.figure(figsize=(6, 5))
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, alpha=0.25)
if len(wrong) > 0:
    plt.scatter(X_np[wrong, 0], X_np[wrong, 1],
                facecolors='none', edgecolors='k', s=120)
plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Misclassified Samples')
plt.savefig(output_dir / 'misclassified.png', dpi=200)
plt.close()

print('\nMisclassified samples:')
for i in wrong:
    print(f'  x=({X_np[i,0]:.3f}, {X_np[i,1]:.3f}), '
          f'true={test_labels[i]}, pred={test_preds[i]}')
