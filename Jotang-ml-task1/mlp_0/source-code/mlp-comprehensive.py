import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import pandas as pd
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
X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=RANDOM_STATE)

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
# dtype 表示 data type: float32 = 32 位浮点数；long = 64 位有符号整数

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
# 创建 best_val_loss 变量，保留历史最低损失值
records = [] 
# 创建 records 列表，用于记录训练过程中每个 epoch 的速度、显存等指标

for epoch in range(NUM_EPOCHS):
# 执行 NUM_EPOCHS 个 epoch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()   
        # 重置显存峰值记录，在这个 epoch 的范围内记录显存峰值
        # 为了防止之前的训练的 epoch 累计影响我们统计测试阶段的显存占用
        torch.cuda.synchronize()               
        # 确保之前的 GPU 任务全部执行完毕，强制同步测试起点
    t0 = time.perf_counter()                   
    # 记录 epoch 开始的精确时间
    # perf_counter 是微秒级精度，且不受系统时间调整影响

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
        # 得到损失函数算出的这一个 batch 的损失
        loss.backward()
        # 反向传播，从 loss 出发，沿着计算图反向计算每个参数的梯度，并将其存到每个参数的 .grad 属性里。
        optimizer.step()
        # 读取每个参数的 .grad 中的梯度，根据优化器的规则调整参数值。
        train_loss += loss.item() * X_batch.size(0)
        # 累加各个批次的损失，累成整个 epoch 的损失
        train_correct += (logits.argmax(1) == y_batch).sum().item()
        # 统计预测正确的样本数
        train_total += X_batch.size(0)
        # 利用 X_batch 的第零维长度（也就是一个 batch 的样本数），统计 epoch 的样本总数
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
# 把 loss/accuracy 写在 history 字典里，便于之后画图线

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 同训练处，也要求强制同步结束。
    epoch_time = time.perf_counter() - t0
    # 记录一个 epoch 训练 + 验证结束的时间，算出总的训练时间。
    samples_per_sec = len(X_train) / epoch_time
    # 算出每秒处理的样本数，此处的 len(X_train) 写成 train_total 与前文对应或许更好。
    if torch.cuda.is_available():
        # 以下数据的单位均由字节换算为 MB
        gpu_alloc_MB = torch.cuda.memory_allocated() / 1024**2
        # epoch 结束的这一刻，读取真正分配（allocated）给张量的显存
        gpu_reserved_MB = torch.cuda.memory_reserved() / 1024**2
        # PyTorch 向 GPU 申请并缓存（reserved）起来的显存
        gpu_max_alloc_MB = torch.cuda.max_memory_allocated() / 1024**2
        # 取整个 epoch 的显存分配峰值
        gpu_max_reserved_MB = torch.cuda.max_memory_reserved() / 1024**2
        # 取整个 epoch 的已保留显存的峰值
    else:
        gpu_alloc_MB = gpu_reserved_MB = 0.0
        gpu_max_alloc_MB = gpu_max_reserved_MB = 0.0
        # 没有 GPU，显存自然为 0.

    records.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'epoch_time_s': epoch_time,
        'samples_per_sec': samples_per_sec,
        'gpu_alloc_MB': gpu_alloc_MB,
        'gpu_reserved_MB': gpu_reserved_MB,
        'gpu_max_alloc_MB': gpu_max_alloc_MB,
        'gpu_max_reserved_MB': gpu_max_reserved_MB,
    })
    # 训练结果/速度/显存等指标打包为训练日志，每一轮验证记一次，便于后续画表格
    # 列表装字典的好处：一个字典 = 一整个 epoch 的所有数据 = 后续表格的一整行
    # 一次性对每一个 epoch 的数据 append 进去，不需要一个键一个 append。

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    # 保存验证阶段损失最小的最好模型到 outputs 为 best_model.pt


    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'epoch {epoch+1:03d}, '
              f'train_loss {train_loss:.4f} train_acc {train_acc:.4f}, '
              f'val_loss {val_loss:.4f} val_acc {val_acc:.4f}')
    # 每十个 epoch 输出一次模型指标

df = pd.DataFrame(records)
# 把 records 列表转成 pandas 表格 df
df['experiment'] = 'origin experiment'
# 标注实验名称
df.to_csv(output_dir / 'training_metrics.csv', index=False, encoding='utf-8-sig')
df.to_excel(output_dir / 'training_metrics.xlsx', index=False)
# 保存 csv/excel 格式表格
print(df.tail())
# 打印表格最后 5 行，方便快速查看训练结果。


# #4 测试
model.load_state_dict(torch.load(save_path))
# 加载最好模型来测试
model.eval()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()   
    # 重置显存峰值，隔离出纯推理阶段的显存
    torch.cuda.synchronize()               
    # 清空训练遗留的 GPU 任务队列
test_start_time = time.perf_counter()      
# 记录测试开始时间

test_loss, test_correct, test_total = 0.0, 0, 0
test_preds, test_labels = [], []
# 创建用于存预测值和正确标签的字典
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
        # 把张量从 GPU 搬到 CPU，便于后续把张量转成 NumPy 数组画混淆矩阵

if torch.cuda.is_available():
    torch.cuda.synchronize()               
    # 强制同步结束
test_end_time = time.perf_counter()        
# 记录测试结束的时间

test_loss /= test_total
test_acc = test_correct / test_total
test_preds = torch.cat(test_preds).numpy()
test_labels = torch.cat(test_labels).numpy()
# torch.cat 把 test_preds/labels 列表（没有.numpy 方法）里的多个小张量各自拼成一个完整张量。
# 再把拼接后的张量转成 NumPy 数组，方便后面 confusion_matrix 使用。

test_time = test_end_time - test_start_time
test_samples_per_sec = test_total / test_time
test_gpu_max_alloc_MB = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
test_gpu_max_reserved_MB = torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0
# 读取测试时间、每秒处理样本数和显存占用

print(f'\ntest loss {test_loss:.4f}, test acc {test_acc:.4f}')
print(f'test time {test_time:.4f}s, samples/sec {test_samples_per_sec:.2f}')
print(f'test gpu max alloc {test_gpu_max_alloc_MB:.2f} MB, '
      f'test gpu max reserved {test_gpu_max_reserved_MB:.2f} MB')

test_summary = pd.DataFrame([{
    'experiment': 'exp1_lr_0.1',
    'test_loss': test_loss,
    'test_acc': test_acc,
    'test_time_s': test_time,
    'test_samples_per_sec': test_samples_per_sec,
    'test_gpu_max_alloc_MB': test_gpu_max_alloc_MB,
    'test_gpu_max_reserved_MB': test_gpu_max_reserved_MB,
}])
test_summary.to_csv(output_dir / 'test_summary.csv', index=False, encoding='utf-8-sig')
test_summary.to_excel(output_dir / 'test_summary.xlsx', index=False)
# 保存测试汇总指标表格

# #5 绘制loss / accuracy 曲线
epochs = range(1, NUM_EPOCHS + 1)
# 注意此处 range 左闭右开，需要加 1
plt.figure(figsize=(10, 4))
# figure 创建图片，figsize 规定尺寸（宽，高）(cm)

plt.subplot(1, 2, 1)
# 选择左边的图（一行两列中的第一个）
plt.plot(epochs, history['train_loss'], label='train')
# plot 画折线图（横坐标，纵坐标）
plt.plot(epochs, history['val_loss'], label='val')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
# 设置 x, y 轴
# legend 显示图例

plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_acc'], label='train')
plt.plot(epochs, history['val_acc'], label='val')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()

plt.tight_layout()
# 自动调整子图之间的间距，防止标签、标题、图例互相重叠。
plt.savefig(output_dir / 'curves.png', dpi=200)
# 保存图片（dots per inch = 200）
plt.close()
# 关闭当前图形，释放内存。

# #6 绘制决策边界
X_np = X_test.numpy()
y_np = y_test.numpy()
# 张量转数组

x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
# 找出测试集 x1 坐标的范围，并向左右各扩展 0.5，让决策边界图边缘不贴太紧。
y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
# 把一维的 x 坐标和 y 坐标交叉组合，生成覆盖整个平面的网格点。

grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                    dtype=torch.float32).to(device)
# 把所有网格点整理成模型能接受的输入张量，形状 (N, 2)。
with torch.no_grad():
    pred = model(grid).argmax(1).cpu().numpy().reshape(xx.shape)
# 得到每个网格点的预测类别，形状和 xx 一致，方便后面画填充图。

plt.figure(figsize=(6, 5))
# 创建一个新的画布，宽 6 英寸、高 5 英寸。
plt.contourf(xx, yy, pred, alpha=0.3, levels=[-0.5, 0.5, 1.5])
# 根据每个网格点的预测类别，用半透明颜色填充出决策区域。
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, edgecolors='k', s=20)
# 把测试集样本按真实标签画成带黑边的散点。
plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Decision Boundary')
# 设置横纵轴、标题
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
