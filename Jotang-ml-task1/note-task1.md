# Task 1 学习笔记

## STEP 1: 生成数据，可视化，划分数据集
>用 make_moons 生成二分类数据，固定随机种子；可视化，并划分训练/验证/测试集，同时理解三者职责和数据泄漏。

### 1. 关于题目中的概念
    分析：
    1. `make_moons`是`scikit-learn`提供的一个数据集生成器，可以直接生成两个互相交错的半圆形数据，也就是所谓的 two interleaving half circles（二个交错的半圆）。它专门适合用于演示分类和聚类。

    2. 随机种子
    计算机通常使用伪随机数生成器（Pseudo-Random Number Generator）：它实际上是按照一个确定的算法产生“看起来随机”的数字。
    相同的起始值（也就是随机种子）可以产生相同的随机序列，可用于后续复现验证。

    3. 划分训练/验证/测试集，理解三者职责和数据泄漏

### 2. `make_moons`的进一步学习
使用`make_moons`时，命令形如：
```python
from sklearn.datasets import make_moons
x, y = make_moons(
    n_sample = 10,
    random_state = 42
)
```
接下来打印到屏幕，仔细分析一下：
```python
print(x)
print(y)
```
![](./../screenshots/make_moons-example1.png)
可以观察到`x`有十行两列，而`y`有一行 10 个
故而猜想，在 make_moons 生成的数据中：
    对于`x`
    行数代表n_sample，也就是样本数，列数则代表特征数。
    对于`y`
    只有两种类别，也就是两种标签（0 或者 1）
- 经查猜想正确，`make_moons`生成的数据，特征数默认是 2，标签也默认只有两种，就是 0 或者 1，达成二分类目的。

补充：关于`random_state = 42`
随机数据虽然还是随机生成的，但把随机过程固定下来。

关于`noise`
添加到数据中的高斯噪声的标准差（standard deviation）
有点像线性回归里的 r 值被人工影响了。

### 3. 可视化数据（用 matplotlib 画散点图）
![](./../screenshots/PASS-data-visualization.png)

### 4. 划分数据集

**#0 分析概念** 

    1. 训练集 Training set 
    用于估计模型参数、让模型从数据中学习规律的数据子集。

    2. 验证集 Valiadation set
    用于在训练过程中调整超参数、选择模型或进行早停。它相当于"模拟考"。

    3. 测试集 Test set 
    在模型和超参数都确定后，用于最终、一次性的性能评估。它是"最终大考"。

#1 划分方法
>参考：[菜鸟教程](https://www.runoob.com/ml/ml-training-and-test-set-splitting.html)

#2 注意事项
stratify=y，让划分后类别比例尽量保持一致
shuffle=True：训练时每个 epoch 打乱顺序。
num_workers=0：在主进程里加载数据

在机器学习中，如果我们在全部数据上训练模型，然后又用这同一份数据去评估它的性能，就会犯同样的错误。模型会表现得异常出色，因为它已经"见过"并"记住"了所有数据的细节，包括其中的噪声和偶然性。这种现象被称为*过拟合*。

本实验采用随机划分训练集、验证集和测试集，不额外使用 stratify 保持类别比例，使类别比例也自然受到随机抽样影响。这样可以观察随机数据划分对模型实验结果产生的影响。


### 5. 数据泄露 Data Leakage



### 6. 常见概念理解

1. `noise`
把 `noise` 设成 0.5 以上会导致数据完全混杂，不可分；

2. `stratify`
按 `y` 分组，每组内部再按比例切。

torch.tensor(np_array) 后修改原数组，Tensor 也变？→ 不会，torch.tensor 会复制数据；想共享内存用 torch.from_numpy。

num_workers
drop_last
pin_memory


## STEP 2: 
>用 PyTorch 构建至少有一个隐藏层的 MLP，完成训练、验证、测试以及模型保存/加载。

### 1. 关于题目中的概念：

#### 训练
1. MLP = Multi-Layer Perceptron 多层感知机


它是一种前馈神经网络（Feedforward Neural Network），通常由若干个全连接层（Fully Connected / Linear Layer）和非线性激活函数组成。
- 结构：

2. 全连接线性层（Fully Connected / Linear Layer）

3. 数据集的 split 和处理


4. Batch size
每次参数更新前通过网络传播的数据样本数量。

>为什么堆叠成 Batch 可以提高运算速度？
5. torch.nn
它是 PyTorch 中专门提供神经网络相关组件的模块。

6. shuffle=True
7. epoch

8. model.train()
9. TensorDataset, DataLoader
- TensorDataset
把 X 和 y 一一配对打包在一起
- DataLoader
按照 batch size 把 Dataset 分批送给模型


10. random_state vs torch.manual_seed()
random_state 只能控制 make_moons 和 train_test_split 随机过程？具体控制了什么？

11. `shuffle`
DataLoader 的布尔参数（只能输出`True`和`False`的参数）：
- `shuffle=True`：每个 epoch 开始前，重新打乱样本顺序。
- `shuffle=False`（默认）：保持原始顺序。

*补充：常见布尔参数名*：

`is_train` 是否是训练集
`shuffle` 是否打乱
`drop_last` 是否丢弃最后一批不足 batch_size 的数据
`download` 是否下载
`verbose` 是否打印详细日志
`bias` 是否加偏置


torch.manual_seed() 控制 PyTorch 的随机数生成过程？
torch.cuda.manual_seed_all() 
给随机数发生器一个固定的起点？
都是固定起点了还怎么随机？出现的数字都是同一批吗？顺序和具体数字都一样吗？

11. train_loop()

`requires_grad`

#### 验证
12. pred = model(X) 模型预测出来的两个类别的 logits。
13. loss = loss_fn(pred, y)

#### 测试

#### 保存
>为什么要保存模型？

>为什么机器会过拟合？为什么反复学习反而学不到通用规律？

*根本区别：*
- 人类学习：我们追求的是“理解背后的原理”，即使数据有限，我们也能用常识、逻辑、因果推理去补全。
- 机器学习：它只是在调整数学参数，让模型在这批数据上的输出尽量接近标签。它不理解任何东西，它只是在拟合。

>举例（ deepseek 解释）：假设你要训练一个模型，判断一张图片是猫还是狗。训练集只有 10 张图片：5 张猫的图片：都是橘猫，背景都是沙发。5 张狗的图片：都是黑狗，背景都是草地。
>模型训练时，它发现了一个“规律”：“橘色 + 沙发 = 猫；黑色 + 草地 = 狗。”
>这个规律在训练集上完全正确，准确率 100%。但它学到的是真正的规律吗？不是。真正的规律是“猫的耳朵尖、胡须长、瞳孔竖”，但模型没学到这些，它学到了“沙发”和“草地”这两个偶然的特征。
>现在你拿一张草地上橘猫的照片去测试，模型会懵——它根据“草地”判断是狗，但根据“橘色”判断是猫，内部参数打架，最终输出可能完全错误。
>**这就是过拟合。模型没有学到“猫之所以是猫”的本质，而是记住了训练集里的噪声和偶然关联。**

我的总结：
也就是说，如果过度反复训练，机器就会记住不属于标签的本质的关联，记住训练集中那些偶然的关联，然后把这种关系赋予更多的权重以期待降低损失。
但迁移到验证集和测试集时，这种偶然关联大概率不成立。所以之前记忆的这些偶然因素（其实就是噪声）就学错了，这种错误关联就会让它在测试中的表现不如训练。
本质上其实是因为机器没有办法像人类一样初步地筛选学习内容，无法区分噪声和准确数据，只是在不加分辨地分析数据全部特征，然后进行数学上的优化。

#### *一些些奇妙的 idea*: 
关于初始的权重设置：
是否可以通过一开始提供对显著相关的权重分配，从而实现减小噪声影响、放大关键数据影响？比如猫狗识别中，在初始设置时就刻意对机器强调识别哪一类特征的关联。
同时，最好这种权重分配可以由机器自主完成，类似语言模型分析词后最大概率词，从而实现更加高效的数据分析
（btw，这应当也是一种机器学习，嗯也就是说通过预备的机器学习，来实现更好地机器学习。）




>保存模型有什么实际意义？

1. `best_val_loss = float('inf')`
best_val_loss， 用来记录目前最好的损失。
float('inf') = infinity

>为什么要在一开始规定一个变量然后让它取无限大，不能直接定义一个函数来保存所有 Loss 值，然后取最小？
ANSWERS: 
1. 训练时每轮都保存一个 checkpoint 浪费磁盘；
2. 多一次遍历：虽然 100 个数字遍历很快，但逻辑上多此一举。


2. `save_path = 'best_model.pt'`
`save_path`: 变量名，保存路径。
`'best_model.pt'`: 字符串，文件名。
`.pt`: PyTorch 模型文件的约定扩展名（PyTorch Tensor 的缩写，也可以是`.pth`）
`torch.save(obj, save_path)` 会在这个路径创建文件
目录必须存在，否则会报错（我代码里前面 os.makedirs('outputs', exist_ok=True) 就是提前创建目录）。
就是给模型存档起个文件名。

3. 
history = {'train_loss': [], 'train_acc': [],
           'val_loss': [], 'val_acc': []}
history：变量名。

{...}：Python 字典字面量。

'train_loss': []：键值对，键是字符串 'train_loss'，值是空列表 []。

[]：空列表，用来后续 append()。




14. torch.save()
把存在于内存里的参数保存到硬盘里，避免程序结束以后内存释放训好的模型的权重就没了。

#### 加载


3. 绘制 loss、accuracy 曲线和二维决策边界。
4. 至少改变两个因素进行对照实验，而且一次只改变一个主要变量，观察训练速度、显存等指标以及最终结果变化。
5. 输出混淆矩阵，并分析几个错误分类样本。

### 2. 关于拓展问题

1. 你在这个任务之中，选择了什么激活函数，为什么选这个？
你选择了什么损失函数？为什么？那如果这一题是三分类问题呢？
optimizer.zero_grad()、loss.backward() 和 optimizer.step() 分别完成了什么工作？如果不清空梯度会发生什么？

optimizer.zero_grad()
清除上一轮（上一轮指的是什么？）积累的梯度，为什么梯度会积累？如果积累不清楚会发生什么？发生过拟合？

loss.backward()
反向传播，分析模型参数变化时 Loss 如何变化

optimizer.step()
优化模型参数


更深、更宽的神经网络一定能取得更好的效果吗？模型过于简单或过于复杂分别可能带来什么问题？
学习率过大或过小会怎样影响模型训练？你能从 loss 曲线中观察到哪些现象？
假设还有一个分类问题，但是两个类别的样本数量非常不均衡，训练出来的模型会怎么样？为什么？说明了什么？ 你可以尝试构造一个数据集试试看
如果想要过拟合这个数据集，你会怎么做？为什么？试试看能不能实现吧！（在别的数据集也可以）训练过程和数据呈现和正常训练有何不同？你通过什么判断他过拟合？



## 拓展了解：
1. 关于 `Xavier`

`init_weights` & `net.apply`

```python
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
net.apply(init_weights)
```
- `def init_weights(m)`
- init_weights = initialize weights：
- m = module，表示接受一个参数（PyTorch 模块）

