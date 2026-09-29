# Notes for Mutilayer Perceptron

## 基本概念

1.  MLP
它是一种前馈神经网络（Feedforward Neural Network），通常由若干个全连接层（Fully Connected / Linear Layer）和非线性激活函数组成。

2. API

3. class
用于创造实例对象，写明对象会有怎样的状态、属性和功能。
（实例对象大概是一种既有属性又有函数功能的东西:/

## 函数和类

### 关于数据

生成数据
1. make_moons
用于生成数据，生成一种只含两种特征的样本，画在图像中呈两个半月形（当然是没有 Noise 的理想情况）

划分数据集

打包数据

## 关于训练

>为什么训练集是`shuffle=True`，验证集和测试集是`shuffle=False`？
一般情况下，训练集为了保证 DataLoader 每轮读取训练样本时的随机性当然是打乱更好。
而验证集不参与参数更新，所以没有必要打乱，而且每次都按照固定顺序检查，更能有效比较出来训练效果。
测试集：如果每次测试都重新随机排列，虽然总体的 accuracy 一般不会因此改变，但样本顺序变了不利于调试和定位具体错误样本。


## 关于模型
1. `nn.Sequential`
按照写的顺序，一层层执行。

>为什么要专门写 Sequential？Linear ReLU不都是函数吗？直接让 python 按照顺序执行函数操作不就行了？
1. 此处`nn.Linear``nn.ReLU`不是普通函数，而实际上是在创建一个`Linear``ReLU`模块对象，本质上属于`nn.Module`的类。`nn.Linear(2, 16)`则是这个类创建出来的一个对象
写成`net(X)`是什么意思？此处的`X`是用来调整超参数的吗？如果只写一个数字如何辨认出来要调整的是什么？


>为什么不写
```python
class MLP(nn.Module):
创建一个属于 PyTorch 神经网络体系的新模块
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 2)

    def forward(self, X):

        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)

        return X
```
这里的`nn.Module`是什么意思？
PyTorch 神经网络模块的基础类。自动拥有 PyTorch 模型的完整能力

2. Linear
全连接层
Linear(x, hidden_units)
输入特征数，输出神经元数。
>隐藏层的神经元数（隐藏岑宽度）到底有什么影响？
理论上，更多的神经元数可以表示更加丰富的中间特征，提高模型的表达能力。
但不是隐藏层越宽越好。增加隐藏层会增加参数数量、计算量和显存占用。
>为什么参数越多，更有可能过拟合？
因为参数越多，模型的描述能力越强，也就是说，可以把噪声、偶然异常等等都解释出来，反而有可能把这些东西也当作正经规律，从而产生过拟合。

3. ReLU
Rectified Linear Unit，修正线性单元
ReLU(x) = max(0, x)：输入的正数保持不变，负数变成 0.
>为什么需要激活函数？
如果没有可以非线性处理的激活函数，那么从线性的全连接层组合到另一个线性的全连接层，仍然只是一个线性变换。

4. logits
由前面模型分析得到的，尚未经过 Softmax 转换的原始分数。大概是一种类似于呈现支持力度的分数，
>模型是如何得出 logits 的？
后续需要使用`CrossEntropyLoss`这个损失函数，所以需要没有 Softmax 处理的 logits，直接输入CrossEntropyLoss.

>Softmax 是什么？归一化有什么用？
Softmax 可以把多个原始分数转换成概率分布形式。归一化相当于把不同尺度的数值转换到一个统一的、可比较的范围（大概是让分布差异更加直观？


5. init_weights
```python
def init_weights(m):
if type(m) == nn.Linear:
nn.init.normal_(m.weight, std=0.01)
```
在训练开始前先给模型参数一个初始值。
- `(m)`表示`apply()`当前传给`init_weights`的一种模块。
- `if type(m) == nn.Linear:`只有当前模块是`Linear`时，才初始化它的权重。
- `nn.init.normal_()`
normal 是指正态分布，也就是说从一个正态分布中随机生成初始权重。std 就是 standard deviation，正态分布中的标准差。
>为什么这里有下划线？
`_`表示 in-place operation 原地操作，也就是直接修改原来的 Tensor，而不是创建一个新的 Tensor 再替换（`_`类似于 bash 中的`source`？
>`nn.init.normal`不是在调整权重吗？和替换/创建 Tensor 有什么关系？
>`m.weigh`是什么？为什么没有`m.bias`？
`m.weight`本质上是一个parameter，
对于`bias`，不是没有`m.bias`，只是`nn.Linear`本身已经有默认参数初始化，所以不需要写在程序里。


>为什么要设置`if type(m) == nn.Linear:`？为什么需要初始化？初始化为什么不是都设置为 0 吗？
1. 因为模型训练开始之前：weight、bias必须有数值。否则 Linear 连计算都无法进行。
2. 如果所有权重都初始化为0，那么这一层所有神经元的输出都一样，反向传播时梯度也完全相同。那么神经元所学到的东西就无法真正分化，所以必须用随机数（如 Xavier 或 Kaiming 初始化）来打破对称性。

6. `net.apply(init_weights)`
把 `init_weights` 应用到 `net` 及其子模块。
>这里的`net`就是前面自定义的`nn.Sequential()`是吗？可是`net`下面的定义并不含`apply`，`apply`是自定义的还是本来就有的？
apply 不是自定义的，是 PyTorch`nn.Module` 自带的方法。


>写`net[0].weight`又是什么意思？
意思是取 net 中第零个子模块研究它的 weight.

>如果写`net = make_net(16)`，Python 解释器怎么确定这里的16指的是`hidden_units`？就因为之前写的是`net = make_net(hidden_units)`，所以后文写16/32等等就可以识别出来是说hidden_units？
不是，是因为之前写过了`def make_net(hidden_units=16)`，所以Python记住了make_net的第一个参数就是hidden_units，


>为什么 make_net() 最后要 return net？为什么要return net才可以让train.py得到模型？难道这个make_net()的自定义函数不是直接写在程序里的吗？为什么不可以让train.py在调用model时直接调用？那么什么情况下不用写return？这个return和经典的hello.c中的 return 0 有什么区别？
`def make_net`只是在定义一个函数，并不是定义完了以后就能让 Python 自动把它交到其他程序里，真正执行需要用`make_net(hidden_units)`
当我们在写：
```python
def make_net(hidden_units=16):

    net = nn.Sequential(
        nn.Linear(2, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 2)
    )

    return net
```
函数调用时发生了什么？
事实上是先调用了make_net然后进入函数内部，创建 Sequential，再把它放入局部变量 net，return net 函数结束
没有return net就是创建好了一个模型，但是并没有把模型交付给最开始调用这个函数的命令。
可以直接调用写`net = model.make_net`
如果调用函数者不需要返回值，比如说只是执行一个动作，那么也可以不加return。
C 中的 return 0 实际上是返回状态数 0，证明程序成功。而 Python 中想要设置程序进程，通常需要exit()等等。


4. CrossEntropyLoss

class MLP(nn.Module): vs net = nn.Sequential()

model.py