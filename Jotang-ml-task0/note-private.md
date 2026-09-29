# Machine Learning 学习笔记

## #0 机器学习入门

#### 1. 搭建环境的流程

操作系统
  ↓
Python
  ↓
虚拟环境
  ↓
第三方库
  ↓
机器学习代码

#### 2. 必须知道的基本概念

概念类：
1. Package 软件包 = 写好的一组 Python 代码
`pip`（package installer for Python）
>类似于 C 语言中的头文件?

2. GPU = Graphics Processing Unit 图形处理器
一种高度并行的处理器架构，具有大量计算核心，特别适合执行可以拆分成大量相似子任务的计算。
3. CUDA = Compute Unified Device Architecture
它是 NVIDIA 提供的 GPU 计算平台和编程模型。

>*为什么一个图形处理器会适合数学计算？


*关系*
- GPU = 一台机器
- Driver = 操作系统和机器沟通的司机
- CUDA = NVIDIA 提供的一套让程序使用 GPU 进行计算的技术体系
PyTorch 想让 Python 程序指挥这块 NVIDIA GPU 工作，需要一套软件计算平台。

这个平台就是 CUDA 生态。
- PyTorch = 我们以后用来进行机器学习计算的软件框架
所以：
PyTorch 想让 Tensor 在 NVIDIA GPU 上计算，需要通过 NVIDIA 的 GPU 软件栈来完成。

工具类
1. conda 环境和软件包管理工具
Miniconda 是一个精简版的 Conda 安装方案
提供：
`conda`+ 基础 Python 环境 + 环境管理能力 + 软件包管理能力

参数类
1. -m 可以理解为：
module（模块）

它告诉 Python：
“不要把后面的东西当成一个普通文件来运行，而是把它当成一个 Python module 来运行。”


#### 3. 一些奇异的小问题记录

1. 使用`bash Miniconda3-latest-Linux-x86_64.sh` = `chmod +x` + `./Miniconda3-latest-Linux-x86_64.sh`


#### 4. 任务流程总结
【题目】
创建隔离 Python 环境
        ↓
【我们的开发意图】
给 Task 0 建立独立的 Python 3.9 运行环境
        ↓
【技术实现】
conda create -n ml_task0 python=3.9
        ↓
【Conda 的实际工作】
寻找并下载 Python 3.9 和依赖
        ↓
【当前提示】
确认是否接受软件包仓库 ToS
        ↓
【你的操作】
a + Enter
        ↓
【继续】
下载并创建 ml_task0

2. 
```bash
conda activate ml_task
# 走入虚拟 python 环境
```

3. 观察机器情况 & 配置 python 独立环境

`nvidia-smi`输出实际机器情况：
GPU Name:       NVIDIA GeForce RTX 5060
Driver Version: 592.01
CUDA Version:   13.1
Memory-Usage
0MiB / 8151MiB

- python -m pip
请当前这个 Python 自己调用属于自己的 pip
- python：当前 ml_task0 环境中的 Python
- -m：--module，让 Python 运行一个模块
- pip：Python 的 Package Installer，用来安装 Python 软件包
>依赖是什么？
Python 本身运行也需要一些其他组件，所以 Conda 会根据软件之间的依赖关系自动解决需要安装什么。
专业上：
**Dependency（依赖）**是一个软件正常运行所需要的其他软件、库或特定版本组件。

4. 安装PyTorch
PyTorch 是一个开源的机器学习（Machine Learning）/深度学习（Deep Learning）框架。
它提供了我们后面会学习的很多核心能力，例如：
- Tensor（张量）计算
- 自动求导（Automatic Differentiation）
- 神经网络构建
- GPU 加速
- 模型训练



有 NVIDIA GPU 时
Python 3.9
     ↓
PyTorch
     ↓
Tensor
     ↓
CUDA
     ↓
RTX 5060

无：
Python
 ↓
PyTorch
 ↓
CPU


install：
告诉 pip：“我要安装软件包。”

torch：
PyTorch 的核心 Python 软件包。

机器学习小白理解：
可以把 PyTorch 想成我们以后做机器学习实验时的“主要工具箱”。

专业定义：
PyTorch 是一个机器学习/深度学习框架，提供 Tensor（张量） 运算、自动微分（Automatic Differentiation）、神经网络等功能，并支持利用 GPU 加速计算。

torchvision：
PyTorch 的计算机视觉（Computer Vision）扩展库。


- 检查版本：
- python -c "import torch; print(torch.__version__)"
不要打开 Python 交互界面，也不用创建 .py 文件，直接让当前 Python 执行这一小段代码。

当前 Python
    ↓
能不能 import torch？
    ↓
PyTorch 是否真的能被 Python 使用？

- python -m pip show torch

python -c "import torch; print(torch.cuda.is_available())"
torch.cuda.is_available
启动当前环境里的 Python → 临时执行一小段 Python 代码 → 导入 PyTorch → 询问 PyTorch“CUDA 现在能不能用？” → 把答案打印出来。
>直接在 python 后面加 torch.cuda.is_available()，难道 Python 不能看到 () 就识别出来吗？
Python 解释器的命令行选项（command-line option）直接执行 <command> 中提供的 Python 代码
没有 -c 时，紧跟在 python 后面的东西通常会被当成脚本/输入入口，而不是自动当作一段 Python 源代码
-c <command>
python [-各种选项] [-c command | -m module-name | script | -] [args]

torch.cuda：
PyTorch 中负责 CUDA/GPU 相关功能的模块。

is_available()：
检查当前 PyTorch 是否能够使用 CUDA。() 表示“调用这个函数”。空的 () 表示调用它时不需要额外传入参数。

第一阶段：Python 命令行解析
──────────────────────────

python
 ↓
看到 -c
 ↓
知道：后面是 Python 源代码
 ↓
把 command 交给 Python 代码解析器


第二阶段：Python 代码解析
──────────────────────────

import torch
 ↓
加载模块

print(torch.cuda.is_available())
 ↓
识别函数调用
 ↓
执行 is_available()
 ↓
得到 True / False
 ↓
print 输出

- 当前这个 ml_task 环境里的 PyTorch，能不能实际访问 CUDA/GPU？



- 完成一次张量测算

第一次 Tensor 运算
Python
  ↓
PyTorch
  ↓
Tensor
  ↓
数学运算

Tensor ：
PyTorch 中用于表示多维数值数据的数据结构，可以记录数据的形状（shape）、数据类型（dtype）以及所在设备（device，如 CPU / CUDA GPU）。PyTorch 的模型输入、输出和模型参数都可以用 Tensor 表示。


```python
import torch
test-tensor1 = torch.tensor()
torch.tensor([[1.,2.],[3.,4.]])
```
// 根据这个 Python 二维列表创建一个 PyTorch Tensor。
表示这是由pytorch提供的创建tensor的函数
后面的 . 表示这些数字写成了 Python 的浮点数

python的赋值：
test-tensor1 = （给这个 Tensor 取名叫做 test-tensor1 ）
`@` 表示矩阵乘法

test-tensor1.device 
device 时tensor的一个属性，表示这个tensor当前存储在哪个计算设备上。
>如果不写`device='cuda'`，那么就是默认放到cpu上吗？哪怕cuda已经available


矩阵 × 矩阵

5. NumPy
Numerical Python
Python 里专门帮你处理大量数字、数组和矩阵计算的工具
Python 科学计算生态中的基础数值计算库，核心对象是多维数组（ndarray），提供数组运算、线性代数等功能。

>tensor 和 numpy 是否可混用于计算？

6. Matplotlib
这是 Python 生态中的数据可视化库，能够创建静态、动画和交互式图形。官方文档将其描述为用于创建这些可视化的综合性库。
在机器学习中有什么用？

- 把数据变成人可以直观看懂的图。
- 这是 Python 生态中的数据可视化库，能够创建静态、动画和交互式图形。官方文档将其描述为用于创建这些可视化的综合性库。

例如以后训练一个模型，你可能得到：

Epoch 1 → loss = 0.8
Epoch 2 → loss = 0.6
Epoch 3 → loss = 0.4
...

我们可以画：

Loss
 │\
 │ \
 │  \
 │   \
 │    \____
 └────────── Epoch

这样你就能观察：

模型训练过程中误差有没有下降。

这里的 loss（损失） 以后我们会专门解释，现在你只需要知道：

它是机器学习训练过程中用来衡量“模型当前表现有多差”的一个数值。

7. scikit-learn
传统机器学习工具箱
现成的机器学习算法和工具
分类 Classification: 给一个东西判断它属于哪个类别。提供逻辑回归、随机森林、最近邻等分类算法
回归 Regression
预测一个连续的数字。
聚类 Clustering
预处理 Preprocessing
机器学习模型通常不能直接把所有原始数据塞进去。

device="cuda"
torch.set_default_device("cuda")
torch.device("cuda")
```python
device = torch.device(
  "cuda" if torch.cudais_available() else "cpu"
)

x = torch.tensor([1., 2., 3.], device=device)
```
device-agnostic 设备无关代码的优势：具体使用什么设备，由程序前面决定
>为什么叫 cuda，不叫 gpu？是不是所有 GPU 都对应 CUDA？
PyTorch 的设备字符串规定使用：cuda
不加引号，python 会认为 cuda 是一个变量名
NVIDIA GPU
    ↓
CUDA

Apple GPU
    ↓
MPS

Intel GPU
    ↓
XPU

CPU
    ↓
CPU

>官方项目名称 & python package
scikit-learn & sklearn
pip install & import
从哪里安装软件包 & 加载什么软件包

`.` attribute access 属性访问
对象 . 名称
从sklearn这个对象中取出叫`__version__`的属性
`__version__`
sklearn.show_versions()


命令行世界：

python --version
       ↑
   命令行参数


Python代码世界：

sklearn.__version__
        ↑
     属性访问

>为什么`print`不需要占位符？
Python 的 print 本身就是一个函数，可以接收多个参数。
直接传多个参数：

print("score:", score)


字符串格式化：

print(f"score: {score}")

都可以。

### Python 
1. list
2. dict: key -> value
3. def

names = ["甲","乙","丙","丁"]
scores = {
        "甲"：99，
        
}



模型就是一个从输入数据中寻找规律，并利用这个规律进行预测的数学结构。

### 概念补充
1. checkpoint
训练过程中的存档，包括epoch、model_state_dict（也就是weight, bias）、optimizer_state_dict、val_loss

每一次运行train.py 是否也在运行 测试集？会不会发生数据泄露？

2. ReLU如何减少梯度消失问题

关于梯度消失：
梯度是多层导数相乘，包含激活导数的层数越多，梯度就几乎变成0，

关于调用GPU：
使用 Pytorch 框架创建一个张量，并指定设备为 CUDA。内存分配 + 主机到设备的数据传输（Host-to-Device transfer）
    ↓
PyTorch 的底层 C++ 实现通过 CUDA Runtime API 发起 kernel launch（内核启动）。
    ↓
CUDA kernel 通过驱动程序的 ioctl 接口，经 PCIe 总线传输到 GPU 的命令队列（command queue）。
    ↓
GPU 的 grid 被划分为多个 block，每个 block 分配给一个 SM，block 内的 thread 在 CUDA Core 上并行执行。
    ↓
Device-to-Host transfer（设备到主机的数据传输）。
    ↓
输出结果
