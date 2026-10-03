**本文件夹用于收录 Jotang-ml-task 1 的所有解答**

### 写在前面
*关于这个文件夹里的文件命名规则：
>虽然本文件夹阅读文件名主要还是**依赖关键词**顺着中文意思排列，并不是完全按照该规则来，但这里写一下还是之后要使用的命名习惯。

- 可以忽略的符号：
    使用`_`一般是指代空格，比如说：`best_model.pt`；`-`则一般用于后置描述属性，比如说`note-comprehensive`；
- `数字`通常用于表示同类型文件的不同版本，比如说不同 MLP 对照实验有不同编号，`mlp_1``2``3`等表示对照实验 1 到 3。

*关于文件用途的标明：

除了**文件名的直接意义表示**外，*文件开头注释*也会有适当标注。
>例如：`TO DO LIST`或者`对照实验 1：LR = 0.01 变为 LR = 0.1`也会标注出这个文件的具体用途。
-----
## 本文件夹构成如下：
### #0 源代码 & 图表

1. **原始实验`mlp_0`**：
- `source-code`：含模型的源代码。
    - `mlp-comprehensive.py`：**内含超多代码注释辅助`note-comprehensive`食用**
    - `mlp-final.py`：只是用来对着代码复习用途/跑模型的**无注释纯代码**。
    - `data_visualization`：用于原始数据可视化的源代码。

- `outputs`：
    - `origin_data.png`：原始数据可视化结果
    - `best_model.pt`:最好模型
    - `confusion_matrix.png`：混淆矩阵
    - `curves.png`：loss/accuracy 曲线
    - `decision_boundary.png`：决策边界
    - `misclassified.png`：判断错误的样本分析
    - `test_summary.csv/.xlsx`：关于测试集的实验表格
    - `trainning_metrics.csv/.xlsx`：关于训练集的实验表格


2. **控制单一变量的对照实验**：
>*PS：下文的“所有源代码/图表”即指的是：原始实验`source-code` + `outputs`的对应内容（除了原始数据可视化部分的代码和输出不再重复）。

- `mlp_1-change-lr`：改变学习率的实验，内含所有源代码/图表；
- `mlp_2-change-hidden_units`:改变隐藏层神经元数的实验，内含所有源代码/图表；
- `mlp_3-change-hidden_layers`：改变隐藏层层数的实验，内含所有源代码/图表；

3. `mlp_imbalanced`:**研究不均衡数据集的实验**，内含所有源代码/图表（不含有显存/速度分析的实验表格）；

4. `mlp-overfit`：**试图过拟合模型的实验**，含源代码和 loss/accuracy 曲线。

### #1 学习笔记 & 错误样本分析 & 

1. `answers.md`：必须知道的**回答趁热打铁**部分的笔记；

2. `mlp_0`中的`analysis-misclassified`：**对几个模型判断错误的样本分析 + 对于分析判断错误样本的一丢丢感悟**；

3. `note-comprehensive.md`: **自用详细版的学习笔记 + 一些些疑问 & idea**，略有些长，虽然除 ai 整理表格外纯手搓，但完全可略过（含泪。~~（会有人懂我的细节吗QAQ（~~

4. `analysis-controlled_experiment.md`：分析对照实验的速度、显存、结果变化，具体图表（位于各自对照实验的`outputs`文件夹中）嵌入笔记辅助分析。

### #2 附件

1. `sreenshots`：笔记中使用的所有截图，包含：
- `lr`开头的一系列`.png`：分析学习率过大过小影响的对照实验截图；
- `mlp-final.png`：原始 MLP 运行结果；
- `orgin-DataVisualization`：原始数据可视化截图。

2. **`README.md`**: 必读的文件用途说明。