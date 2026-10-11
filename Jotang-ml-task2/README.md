# Jotang-ml-task2

**本仓库用于收录 Jotang-ml-task 2 的所有解答、实验结果与拓展学习记录。**

## 写在前面

*关于这个仓库的内容组织：

本仓库主要分为三个部分：

- `#0 源代码 & 图表`：存放实践代码及其对应的实验结果。
- `#1 学习笔记 & 实验分析`：存放问题解答、详细学习笔记、代码原理说明与问题排查记录。
- `#2 附件`：存放实验所需的图片、数据集压缩包及其他辅助材料。

*关于文件用途的标明：

除文件名本身所表达的含义外，部分文件还会在开头注释或正文中标明具体用途，方便查找对应的实验内容。

**-----**
以下为 GPT 老师总结 + 我审核

## #0 源代码 & 输出图表

### 1. **实践 1：卷积核（`practice1`）**

本部分主要研究图像像素、卷积核、二维互相关、填充方式，以及不同卷积核对图像的处理效果。

- `practice1/print_pixel.py`：查看图片的像素信息，辅助理解图像数据的基本组成。
- `practice1/convolution_function.py`：实现卷积相关计算函数，供后续实验调用。
- `practice1/use_function.py`：调用编写的函数进行实验。
- `practice1/compare_4kernels.py`：对比四种卷积核的图像处理效果，包括均值模糊、高斯模糊、锐化和 Sobel 边缘检测。
- `practice1/valid_same.py`：对比不同填充方式对输出尺寸及图像结果的影响。

**实验结果图表：**

- `practice1/compare_result.png`：四种卷积核的处理结果对比图。
- `practice1/valid_same_result.png`：不同填充方式的实验结果图。

### 2. **实践 2：CNN 猫狗分类（`practice2`）**

**核心代码：**

- `practice2/code/samples_preview.py`：随机展示数据集中的图片，观察图片尺寸、清晰度和背景差异。
- `practice2/code/data.py`：统计猫狗图片数量，读取数据集，完成图像预处理、数据划分、Batch 组成及相关可视化。
- `practice2/code/model.py`：搭建并训练 CNN，记录各层 Tensor 形状，绘制训练曲线、特征图、混淆矩阵和错误样本，并保存训练得到的模型。
- `practice2/code/best_predict.py`：独立推理程序，加载已保存的模型，对本地图片进行分类，输出预测类别、预测概率及处理后的图片。
- `practice2/code/better_model.py`：改进模型的实验代码，用于尝试更丰富的数据增强或其他模型改进方法。具体实验内容以代码实际实现为准。

----

**实验结果与输出图表：**

- `practice2/outputs/sample_images.png`：随机抽取的数据集图片，用于观察原始图像的差异。
- `practice2/outputs/preprocessing_comparison.png`：图像预处理前后的对比结果。
- `practice2/outputs/augmented_samples.png`：数据增强后的样本展示。
- `practice2/outputs/better_model/`：改进模型的实验结果目录，保存改进实验的相关输出。
- `practice2/outputs/model/`：模型训练相关输出目录。
- `practice2/outputs/predict_process/`：独立推理程序生成的相关结果。


---

**模型相关输出主要指的是**：

**1. `practice2/outputs/model/`：基础模型实验结果**

- `best_model.pth`：最终选定的最佳模型权重。
- `cnn_aug_best.pth`：使用数据增强训练得到的最佳模型权重。
- `cnn_plain_best.pth`：未使用额外数据增强训练得到的最佳模型权重。
- `cnn_aug_curves.png`：数据增强模型的训练曲线，用于观察训练过程中的 Loss 和 Accuracy 变化。
- `cnn_plain_curves.png`：基础模型的训练曲线，用于观察训练过程中的 Loss 和 Accuracy 变化。
- `confusion_matrix.png`：混淆矩阵，用于分析模型对猫、狗两类样本的分类情况。
- `feature_maps.png`：CNN 卷积层输出的特征图，用于观察不同通道提取的图像特征。
- `misclassified_samples.png`：模型预测错误的样本，用于分析分类失败的原因。
- `model_architecture.png`：CNN 模型结构图，用于展示网络的主要组成及数据流向。

**2. `practice2/outputs/better_model/`：改进模型实验结果**

- `better_augmented_samples.png`：改进数据增强方案后的样本展示，用于观察不同增强变换对图像的影响。
- `cnn_aug_best.pth`：改进实验中保存的数据增强模型权重。
- `cnn_aug_curves.png`：改进实验中数据增强模型的训练曲线。
- `confusion_matrix.png`：改进模型的混淆矩阵，用于观察模型的分类表现。
- `experiment_report.md`：改进模型的实验报告，记录实验方案、结果及分析。
- `misclassified_samples.png`：改进模型预测错误的样本，用于分析剩余的分类问题。


**-----**

## #1 学习笔记 & 问题解答 & 实验分析

1. `practice1-answers.md`：**实践 1 的问题与解答**，用于整理卷积与图像处理相关题目的回答。

2. `private-extention/code_explanation.md`：**代码实现与原理说明**，辅助理解实验代码的具体写法、函数用途及相关原理。

3. `note-comprehensive.md`：**综合学习笔记 + 个人疑问与思考**，记录实践过程中涉及的知识点、原理解释以及额外的想法。（复习自用版，具体内容以文件正文为准。）

4. `problem-solving_log.md`：**问题排查与解决记录**，记录实践过程中遇到的问题、排查思路、解决方法及相关经验。

5. `practice2-answers.md`：**实践 2 的问题与解答**，用于整理猫狗分类相关题目的回答。

6. `private-extention/data_augmention.md`：猫狗二分类任务：数据增强的合理性与风险分析。

**-----**

## #2 📎 附件

1. `images`：项目图片目录，存放卷积与图像处理实验使用的原始图片等材料。

2. `screenshots`：**实验截图附件**，用于保存学习笔记或实验记录中引用的截图。具体截图内容以文件夹内实际文件为准。

3. `dogs-vs-cats.zip`：**猫狗分类数据集压缩包**，供实践 2 使用。使用前请确认压缩包中的实际目录结构，并据此设置数据读取路径。

4. `README.md`：**本仓库的文件用途说明**，用于快速了解项目结构、各文件的用途及实践内容。
