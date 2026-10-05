# TO DO LIST：插入图片，分析模型训练各个指标（速度，显存等），还有结果变化。同时，针对对照实验操作进行反思。

## 写在前面
*输出实验表格参考的方法：
[mmdetection测试模型显存占用与推理速度](https://blog.csdn.net/yshMars/article/details/120652476)

*实际实验表格保存的指标有：

1. 训练过程表：`training_metrics.csv / .xlsx`

| 表头 | 实际意义 | 计算方式 / 说明 |
|---|---|---|
| `epoch` | 第几个训练轮次 | 从 1 开始，到 `NUM_EPOCHS` 结束 |
| `train_loss` | 当前 epoch 在**训练集**上的平均损失 | 每个 batch 的损失乘以 batch 样本数，累加后除以训练集总样本数 |
| `train_acc` | 当前 epoch 在**训练集**上的分类准确率 | 预测正确的训练样本数 / 训练集总样本数 |
| `val_loss` | 当前 epoch 在**验证集**上的平均损失 | 同上，但用验证集数据，且不反向传播 |
| `val_acc` | 当前 epoch 在**验证集**上的分类准确率 | 验证集预测正确数 / 验证集总样本数 |
| `epoch_time_s` | 当前 epoch 的总耗时（秒） | 从 `t0 = time.perf_counter()` 到该 epoch 验证结束并同步后的时间差，包含训练和验证 |
| `samples_per_sec` | 训练吞吐量，粗略表示每秒处理多少训练样本 | `len(X_train) / epoch_time_s`。注意分母是整个 epoch 时间，里面还包含了验证时间，所以不是纯训练速度 |
| `gpu_alloc_MB` | 当前 epoch 结束时，GPU 上**已分配**的显存（MB） | `torch.cuda.memory_allocated() / 1024**2`，表示当前张量实际占用的显存 |
| `gpu_reserved_MB` | 当前 epoch 结束时，GPU 上**保留**的显存（MB） | `torch.cuda.memory_reserved() / 1024**2`，PyTorch 缓存分配器向 CUDA 申请的总显存，通常 ≥ `gpu_alloc_MB` |
| `gpu_max_alloc_MB` | 从上次重置峰值统计以来，GPU **已分配显存的峰值**（MB） | `torch.cuda.max_memory_allocated() / 1024**2`，记录过程中出现过的最大分配量 |
| `gpu_max_reserved_MB` | 从上次重置峰值统计以来，GPU **保留显存的峰值**（MB） | `torch.cuda.max_memory_reserved() / 1024**2` |
| `experiment` | 实验名称标签 | 代码里固定写成 `'origin experiment'`，用于区分不同实验 |

> 注意：每个 epoch 开始时调用了 `torch.cuda.reset_peak_memory_stats()`，所以 `gpu_max_alloc_MB` 和 `gpu_max_reserved_MB` 是**当前 epoch 内**的峰值，而不是整个训练过程的峰值。  
> 如果只用 CPU，这些显存字段都是 `0.0`。

2. 测试汇总表：`test_summary.csv / .xlsx`

| 表头 | 实际意义 | 计算方式 / 说明 |
|---|---|---|
| `experiment` | 实验名称标签 | 代码里写的是 `'exp1_lr_0.1'`，但实际 `LR = 0.01`，建议改为 `'exp_lr_0.01'` 或 `'origin experiment'` |
| `test_loss` | 测试集上的平均损失 | 每个 batch 损失 × batch 样本数，累加后除以测试集总样本数 |
| `test_acc` | 测试集上的分类准确率 | 测试集预测正确数 / 测试集总样本数 |
| `test_time_s` | **纯推理阶段**总耗时（秒） | 从 `test_start_time` 到 `test_end_time`，只包含测试循环，不包含加载模型、保存表格等 |
| `test_samples_per_sec` | 推理吞吐量，每秒处理多少测试样本 | `test_total / test_time` |
| `test_gpu_max_alloc_MB` | 测试阶段 GPU **已分配显存的峰值**（MB） | 从测试前 `reset_peak_memory_stats()` 之后，到测试结束时的 `max_memory_allocated()` |
| `test_gpu_max_reserved_MB` | 测试阶段 GPU **保留显存的峰值**（MB） | 同上，但取 `max_memory_reserved()` |

> 测试前也调用了 `torch.cuda.reset_peak_memory_stats()` 和 `torch.cuda.synchronize()`，所以这里的峰值是**纯测试阶段**的显存峰值，隔离了训练阶段的显存影响。  
----

## 具体分析
>下面从显存/速度/`test_acc`/`test_loss`这几个方面来看。

**分析方法**：将原始实验和几个对照试验的这些数据放到同一表格里比较
- 实际画表操作交给 AI~~（懒得再写合并表格的程序哩（bushi）~~, 使用命令如下:
上传四个`test_summary.csv`文件
```
请把上面四个实验的测试集数据，画成同一张 markdown 格式表格，按照0-3的顺序排列，表头用反引号括起来，并且每个数据只保留6位有效数字，最后整体用反引号括起来。
```
表格 be like：

| `experiment` | `test_loss` | `test_acc` | `test_time_s` | `test_samples_per_sec` | `test_gpu_max_alloc_MB` | `test_gpu_max_reserved_MB` |
| --- | --- | --- | --- | --- | --- | --- |
| MLP0: origin experiment (LR = 0.01; HIDDEN_UNITS = 16; HIDDEN_LAYERS = 1) | 0.0771641 | 0.973333 | 0.00545222 | 55023.5 | 64.0137 | 66.0000 |
| MLP1: LR = 0.1 | 0.0806701 | 0.973333 | 0.00511960 | 58598.3 | 64.0137 | 66.0000 |
| MLP2: HIDDEN_UNITS = 32 | 0.0780769 | 0.973333 | 0.00581980 | 51548.2 | 64.0176 | 66.0000 |
| MLP3: HIDDEN_LAYERS = 2 | 0.0834927 | 0.973333 | 0.0100705 | 29790.0 | 64.0195 | 66.0000 |

接下来，具体分析每个变量的影响：

### #1 LR

| `experiment` | `test_loss` | `test_acc` | `test_time_s` | `test_samples_per_sec` | `test_gpu_max_alloc_MB` | `test_gpu_max_reserved_MB` |
| --- | --- | --- | --- | --- | --- | --- |
| MLP0: origin experiment (LR = 0.01) | 0.0771641 | 0.973333 | 0.00545222 | 55023.5 | 64.0137 | 66.0000 |
| MLP1: LR = 0.1 | 0.0806701 | 0.973333 | 0.00511960 | 58598.3 | 64.0137 | 66.0000 |

可以观察到：
*相比 LR = 0.01, LR = 0.1 时*

1. loss 略有上升， acc 基本一致，这是因为大学习率导致模型暴力拟合的风险增大，所以模型能力略略下降。

2. 速度上升，这似乎直觉上符合学习率大更新步长大的规律，**但其实学习率只影响训练时的权重更新路径，不改变推理时的网络结构**（层数、神经元数、矩阵乘法规模均相同），所以推理速度理论上应该一致。

3. 显存占用：两者完全一致，因为神经网络结构相同，参数量和激活值大小不变。

### #2 HIDDEN_UNITS

| `experiment` | `test_loss` | `test_acc` | `test_time_s` | `test_samples_per_sec` | `test_gpu_max_alloc_MB` | `test_gpu_max_reserved_MB` |
| --- | --- | --- | --- | --- | --- | --- |
| MLP0: origin experiment (HIDDEN_UNITS = 16) | 0.0771641 | 0.973333 | 0.00545222 | 55023.5 | 64.0137 | 66.0000 |
| MLP2: HIDDEN_UNITS = 32 | 0.0780769 | 0.973333 | 0.00581980 | 51548.2 | 64.0176 | 66.0000 |

可以观察到：
*相比 HIDDEN_UNITS = 16, HIDDEN_UNITS = 32 时*

1. loss 略上升，acc 基本一致：说明这时神经元数量翻倍并没有带来收益，反而增加了一点过拟合的风险。

2. 速度下降，因为矩阵乘法维度变大，计算量增加，导致速度变慢。

3. 显存略高，因为神经元数量增加后，使用的模型参数也增加，所以显存需求增大，当然由于这个神经元数量还是很小的值，所以也没有明显变化。

### #3 HIDDEN_LAYERS

| `experiment` | `test_loss` | `test_acc` | `test_time_s` | `test_samples_per_sec` | `test_gpu_max_alloc_MB` | `test_gpu_max_reserved_MB` |
| --- | --- | --- | --- | --- | --- | --- |
| MLP0: origin experiment (HIDDEN_LAYERS = 1) | 0.0771641 | 0.973333 | 0.00545222 | 55023.5 | 64.0137 | 66.0000 |
| MLP3: HIDDEN_LAYERS = 2 | 0.0834927 | 0.973333 | 0.0100705 | 29790.0 | 64.0195 | 66.0000 |

可以观察到：
*相比 HIDDEN_LAYERS = 1, HIDDEN_LAYERS = 2 时*

1. loss 显著高，且是四个实验里最高的，acc 基本一致。因为增加隐藏层后，模型容量增大，过拟合程度显著增大，模型能力差。

2. 速度显著减慢，测试时间几乎是基线实验的两倍，因为增加一层隐藏层多了一次矩阵乘法和 ReLU 激活的数学计算，耗时显著增加。

3. 显存占用略略高，但差距很小。原因：虽然多一层网络增加了要处理的参数和中间值，这里也只是增加了一层而已，与原先的基础显存占用差距不大。

## 关于对照实验的一些些感悟 & 反思

事实上，在不熟悉神经网络的学习初期，我能感受到：我其实并不知道如何去选取超参数/设计模型，我猜想这些应当需要一定的阅读和实践积累，才会产生对应的经验。

所以这里我反思了一下：如果出于学习各种变量影响的目的来调整变量的话，我这里的变量选取是否跨度太小了？太保守了？相比原始实验，对照实验的变化似乎并没有那么大。

那么下次进行对照实验，*出于学习目的*，应当进行更加夸张地选取数值，或许能获得更显著的对比？同时，多选取连续梯度的不同数值或许可以更好地看出趋势，比如说 lr 取 0.01, 0.1, 1, 10等等:/

当然，*在现实优化模型的场景中*，给这些变量选取变化不大的范围内的数据或许可以让我们找到更加精确的最优数值。👍