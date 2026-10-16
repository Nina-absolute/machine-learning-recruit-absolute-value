# Transformer 论文精读笔记 · 概念解析 / 疑问 & 思考

**写在前面：**

本文是基于自己的阅读疑问汇总出来的 AI 解答。


> *论文：Attention Is All You Need (Vaswani et al., NIPS 2017)*
> *组织方式：按论文的章节顺序逐段整理。每一节分成三块：*
> *【论文说了什么】—— 原文要点，关键处附原文英文，方便对照。*
> *【概念解析】—— 把这一节出现的概念讲透，含公式的逐步推导。*
> **【*疑问*】** —— 每个疑问都保留提出背景（当时读到哪、想到哪一步），再给解答与「现在怎么理解 / 仍存疑」。

---

## 0. 全局速查（先记住这些数字，读后面会顺很多）

### 0.1 论文的超参数（已核对原文 Table 3 与第 3 节）

| 符号 | 含义 | base 模型取值 |
|---|---|---|
| `N` | Encoder / Decoder 各堆叠多少层 | 6 |
| `d_model` | 所有子层和嵌入层的输出维度 | 512 |
| `d_ff` | FFN 内部隐藏层维度 | 2048（= 4 × d_model） |
| `h` | 注意力头数 | 8 |
| `d_k = d_v` | 每个头的 Q/K/V 维度 | 64（= d_model / h = 512/8） |
| `P_drop` | Dropout 概率 | 0.1 |
| `ε_ls` | Label Smoothing | 0.1 |
| 训练步数 | base 模型 | 100K steps |
| 参数量 | base 模型 | 65 M |
| 优化器 | Adam | β₁ = 0.9，β₂ = 0.98，ε = 10⁻⁹ |
| warmup | 学习率先线性升、再按步数平方根倒数降 | warmup_steps = 4000 |

> *原文对多头维度的一句关键说明：*
> *"Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."*
> *—— 因为每个头的维度缩小了，多头的总计算量与「全维度的单头」差不多。*

### 0.2 三个复杂度（论文 Table 1，务必分清是三件不同的事）

论文第 4 节用三个指标比较不同层类型，它们不是一回事：

| 层类型 | 每层计算复杂度 | 顺序操作数（能否并行） | 最大路径长度（长依赖好不好学） |
|---|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent (RNN) | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional (CNN) | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

* $n$ = 序列长度，$d$ = 表示维度，$k$ = 卷积核大小，$r$ = 限制注意力的邻域大小。
* 顺序操作数 $O(1)$ 是 Transformer 能并行训练的根本原因（RNN 是 $O(n)$，必须一步步来）。
* 最大路径长度 $O(1)$ 是它能轻松学长距离依赖的原因（RNN 要走 $n$ 步、CNN 要堆 $\log_k n$ 层，路径越长梯度越容易衰减）。

> *⚠️ 初版笔记里的一处混淆（这里更正）：*
> *笔记把 $O(n)$ 写成「Transformer 的自注意力层」，其实反了。自注意力的顺序操作数是 $O(1)$（一步矩阵乘法算完所有位置），但它的计算量是 $O(n^2 \cdot d)$（$n \times n$ 的注意力矩阵，序列翻倍计算量变 4 倍）。一句话区分：「能不能并行」看顺序操作数，「要算多久」看计算复杂度。*

---

# Abstract

## 【论文说了什么】

> *"The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. … We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."*

一句话：用纯注意力取代 RNN 和 CNN，在机器翻译上效果更好、还能大幅并行。

## 【概念解析】

**什么是 sequence transduction（序列转换）？**
把一个序列映射成另一个序列的任务。机器翻译是最典型的例子：输入英文句子（长度 m）→ 输出德文句子（长度 n），m 和 n 通常不相等。

**什么是 sequence transduction model（序列转换模型）？**
解决这类任务的模型，经典结构是 encoder–decoder：Encoder 读入源序列，压成一个包含了整句信息的表示；Decoder 根据这个表示，一个词一个词地生成目标序列。

**RNN / LSTM / GRU 是什么关系？**

| 名称 | 全称 | 一句话 |
|---|---|---|
| RNN | Recurrent Neural Network | 最朴素：每个时间步把「上一步的隐藏状态」和「当前输入」一起算，得到新隐藏状态 |
| LSTM | Long Short-Term Memory | 给 RNN 加了「输入门/遗忘门/输出门」和一个细胞状态，缓解长序列上的梯度消失 |
| GRU | Gated Recurrent Unit | LSTM 的简化版：把三个门合并成两个（更新门、重置门），参数更少、训练更快 |

它们的共同点是一定要按顺序一个词一个词算 —— 这正是 Transformer 要打破的东西。

## 【*疑问*】

### *疑问* A-1｜什么是「序列传导模型」（sequence transduction model）？

*提出背景：读 Abstract 时看到 "dominant sequence transduction models"，不确定这个词具体指什么。*

**解答**：就是上面说的「序列到序列」的模型统称，不是某个具体模型的名字。它描述的是任务形态（序列 → 序列），而不是网络结构。RNN、CNN、Transformer 都可以做序列传导模型，区别在用什么机制来建模序列内部的关系。

*思考：这个词容易被误当成某种架构。记住它是问题类别，不是解法类别就够了。*

### *疑问* A-2｜LSTM 和 GRU 到底解决了 RNN 的什么问题？

*提出背景：笔记里只记了缩写 LSTM: long short-term memory、GRU: gated recurrent neural networks，没展开。*

**解答**：朴素 RNN 在长序列上会梯度消失（反向传播时梯度沿时间步连乘，越乘越小），导致它记不住远处信息。LSTM/GRU 用门控机制给信息开了一条高速公路：门的取值在 0~1 之间，可以近似实现原样保留或彻底丢弃，于是梯度不必每步都衰减，长距离信息得以传递。

*仍存疑 / 下一站：门控具体怎么写、为什么这样就能缓解梯度消失，需要单独读 LSTM 原论文，本笔记不展开。这不是本文的重点。*

---

# 1 Introduction

## 【论文说了什么】

原文核心三句：

> *"Recurrent models typically factor computation along the symbol positions of the input and output sequences. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples."*

> *"Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences."*

> *"In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output."*

## 【概念解析】

### 1.1 「固有的顺序性」（inherently sequential nature）到底卡在哪里

RNN 的计算方式是：
$h_1 = f(x_1, h_0)$
$h_2 = f(x_2, h_1)$
$h_3 = f(x_3, h_2)$

第 $t$ 步必须先有第 $t-1$ 步的结果。这带来两个后果：
1. 无法在样本内部并行：一句话里的 100 个词，必须串行算 100 次。
2. 显存限制进一步卡死批处理：因为要串行，一个 batch 里每条样本的中间状态都得存着，序列一长，batch size 只能开很小，训练更慢。

*这是 Transformer 最核心的动机：把「串行」换成「一步矩阵乘法算完全部位置」。*

### 1.2 Introduction 里埋下的三个伏笔

| 伏笔 | 论文在哪展开 | 展开后的结论 |
|---|---|---|
| 时间步（顺序操作数） | 第 4 节 Table 1 | Self-Attention 顺序操作数 $O(1)$，RNN 是 $O(n)$ → Transformer 可以并行 |
| 分解技巧与条件计算 | Introduction | 这些是别人用来降低 RNN 串行代价的补救手段，但没有解决根本问题；Transformer 直接换掉 RNN |
| 依赖关系在序列中的距离 | 第 4 节「最大路径长度」 | Self-Attention 的路径长度 $O(1)$，RNN 是 $O(n)$，CNN 是 $O(\log_k n)$ → 长依赖更好学 |

## 【*疑问*】

### *疑问* 1-1｜这篇论文最初处理的是什么任务？它为什么尝试不用 RNN 和 CNN？

*提出背景：作业正题第 4 问。*

**解答**：
* 任务：机器翻译，具体是 WMT 2014 英德（4.5M 句对）和英法（36M 句对）两个翻译任务。
* 为什么不用 RNN：RNN 有 inherently sequential nature（固有顺序性），第 $t$ 步依赖第 $t-1$ 步，无法在样本内部并行，长序列时速度慢到无法接受。
* 为什么不用 CNN：CNN 是局部操作，一个卷积核只看 $k$ 个邻域，要连接距离较远的两个位置必须堆叠 $O(n/k)$ 或 $O(\log_k n)$ 层，路径随距离增长，学长依赖效率低。

*思考：这两条理由分别对应「并行性」和「路径长度」，正是第 4 节 Table 1 的两个指标。换句话说，第 4 节是第 1 节这两句话的数学证明。*

### *疑问* 1-2｜「序列较长时内存限制限制了跨样本批处理」怎么理解？

*提出背景：笔记里原话「这种固有的顺序性阻碍了训练样本内的并行化，这在序列较长时变得至关重要，因为内存限制限制了跨样本的批处理」。*

**解答**：拆成两层：
1. 样本内不能并行 —— 一条 100 词的句子要算 100 步。
2. 样本间也受限 —— 虽然不同样本理论上可以并行，但每个样本都要保存整条链上的中间激活值用于反向传播；序列越长，单个样本占的显存越多，能塞进一个 batch 的样本数就越少。

结果就是「两头都堵」，训练极慢。Transformer 一次性解决第 1 条，并大幅缓解第 2 条。

---

# 2 Background

## 【论文说了什么】

> *"The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S … In these models, the number of operations required to relate signals from two arbitrary input or output positions grows … linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions."*

> *关于注意力函数："… additive attention … and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $1/\sqrt{d_k}$. … dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code."*

## 【概念解析】

### 2.1 加性注意力 vs 点积注意力

论文里两种注意力打分函数的区别只在于「怎么算兼容性」：

| | Additive Attention | Dot-Product Attention |
|---|---|---|
| 打分函数 | 用一个前馈网络算：$score = v^T \tanh(W_1 q + W_2 k)$ | 直接矩阵乘法：$score = q \cdot k$ |
| 直觉 | 先各自线性变换、相加、过 $\tanh$，再投影成一个标量 | 两个向量对齐程度（缩放后的余弦相似度） |
| 速度 | 慢（要走一遍小网络） | 快（一次大矩阵乘法，GPU 最擅长） |
| 论文选择 | | 点积（当 $d_k$ 不太大时，两者效果相当，但点积快得多、省得多） |

> *⚠️ 论文特意加了一句 "except for the scaling factor" —— 也就是说 Transformer 的注意力和「标准点积注意力」唯一的差别就是多除了一个 $\sqrt{d_k}$。为什么必须除，见 3.2.1。*

### 2.2 ByteNet / ConvS2S：CNN 路线的天花板

这两个模型都想用卷积替代 RNN 来减少串行计算。但它们仍有一个无法回避的问题：路径长度随距离增长。

| 模型 | 连接任意两位置需要多少层 | 路径长度 |
|---|---|---|
| ConvS2S | $O(n/k)$ | 线性增长 |
| ByteNet | $O(\log_k n)$ | 对数增长 |
| Transformer | $O(1)$ | 常数，与距离无关 |

### 2.3 Self-Attention 在 Background 里是怎么被引出的

论文的意思是：自注意力早就有人用过，但从来没有人把它当成「唯一的」序列转换机制。之前的做法都是「RNN/CNN + 注意力」的混合体。Transformer 的贡献是：把注意力从配角提到主角，其余全部拿掉。

## 【*疑问*】

### *疑问* 2-1｜Attention 和卷积操作到底有什么区别？

*提出背景：论文说 Transformer 取代了 CNN 和 RNN，但注意力机制看起来也是在「滑动/加权」，和卷积有什么区别？*

**解答**：

| | Attention | Convolution |
|---|---|---|
| 权重从哪来 | 动态计算：由 Q 和 K 的内容实时算出来 | 静态学习：卷积核的数值是训练学到的固定参数 |
| 感受野 | 全局：每个位置可以和序列里所有位置交互 | 局部：只看 $k \times k$ 邻域 |
| 权重依赖什么 | 内容（content-based） | 位置（position-based） |
| 复杂度 | $O(n^2 \cdot d)$ | $O(k \cdot n \cdot d^2)$ |

一句话：卷积是「位置决定权重」，注意力是「内容决定权重」。

### *疑问* 2-2｜Transformer 是不是完全废掉了局部性？

**解答**：不是。它靠两个东西保留了一定的局部性：
1. FFN 逐位置作用（每个位置独立变换，是一种逐点非线性）；
2. 多头中总会有些头学到「关注相邻词」（经验观察）。

但位置信息必须显式注入（3.5 的位置编码），否则自注意力对词序完全无感。

---

# 3 Model Architecture

## 【论文说了什么】

> *"Most competitive neural sequence transduction models have an encoder-decoder structure. … The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder."*

## 【概念解析】Figure 1 怎么看

* **左半边 = Encoder**：底部是 Inputs → Input Embedding + Positional Encoding → 然后 N×（Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm）→ 输出给 Decoder。
* **右半边 = Decoder**：底部是 Outputs (shifted right) → Output Embedding + Positional Encoding → 然后 N×（Masked Multi-Head Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm）→ 顶部 Linear → Softmax → Output Probabilities。
* **中间那条箭头**：Encoder 的最终输出被送进 Decoder 的每一个层（作为 Cross-Attention 的 K 和 V）。

三者最容易混的地方：

| | 有几种注意力子层 | 是什么 |
|---|---|---|
| Encoder 的一层 | 1 种 | Multi-Head Self-Attention |
| Decoder 的一层 | 2 种 | Multi-Head Masked Self-Attention + Multi-Head Cross-Attention |

## 【*疑问*】

### *疑问* 3-0｜Encoder 和 Decoder 在 Figure 1 的哪里？我能自己重画吗？

*提出背景：作业正题第 1 问。*

**解答**：位置见上面的「Figure 1 怎么看」。重画时建议抓住四条主线：
1. 数据流：Inputs → Embedding+PE → Encoder×6 → Decoder×6 → Linear → Softmax → 概率
2. 每层内部：每个子层外面都套「残差 + LayerNorm」
3. 注意力的三种用法：Encoder 自注意力 / Decoder 掩码自注意力 / 交叉注意力
4. 维度守恒：全程 $d_{model} = 512$ 不变（这是残差能相加的前提）

*思考：图里最容易看漏的是 Decoder 顶部还有一个 Linear + Softmax —— 它把 512 维的表示投影到词表大小，再变成每个词的概率。*

---

## 3.1 Encoder and Decoder Stacks

### 【论文说了什么】

> *"Encoder: … a stack of $N = 6$ identical layers. Each layer has two sub-layers. … We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is $\text{LayerNorm}(x + \text{Sublayer}(x))$ … all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model} = 512$."*

> *"Decoder: … the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. … We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions."*

### 【概念解析】

**① 一个「子层」（sub-layer）是什么？**
Encoder 一层里有两个子层：多头自注意力和 FFN；Decoder 一层里有三个（多一个交叉注意力）。每个子层都是一个「输入 512 维 → 输出 512 维」的模块。

**② 残差连接（Residual Connection）——解决什么问题？**
退化问题是「网络越深，训练误差反而越大」（注意不是过拟合，训练误差本身就变大了）。残差的解法是把 $x \to \text{Sublayer}(x)$ 改成：
$\text{输出} = \text{Sublayer}(x) + x$

**为什么能解决梯度消失？** 反向传播时，对 $y = F(x) + x$ 求导：
$dy/dx = dF/dx + 1$
即使 $dF/dx$ 非常小（趋近 0），梯度里永远有一个 1，所以梯度可以「原封不动」地往前传，不会衰减到 0。用大白话说：残差给梯度修了一条高速公路。

**③ 「包围」（around）是什么意思？**
图中画的是 $x$ 从左边分出一条线绕过子层、在右边和子层输出相加。所以残差不是在子层「之后」加的一步，而是把子层的输入和输出连起来：$x + \text{Sublayer}(x)$。这就是笔记里问的「包围」的含义。

**④ LayerNorm ——「训练的稳定性」指什么？**
先看它做什么：对单个样本的单个位置，把 512 维向量归一化成均值 0、方差 1，再用两个可学习参数 γ、β 缩放平移：
$\mu = (1/d) \sum x_i$
$\sigma^2 = (1/d) \sum (x_i - \mu)^2$
$\text{LN}(x) = \gamma \cdot (x - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta$

「稳定性」指的是：让每一层的输入分布保持在一个相对固定的范围内，后果是损失曲线更平滑、可以用更大学习率、收敛更快。

> *和 BatchNorm 的区别（容易混）：BN 是跨样本在同一个特征维上归一化（依赖 batch 里其他样本，batch 小就不稳）；LN 是单个样本内部跨特征归一化（与 batch size 无关），所以 NLP 里序列长度、batch size 多变时 LN 更合适。*

**⑤ 为什么偏要用 $\text{LayerNorm}(x + \text{Sublayer}(x))$ 这个顺序（Post-LN）？**
这是原论文的写法（Post-LN）。后来的工作（如 GPT-2、原始 ViT）发现把 LN 放到子层之前（Pre-LN：$x + \text{Sublayer}(\text{LN}(x))$）训练更稳、更容易堆深。本笔记按论文讲 Post-LN，但知道这个演进很有用。

### 【*疑问*】

#### *疑问* 3-1｜FFN 是什么？是「两层神经网络」吗？它是激活函数吗？

*提出背景：读到「每个子层都有残差、LayerNorm、还有前馈网络」时，分不清 FFN 和 ReLU 的关系。*

**解答**：
* FFN 是一个完整的子网络，不是激活函数。它由两个线性层 + 中间一个 ReLU 组成：
  $\text{FFN}(x) = \max(0, x \cdot W_1 + b_1) \cdot W_2 + b_2$
* 维度变化：$512 \to 2048 \to 512$（先升维 4 倍再降回来）。
* ReLU 才是激活函数，它只是 FFN 内部的一个零件。之所以会产生「FFN 是不是激活函数」的疑惑，是因为公式里 ReLU 的形状太显眼，容易把它当成 FFN 的本体。

*思考：为什么要先升维到 2048 再降回来？升维提供了更大的「临时工作空间」，让非线性变换有更强的表达能力；降回来是为了让残差能相加（必须回到 512 维）。*

#### *疑问* 3-2｜残差连接到底位于哪里？

*提出背景：看结构图时不确定残差是「一个层」还是「一个操作」。*

**解答**：残差不是独立一层，而是包裹每个子层的操作。Encoder 里出现 2 次/层，Decoder 里出现 3 次/层。图里的 Add 方框就是「加法」，那条绕过子层的弧线就是捷径。顺序严格是：子层 → Add（加上输入）→ LayerNorm。

#### *疑问* 3-3｜这三者（残差、LayerNorm、FFN）为什么要出现在这个位置？

*提出背景：不只是问「是什么」，还问「为什么在这儿」。*

**解答**：三者的位置各有理由：

| 组件 | 位置 | 为什么必须在这里 |
|---|---|---|
| 残差 | 包裹每一个子层 | 让梯度有通路、让深层可训练。任何子层都可能退化，所以每个都要包 |
| LayerNorm | 紧跟在 Add 之后 | 残差相加会把数值放大，正好需要归一化把它压回去。所以 LN 放在 Add 之后不是巧合 |
| FFN | 注意力子层之后 | 注意力是加权平均，本质是线性运算。如果只有注意力，整个模型仍然是线性的，表达能力受限。FFN 提供逐位置的非线性变换，是模型唯一的非线性来源（ReLU） |

> *一句话：注意力负责「在位置之间搬运信息」，FFN 负责「在每个位置上加工信息」。*

---

## 3.2 Attention

### 【论文说了什么】

> *"An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key."*

这段话是整个注意力机制的定义，值得逐字拆开 —— 它其实就是一个软化的查字典过程。

### 3.2.1 Scaled Dot-Product Attention

#### 【论文说了什么】

> *"We call our particular attention 'Scaled Dot-Product Attention'. … we compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values."*

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \tag{1}$$

#### 【概念解析】

**① 用「查字典」理解 Q / K / V**
* Query（查询）= 你心里想找的主题（「我想找关于猫的书」）。
* Key（键）= 每本书书脊上的标签/索引（「这本讲动物」「这本讲物理」）。
* Value（值）= 书里的实际内容。

流程：拿你的 Query 去和所有 Key 比对（打分），分数越高说明越匹配；然后按匹配度把各本书的内容混合起来 —— 这就是输出。

**② 四步计算，逐步拆解**

| 步骤 | 公式 | 形状变化 | 在干什么 |
|---|---|---|---|
| ① 打分 | $S = Q K^T$ | $(L, d_k) \cdot (d_k, L) \to (L, L)$ | 每个 query 和所有 key 做点积，得到 $L \times L$ 的相关性分数 |
| ② 缩放 | $S = S / \sqrt{d_k}$ | $(L, L)$ 不变 | 把方差拉回 1 附近，防止 softmax 饱和 |
| ③ 归一化 | $A = \text{softmax}(S, \text{dim}=-1)$ | $(L, L)$ | 按行 softmax，每行变成和为 1 的权重 |
| ④ 聚合 | $O = A V$ | $(L, L) \cdot (L, d_v) \to (L, d_v)$ | 用权重对所有 value 做加权求和 |

**③ 为什么要除以 $\sqrt{d_k}$？**
设 $q$、$k$ 的每个分量都是均值 0、方差 1 的独立随机变量，那么它们的点积 $q \cdot k = \sum_i q_i k_i$ 是 $d_k$ 个项的求和，其方差是 $d_k$，标准差是 $\sqrt{d_k}$。$d_k = 64$ 时，点积的典型幅度就是 ±8 上下；$d_k = 512$ 时能到 ±22。softmax 对数值幅度极其敏感：输入一大，它的输出就变成一个 1、其余全 0 的饱和状态，此时梯度几乎为 0，训练不动。

**④ 为什么必须 softmax？**
1. 要的是「加权平均」：输出是 $\sum w_i V_i$。只有当 $w_i \ge 0$ 且 $\sum w_i = 1$ 时，输出才是 V 的凸组合，数值量级才和 V 一致。
2. 要可微：softmax 处处可导，梯度能回传。
3. 要平滑：softmax 是「软化版的最大值」。

**⑤ 注意力矩阵里每一行代表什么？**
第 $i$ 行 = 第 $i$ 个 token 作为 query 时，把注意力分配给了哪些 key，行和为 1。

#### 【*疑问*】

##### *疑问* 3-4｜什么叫「Q、K、V 来自不同的序列」？

*提出背景：在理解 Self-Attention 与普通 Attention 差异时，看到「Self 是指 Q、K、V 来自同一序列，其他的来自不同序列」，不懂「不同序列」指什么。*

**解答**：在机器翻译里，源语言句子（英文）和目标语言句子（德文）是两个完全独立的序列 —— 长度不同、词汇不同、语序不同。
* 交叉注意力：Q 来自目标序列（德文侧，Decoder），K、V 来自源序列（英文侧，Encoder）。
* 自注意力：Q、K、V 全部来自同一个序列。

*思考：判断是 self 还是 cross，只需问一句：Q 和 K/V 是同一句里来的吗？*

##### *疑问* 3-5｜权重矩阵是什么？在哪里学习？训练中发生了什么改变？

*提出背景：读到 QKV 由「可学习的权重矩阵」线性变换而来。*

**解答**：
* 是什么：$W^Q$、$W^K$、$W^V$、$W^O$ 是可学习的参数矩阵。
* 初始化：随机。
* 在哪里学：在训练中通过反向传播 + 梯度下降更新 —— 和普通神经网络没有任何区别。
* 训练中改变什么：这四个矩阵的数值。
* 推理时：这四个矩阵完全固定，不再变化。

##### *疑问* 3-6｜怎么通过矩阵做线性变换？是矩阵乘法吗？和前向传播一样吗？

*提出背景：对线性变换的具体实现存疑。*

**解答**：是的，就是矩阵乘法，和 MLP 的前向传播一模一样：$Q = X \cdot W^Q$。

| 名称 | 写法 | 规则 | 维度变化 |
|---|---|---|---|
| 矩阵乘法 | $A \cdot B$ | 行乘列求和 | 会变 |
| 逐元素相乘 | $A \odot B$ | 对应位置相乘 | 不变 |
| 点积 | $a \cdot b$ | 对应分量乘积之和 | 两个向量 → 一个标量 |

三者关系：点积是矩阵乘法在向量上的特例。

##### *疑问* 3-7｜如何产生注意力权重？直接点积就可以？为什么点积的结果就「一定是」权重？

*提出背景：对点积 → 权重的跳跃存疑。*

**解答**：分三步看：点积给出的不是权重，是「分数」；softmax 才是把分数变成权重的步骤；为什么点积能表示相关性？几何上 $a \cdot b = |a||b|\cos\theta$。

##### *疑问* 3-8｜什么是「聚合」？为什么要用权重来聚合 V？V 不就是 token 的真实信息吗？

*提出背景：对聚合的本质存疑。*

**解答**：
* 聚合 = 加权求和：$\text{Output}_i = \sum_j w_{ij} \cdot V_j$。
* V 确实是内容，但单个 token 的 V 是孤立的、没有上下文的表示。
* 聚合的目的就是造出「上下文感知」的表示。

##### *疑问* 3-9｜什么叫「衡量自相似性」？请举具体例子

*提出背景：对自相似性的含义存疑。*

**解答**：自相似性 = 同一序列内部，token 之间的相关性。经典例子："The animal didn't cross the street because it was too tired." 处理 it 这个词时，它的 Query 会和句中所有词的 Key 做点积；与 animal 的 Key 点积高；softmax 后，animal 分到很高的注意力权重。

##### *疑问* 3-10｜点积到底是什么？请给具体计算过程

*提出背景：想要数学层面的理解。*

**解答**：定义：$a \cdot b = \sum_{i=1}^{d} a_i b_i$。算例：$[1,2,3] \cdot [4,5,6] = 4+10+18 = 32$。几何意义：$a \cdot b = |a||b|\cos\theta$。

##### *疑问* 3-11｜Self-Attention 的「Self」是什么意思？它和普通 Attention 有什么区别？

*提出背景：作业正题第 5 问。*

**解答**：
| | Self-Attention | 普通 Attention（Cross / Encoder-Decoder） |
|---|---|---|
| Q 来自 | 同一序列 | 目标序列（Decoder） |
| K、V 来自 | 同一序列 | 源序列（Encoder） |
| 在算什么 | 序列内部 token 之间的相关性 | 两个序列之间的匹配度 |

---

### 3.2.2 Multi-Head Attention

#### 【论文说了什么】

> *"Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively. … Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."*

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{其中} \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### 【概念解析】

**① 执行顺序**
1. $h$ 组不同的线性投影：把 Q, K, V 分别投到 $d_k = d_v = d_{model}/h$ 维
2. 对这 $h$ 组并行做缩放点积注意力
3. 把 $h$ 个输出 Concat 拼接，回到 $h \cdot d_v = d_{model}$ 维
4. 再乘一次 $W^O$ 做输出投影

**② 为什么不能只用一个头？——「averaging inhibits this」**
论文这句话是全节的关键：单个头会把注意力「平均」掉。一个头只有一套 Q/K/V 投影，它只能表达一种「什么叫相关」的度量。面对「it 指代 animal」（语义关系）和「形容词修饰名词」（语法关系）这类不同性质的关系，一个头必须在它们之间折中，结果就是每种关系都学得糊。

**③ 多头不是「重复多次」**
每个头有独立的 $W_i^Q$、$W_i^K$、$W_i^V$，投影到的子空间不同，算出来的注意力矩阵也不同。如果多头就是把 512 维的单头跑 8 遍，算力会变成 8 倍。但论文说每个头只有 64 维，所以总计算量和全维度单头差不多。

**④ $W^O$ 的作用**
拼接后的 512 维只是「8 段各自为政的信息并排摆放」。$W^O$ 是一次跨头的线性融合，让不同头的信息能够交互、重新组合。

**⑤ head / subspace / sub-layer —— 三个词必须分清**

| 术语 | 中文 | 含义 | 关系 |
|---|---|---|---|
| head | 头 | 一套独立的 Q/K/V 投影 + 一次注意力计算 | 一个 head 投到一个 subspace |
| subspace | 子空间 | $d_{model}$ 维空间中被一个头使用的 $d_k$ 维切片 | 是 head 工作的维度范围 |
| sub-layer | 子层 | 网络结构上的一个模块（如多头注意力、FFN） | 是架构层的概念，与维度无关 |

#### 【*疑问*】

##### *疑问* 3-12｜为什么要分成多个头？多个头只是把同一件事重复多次吗？

*提出背景：作业正题第 8 问。*

**解答**：不是重复。一个头 = 考察 token 之间关系的一种「角度」（语义、语法、位置、指代……）。多头 = 同时从多个角度考察，并巧妙实现并行。

##### *疑问* 3-13｜每个子空间里到底有什么？子空间和子层有什么区别？

*提出背景：对子空间的具体内容存疑。*

**解答**：
* 子空间里有什么：它是训练中学到的一个 64 维投影方向集合。
* 和子层的区别：一个是维度概念，一个是架构概念，不在一个层面上。

---

### 3.2.3 Applications of Attention in our Model

#### 【论文说了什么】

> *"The Transformer uses multi-head attention in three different ways:*
> * *In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.*
> * *The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place.*
> * *Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow … by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections."*

#### 【概念解析】

| 用法 | Q 来自 | K、V 来自 | 出现在 | 作用 |
|---|---|---|---|---|
| ① Encoder Self-Attention | 前一 Encoder 层输出 | 同左（同源） | Encoder 每层第 1 子层 | 编码器内部捕捉上下文，双向（无 mask） |
| ② Decoder Masked Self-Attention | 前一 Decoder 层输出 | 同左（同源） | Decoder 每层第 1 子层 | 生成时只能看已生成的部分（单向） |
| ③ Encoder-Decoder Cross-Attention | Decoder | Encoder 最终输出 Z | Decoder 每层第 2 子层 | 让解码器「查看」整句源语言 |

**Mask 是怎么实现的？**
具体做法：构造一个上三角为 1 的掩码矩阵，把未来位置的分数设成 $-\infty$。经过 softmax 后，$e^{-\infty} = 0$，那些位置的权重精确地变成 0。

#### 【*疑问*】

##### *疑问* 3-14｜论文说的是 self-attention，为什么又出现 cross attention？

*提出背景：对 self 和 cross 的共存存疑。*

**解答**：因为论文标题里的 "Attention" 是泛指，而 Transformer 在三处用了多头注意力，其中两处是 self、一处是 cross。它们共用同一个 $\text{Attention}(Q,K,V)$ 函数，只是 Q/K/V 的来源不同。

##### *疑问* 3-15｜不同 attention 的 Q、K、V 分别来自哪里？为什么要经过不同的线性变换？

*提出背景：作业正题第 6 问。*

**解答**：来源见上面「三种用法对照表」。为什么要不同的线性变换？两个理由：角色不同，需要的表示就不同；避免只比较原始词嵌入的单一相似度。

##### *疑问* 3-16｜Decoder 中为什么需要 Masked Attention？提前看到后面的答案会怎样？

*提出背景：作业正题第 10 问。*

**解答**：解码器是自回归的——生成第 $t$ 个词时，第 $t+1$ 个词还不存在。但训练时为了并行，我们把整句目标句一次性喂进去。如果不加 mask，模型在第 $t$ 个位置就能看到第 $t+1$ 个位置的真实词——信息从未来泄漏到过去。如果能看到会怎样：模型会发现「直接抄下一个词」是让损失最小的捷径，于是根本不学从上下文推断词，只学复制。

---

## 3.3 Position-wise Feed-Forward Networks

### 【论文说了什么】

> *"… a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between."*

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \tag{2}$$

> *"While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1."*

### 【概念解析】

**① 「position-wise」是什么意思？**
就是对序列中每一个位置独立地、用同一套参数做变换。位置之间没有任何交互——FFN 不混合不同 token 的信息，那是注意力子层的活。

**② 为什么等价于两个 1×1 卷积？**
把序列看成 $(batch, d_{model}, L)$ 的「一维图像」，那么 1×1 卷积就是在每个位置上做一次线性变换、通道数从 $d_{model}$ 变到 $d_{ff}$。两次 1×1 卷积正是 FFN 的两个线性层。

**③ 维度：512 → 2048 → 512**
先升维 4 倍提供表达空间，再降回 512 以便残差相加。

**④ 「不同层之间参数不同，同一层内所有位置共享」**
是的，就是连续两个线性变换，但不是在整个模型里共享，而是每层各自一套。

### 【*疑问*】

#### *疑问* 3-17｜FFN 的公式和 ReLU 那么像，它是一种激活函数吗？

*提出背景：对 FFN 和 ReLU 的关系存疑。*

**解答**：不是。激活函数：逐元素的非线性，输入输出同维度（如 ReLU: R→R）。它没有参数。FFN：是一个带参数的子网络（两个权重矩阵），输入输出也是 512 维，但内部升到 2048 维再降回来。它包含激活函数，但它本身不是。

#### *疑问* 3-18｜为什么注意力之后还要 FFN？注意力不够吗？

**解答**：不够，因为注意力本质是线性的。注意力的输出是 $\sum w_i V_i$，其中 $V = XW^V$ 是线性的，$w$ 虽然经过 softmax（非线性），但它是作用在加权系数上，对 V 的聚合过程仍是线性组合。如果整个模型只有注意力 + 线性投影，叠多少层都还是线性的，表达能力极其有限。

---

## 3.4 Embeddings and Softmax

### 【论文说了什么】

> *"… we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation … In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$."*

### 【概念解析】

**① 词嵌入（Embedding）是什么**
一张查找表：词表大小 $V$ × 维度 $d_{model}$。输入一个 token id，取出一行向量。是学出来的（learned），因为模型需要把离散的符号映射到有语义结构的连续空间。

**② 三处权重共享（weight tying）**
输入嵌入、输出嵌入、Softmax 前的线性层共用同一个矩阵。
* 好处：参数量大幅减少，且效果往往更好。
* 直观理由：输入和输出用的是「同一套词义空间」。

**③ 为什么嵌入要乘 $\sqrt{d_{model}}$？**
位置编码的数值幅度大致是 O(1)，而随机初始化的嵌入向量幅度很小（≈ $1/\sqrt{d_{model}}$ 量级）。把嵌入乘上 $\sqrt{d_{model}}$ 是为了把两者的量级对齐，否则位置信息会淹没词义信息（或被淹没）。

**④ 输出层**
$\text{Linear}(d_{model} \to V)$ + softmax → 词表上的概率分布 → 和独热标签算交叉熵。

### 【*疑问*】

#### *疑问* 3-19｜独热标签是什么？这种分布是什么？是人工设定的吗？

*提出背景：学习交叉熵损失时涉及「真实标签」。*

**解答**：
* 独热标签（one-hot）：一个长度 = 词表大小的向量，只有目标词的位置是 1，其余全是 0。
* 是什么分布：一个退化（确定性）的类别分布 —— 概率 1 给某一个类别，其余为 0。
* 是人工设定的吗：是的。它来自训练语料的真实目标句，在数据预处理阶段就生成好了，不是模型产生的。
* 现实场景：英译德训练时，目标句是 "Ich liebe dich"。第一个预测步的标签就是 "Ich" 在词表里的独热向量。模型输出的是概率分布，把模型分布和独热真实分布放在一起算交叉熵。

*思考：注意论文用了 label smoothing = 0.1，即把 $[0,1,0]$ 软化成 $[0.05, 0.9, 0.05]$（大致）。论文原话："This hurts perplexity … but improves accuracy and BLEU score." —— 让模型别那么自信，泛化反而更好。*

#### *疑问* 3-20｜交叉熵损失是什么？

*提出背景：对损失函数的细节存疑。*

**解答**：
* 定义：衡量两个概率分布的差异。
  $$L = -\sum_{i} y_i \log(\hat{y}_i)$$
* 在分类任务里会大幅简化：因为 $y$ 是独热向量，只有目标位置是 1，其余项都被 0 乘掉了：
  $$L = -\log(\hat{y}_{\text{target}})$$
* 含义：只关心「模型给正确答案分配了多大概率」，给得越低，惩罚越大（对数级增长）。

| 模型给正确答案的概率 | 交叉熵损失 |
|---|---|
| 0.99 | 0.01 |
| 0.90 | 0.11 |
| 0.50 | 0.69 |
| 0.10 | 2.30 |
| 0.01 | 4.61 |

*思考：为什么用 $-\log$ 而不是直接用 $1-p$？因为 $-\log$ 在 $p \to 0$ 时趋向无穷，对「极其自信地答错」施加了极重的惩罚；而 $1-p$ 最多只有 1。*

---

## 3.5 Positional Encoding

### 【论文说了什么】

> *"Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. … we add 'positional encodings' to the input embeddings at the bottoms of the encoder and decoder stacks."*

$$PE_{(pos, 2i)} = \sin\left(pos / 10000^{2i / d_{model}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i / d_{model}}\right)$$

### 【概念解析】

**① 为什么非要位置编码？**
自注意力是集合运算——它对输入顺序完全不敏感。把一句话的词打乱，注意力算出来的结果（在 softmax 之前）是同一个多重集的运算，模型无法区分「狗咬人」和「人咬狗」。所以必须显式地把位置信息注入进去。

**② 正弦编码的巧妙之处**
* 每个维度 $i$ 对应一个不同频率的正弦波，波长从 $2\pi$ 到 $10000 \cdot 2\pi$ 构成等比数列。
* 固定、不学习（论文对比了 learned 版本，效果差不多，但正弦版可以外推到比训练时更长的序列）。
* 相对位置可线性表示：对任意固定偏移 $k$，$PE(pos+k)$ 可以写成 $PE(pos)$ 的线性变换。

**③ 怎么加进去？**
和词嵌入逐元素相加（不是拼接），得到最终输入。这也是为什么 3.4 要把嵌入乘 $\sqrt{d_{model}}$ —— 两者量级要对齐才能相加。

### 【*疑问*】

#### *疑问* 3-21｜Transformer 没有循环结构，它怎样知道词语的先后顺序？

*提出背景：作业正题第 9 问。*

**解答**：靠 positional encoding。虽然不用 RNN/CNN，但模型把正弦位置编码加到编码器和解码器底部的词嵌入上，从而让每个 token 携带一个与其绝对位置绑定的向量，把顺序信息注入进去。

*思考：注意区分两个词：Positional Encoding（位置编码）是论文用的正弦函数，固定不学习；Positional Embedding（位置嵌入）是让位置向量也参与训练（BERT/GPT 用的是这种）。*

---

# 4 Why Self-Attention

## 【论文说了什么】

论文用三个指标（"three desiderata"）比较自注意力、循环层、卷积层：
1. 每层计算复杂度
2. 可并行程度
3. 长距离依赖的路径长度

结果就是 0.2 节那张 Table 1。论文结论：
> *"… a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $O(n)$ sequential operations."*

## 【概念解析】

**① 为什么「最大路径长度」这么重要？**
路径长度 = 前向传播时，信息从位置 $i$ 传到位置 $j$ 要经过多少步运算。路径越短，梯度回传越直接，越容易学到依赖关系。

| 层类型 | 最大路径长度 | 含义 |
|---|---|---|
| Self-Attention | $O(1)$ | 任意两位置一步直达 |
| Recurrent | $O(n)$ | 要走 $n$ 步，长依赖难学 |
| Convolutional | $O(\log_k n)$ | 要堆 $\log_k n$ 层才能连上远处 |
| Restricted SA | $O(n/r)$ | 只看邻居 $r$ 个，退化成局部 |

**② 计算复杂度的权衡**
自注意力的 $O(n^2 \cdot d)$ 在短序列上比 RNN 的 $O(n \cdot d^2)$ 更贵（当 $n < d$ 时，$n^2 d < n d^2$）；但在长序列上 $n^2$ 项占主导，会变得不可接受。

**③ 论文提出的折中：restricted self-attention**
只看邻域 $r$ 个位置，复杂度降到 $O(r \cdot n \cdot d)$，路径长度变成 $O(n/r)$。目的是提升长序列的计算效率、防止内存爆炸。

**④ 第四个理由：可解释性**
论文还提了一句：注意力权重矩阵可以可视化，"individual attention heads clearly learned to perform different tasks"。

## 【*疑问*】

### *疑问* 4-1｜$O(n)$、$O(1)$、$O(n^2)$、$O(n/k)$、$O(\log_k n)$ 分别指什么？为什么容易记混？

*提出背景：笔记里有一大段关于 $O(n)/O(1)/O(n^2)$ 的记录，说明当时读得很吃力。*

**解答**：记混的根源是没区分「三列」。论文 Table 1 明明是三列，但笔记把它们混在一段话里了。正确的对应关系：

| 你看到的 | 它属于哪一列 | 谁的值 |
|---|---|---|
| $O(n^2 \cdot d)$ | 计算复杂度 | Self-Attention |
| $O(n \cdot d^2)$ | 计算复杂度 | Recurrent |
| $O(k \cdot n \cdot d^2)$ | 计算复杂度 | Convolutional |
| $O(1)$ | 顺序操作数 | Self-Attention / Convolutional |
| $O(n)$ | 顺序操作数 | Recurrent |
| $O(1)$ | 最大路径长度 | Self-Attention |
| $O(n)$ | 最大路径长度 | Recurrent |
| $O(\log_k n)$ | 最大路径长度 | Convolutional |
| $O(n/r)$ | 最大路径长度 | Restricted Self-Attention |

*记忆口诀：*
*「能不能并行」看顺序操作数（自注意力 $O(1)$ ✅）*
*「要算多久」看计算复杂度（自注意力 $O(n^2)$ ⚠️）*
*「长依赖好不好学」看最大路径长度（自注意力 $O(1)$ ✅）*
*自注意力是「两优一劣」：并行好、路径短，但序列长了计算量爆炸。*

---

# 5 Training

## 【论文说了什么】

| 小节 | 关键内容（已核对原文） |
|---|---|
| 5.1 训练数据与批处理 | WMT 2014 英德 4.5M 句对（BPE 37000 词表）；英法 36M 句（32000 word-piece）；按 token 数组 batch，base 模型每批约 25000 源 + 25000 目标 token |
| 5.2 硬件与时间 | 8 × NVIDIA P100；base 模型 12 小时（100K 步，每步 0.4 秒）；big 模型 3.5 天（300K 步，每步 1.0 秒） |
| 5.3 优化器 | Adam，β₁=0.9，β₂=0.98，ε=10⁻⁹；学习率公式见下 |
| 5.4 正则化 | ① 残差 dropout（每个子层输出后，base P=0.1）② 嵌入+位置编码的 dropout ③ Label Smoothing ε=0.1 |

**学习率公式（论文式 (3)）——这是 Transformer 最著名的训练技巧之一：**

$$\text{lrate} = d_{model}^{-0.5} \cdot \min\left(\text{step\_num}^{-0.5}, \text{step\_num} \cdot \text{warmup\_steps}^{-1.5}\right)$$

其中 `warmup_steps = 4000`。论文解释：
> *"This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number."*

## 【概念解析】

**① 为什么要 warmup（先升后降）？**
* 前 4000 步线性上升：训练刚开始时参数是随机的，梯度方向不可靠。如果一上来就用大学习率，可能把参数推到很差的区域。先用小学习率「热身」，让模型先找到一个大致的下降方向。
* 之后按 $\text{step}^{-0.5}$ 下降：接近收敛时需要小步走，避免在最优点附近震荡。

**② 三种 dropout 分别加在哪？**
1. 子层输出：每个子层（注意力 / FFN）的输出在进入 Add 之前做 dropout；
2. 嵌入 + 位置编码之和：在 Encoder/Decoder 最底部；
3. 注意 big 模型（英法）用的是 P=0.3，base 是 0.1。

**③ 推理时的解码策略**

| | 训练 | 推理 |
|---|---|---|
| 输入 | 完整目标句右移一位（`<SOS> Ich liebe dich`） | 只有源句 + `<SOS>` |
| 是否并行 | 并行（整句一次算完，靠 mask 防泄漏） | 串行（一个词一个词生成） |
| 上一词的来源 | 真实的词（Teacher Forcing） | 模型自己刚预测的词 |
| 搜索方式 | — | Beam Search（beam size = 4，长度惩罚 α = 0.6，论文 6.1 节） |
| 权重 | 更新 | 固定 |

## 【*疑问*】

### *疑问* 5-1｜训练 vs 推理，到底分别改变了什么？完整流程是什么？

*提出背景：对训练和推理的区别存疑。*

**解答**：

| | 训练 | 推理 |
|---|---|---|
| 权重矩阵 $W^Q/W^K/W^V/W^O$ | 参数，每步由反向传播更新 | 固定不变 |
| 改变的只有 | 这些矩阵的数值 | 输入序列（自回归增长） |
| 损失 | 每次前向都算，用于更新 | 不算损失 |

**英译德的完整流程（逐步走一遍）**

**训练时（源句 "I love you" → 目标句 "Ich liebe dich"）：**
1. 分词：`"I love you"` → `["I","love","you"]`；目标句加 `<SOS>/<EOS>`
2. 嵌入+PE：每个 token → 512 维向量，加上位置编码
3. Encoder：6 层，每层：自注意力 → Add&Norm → FFN → Add&Norm；输出记忆 Z
4. Decoder输入：目标句右移 → `["<SOS>","Ich","liebe","dich"]`
5. Decoder：6 层，每层三步：掩码自注意力；交叉注意力（Q←解码器，K/V←编码器的 Z）；FFN
6. 输出投影：$\text{Linear}(512 \to \text{词表大小}) + \text{Softmax}$ → 下一个词的概率分布
7. 算损失：与真实标签（独热）算交叉熵，反向传播，更新所有参数

**推理时：**
1. 只有源句，没有目标句
2. Encoder 照样跑一遍，得到 Z（这部分和训练一样）
3. 输入 `<SOS>`，Decoder 预测第一个词 → "Ich"
4. 把 "Ich" 接到输入后面 → 预测 "liebe"
5. 重复，直到生成 `<EOS>`
6. 用 Beam Search 保留概率最高的 k 条路径（论文用 4），避免贪心陷入局部最优

*几个必须知道的细节：*
* *BPE / WordPiece 分词：把词拆成子词单元，解决未登录词问题。*
* *`<SOS>` / `<EOS>`：句首/句尾特殊标记。*
* *Teacher Forcing：训练时每一步都用真实的上一词作为输入。*
* *Beam Search：每步保留概率最高的 k 条完整路径前缀。*

*思考：训练与推理的最大区别其实就一句话：训练时「答案」是给定的（教师强制），推理时「答案」是自己造的（自回归）。所有别的差异都是这句话的推论。*

---

# 6 Results / 7 Conclusion（简记）

* WMT 2014 英德：Transformer (big) BLEU 28.4，比此前最好结果高 2 个 BLEU 以上。
* WMT 2014 英法：BLEU 41.8（单模型），训练成本不到此前 SOTA 的 1/4。
* 消融实验：Table 3 显示头数 h：1 头明显差（单头 24.9 vs 8 头 25.8），证明确实需要多头；但头太多（32 头）也变差。
* 结论：Transformer 是第一个完全依赖注意力的序列转换模型，训练快、效果好。

---

# 拓展一：代码实现流程与热力图

> *对应作业拓展第 1、2 问。*

## 单头缩放点积注意力

### 代码流程

1. 设定超参数：`SEED`、`BATCH_SIZE`、`SEQ_LEN`、`D_MODEL`、`D_K`。
2. 随机生成输入数据 `X`，形状 `(BATCH_SIZE, SEQ_LEN, D_MODEL)`。
3. 初始化三个权重矩阵 `W_q`、`W_k`、`W_v`，形状 `(D_MODEL, D_K)`。
4. 计算 Q、K、V：`Q = X @ W_q`，`K = X @ W_k`，`V = X @ W_v`。
5. 计算注意力分数：`scores = Q @ K.T / sqrt(D_K)`。
6. 计算注意力权重：`attn_weights = softmax(scores, dim=-1)`。
7. 打印分数矩阵和权重矩阵的形状及数值。
8. 绘制两张热力图：左图为分数矩阵，右图为权重矩阵。

### 热力图含义

* **左图（注意力分数）**：颜色深浅代表 Query 和 Key 的原始点积分数，有正有负。
* **右图（注意力权重）**：颜色深浅代表 Softmax 归一化后的概率值，每行和为 1。第 $i$ 行表示第 $i$ 个 token 作为 Query 时，对序列中所有 Key 分配的注意力权重。

## 多头注意力

### 代码流程

1. 设定超参数：`SEED`、`BATCH_SIZE`、`SEQ_LEN`、`D_MODEL`、`NUM_HEADS`、`D_K = D_MODEL // NUM_HEADS`。
2. 随机生成输入数据 `X`。
3. 初始化权重矩阵 `W_q`、`W_k`、`W_v`、`W_o`，形状均为 `(D_MODEL, D_MODEL)`。
4. 计算 Q、K、V 并拆分为多头：`Q = X @ W_q`，然后 `view` + `transpose` 得到 `(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)`。
5. 计算注意力分数：`scores = Q @ K.transpose(-2, -1) / sqrt(D_K)`。
6. 计算注意力权重：`attn_weights = softmax(scores, dim=-1)`。
7. 加权求和并拼接：`context = attn_weights @ V`，然后 `transpose` + `view` 回到 `(BATCH_SIZE, SEQ_LEN, D_MODEL)`。
8. 最终线性变换：`output = context @ W_o`。
9. 打印分数矩阵和权重矩阵的形状（每个头一张权重矩阵）。
10. 为每个头绘制一张注意力权重热力图。

### 热力图含义

每张子图对应一个头（子空间）的注意力权重矩阵。对比不同头的热力图，可以直观看出模型从不同角度捕捉了序列关系（例如 Head 1 关注局部，Head 2 关注远距离依赖）。这证明多头机制不是简单重复，而是从不同投影子空间学习多样化的关注模式。

# 拓展二：ViT（Vision Transformer）

> *对应作业拓展第 3 问。*

**输入是什么？**
把图像切成固定大小的图块（如 16×16），展平，线性投影成嵌入向量，再加上位置编码 —— 于是图片变成了一个「图块序列」。

**Attention 在哪里起作用？**
自注意力作用在图块序列上，允许每个图块关注其他任何图块，从而捕捉图像的全局上下文。

**与原论文有什么不同？**

| | 原 Transformer | ViT |
|---|---|---|
| 架构 | Encoder + Decoder | Encoder-only |
| 任务 | 生成（翻译） | 分类 |
| 输入单元 | 词（子词） | 图像图块 |
| 位置编码 | 1D 正弦（固定） | 2D 位置嵌入（可学习） |
| 输出头 | Linear + Softmax → 词表 | 标准分类头 |
| 特殊 token | `<SOS>`/`<EOS>` | `[CLS]` token |

---

# 拓展三：Encoder-only vs Decoder-only

> *对应作业拓展第 4 问。*

| | Encoder-only：BERT | Decoder-only：GPT |
|---|---|---|
| 结构 | 多层 Transformer Encoder 堆叠 | 多层 Transformer Decoder 堆叠（去掉交叉注意力层） |
| 注意力方向 | 双向（无 mask） | 单向（因果掩码 Causal Mask） |
| 训练目标 | MLM（完形填空）+ NSP | CLM（自回归预测下一个词） |
| 适用任务 | 理解型：分类、NER、抽取式问答 | 生成型：文本生成、对话、代码补全 |
| 为什么这样选 | 不需要生成序列，双向注意力能更好理解上下文 | 自回归属性契合生成任务 |

*思考：为什么不同任务选不同结构？关键在于注意力方向和训练目标必须匹配任务：能看未来 → 适合理解；不能看未来 → 适合生成。*

---

# 附录：本次整理中修正的 4 处理解偏差

诚实记录，避免以后再错：

| # | 原来的说法 | 修正后 | 为什么重要 |
|---|---|---|---|
| 1 | $O(n)$ 是「Transformer 的自注意力层」 | 自注意力的顺序操作数是 $O(1)$，计算复杂度才是 $O(n^2 \cdot d)$ | 把 Table 1 的三列混成了一列，会得出「Transformer 比 RNN 慢」的错误结论 |
| 2 | 「一个 head 其实就是 subplace（subspace）」 | head 是计算单元，subspace 是它作用的 $d_k$ 维切片；两者不等同，且都与 sub-layer（架构层）不是一个层面的概念 | 三个术语混用会导致后面理解多头时含糊 |
| 3 | 「不 softmax 会因为点积数值大而梯度爆炸」 | 更准确的说法：softmax 是为了得到凸组合（权重和为 1）；防止数值过大是「除以 $\sqrt{d_k}$」那一步负责的 | 缩放和 softmax 是两件事，各自解决不同问题 |
| 4 | FFN「和 ReLU 很像，是不是激活函数」 | FFN 是带参数的子网络（两层线性 + 中间 ReLU），ReLU 只是它的一个零件 | 概念层级不同（网络 vs 零件），混淆会影响对「非线性来自哪里」的判断 |

---