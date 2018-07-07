# 第5课:Word2vec和管理实验
---
我们已经建立了几个非常简单的模型，它们只需要几分钟就能训练完毕。如果要训练更复杂的模型，我们需要一些更多的工具。在这节课中，我们将介绍模型库、变量共享、模型共享以及如何管理你的实验。我们将会用word2vec作为例子演示这些。

## Word2vec
你也许还不了解词嵌入（word embedding），那么你应该看看[Stanford CS 224N的词向量课程](http://web.stanford.edu/class/cs224n/lectures/lecture2.pdf)。了解之后，跟一下这两篇论文是一个好主意：

- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)(Mikolov et al., 2013),
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)(Mikolov et al., 2013)

在较高的层面上，我们需要找到一个表示文本数据（比如词汇）的方法来让我们能将其用于解决自然语言处理任务。在像语言建模、机器翻译和语义分析等任务的解决方案中词嵌入是核心。

Tomas Mikolov带领的研究团队提出的word2vec是一组用来做词嵌入的模型，其中有两种主要的模型：skip-gram和CBOW。

> **Skip-gram vs CBOW**（Continuous Bag-of-Words）

> 在算法上，这两个模型十分相似，除了CBOW是从上下文词（context words）预测中间词（center words），而skip-gram与其相反是从中间词预测上下文词。

> 比如我们有一句话："The quick brown fox jumps"，以"brown"为中心词，然后CBOW尝试从"the"、"quick"、"fox"和"jumps"去预测"brown"，而skip-gram尝试从"brown"去预测"the"、"quick"、"fox"和"jumps"。

> 统计上看，CBOW平滑了很多分布信息（把整个上下文当成一次观察），这使它在较小的数据集上很有用。而skip-gram将每个上下文词当做一个新的观察，它在大数据集上效果更好。

在这节课中，我们将会建立word2vec的skip-gram模型。为了获得词汇的向量表示，我们训练一个简单的单隐层神经网络来完成一个特定任务（假任务，fake task），但是之后我们不会用我们训练的这个神经网络来完成skip-gram的任务。相反，我们只关心假任务训练完后隐层的权值，这些权值被称为词向量（word vector）或者嵌入矩阵（embedding matrix）。

我们要去训练模型的假任务是通过给定的中心词去预测上下文词，在一句话中指定一个中心词，查看它附近并随机选取一个词作为标签。这个网络将会告诉我们词汇表中的每一个词作为中心词的邻居的概率。[这里](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)有一个解释skip-gram模型细节的精彩教程。

在TensorBoard中用t-SNE将词向量投影到3D空间上：

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_skip-gram-t-SNE.jpg)

## Softmax，负采样（Negative Sampling）和NCE（Noise Contrastive Estimation）

获得可能的邻近词的分布，在理论上，我们经常用softmax。Softmax将一组随机值\\(x_i\\)映射成一组和为1的概率值\\(p_i\\)。在这种情况下，\\(softmax(x_i)\\)表示\\(x_i\\)是指定的中心词的邻近词的概率。

$$softmax(x_i) = exp(x_i) / ∑_i exp(x_i)$$

然而，分母的标准形式需要我们计算字典中所有词（可能有几百万个）的exp并求和。就算去掉不常用的词，一个自然语言模型也必须考虑至少成千上万个最常用词，标准形式的softmax还是不好计算。

这里有两个主要的方法可以规避这个瓶颈：分层softmax（hierarchical softmax）和基于采样的softmax（sample-based softmax）。Mikolov团队在他们的论文中展示了负采样加速了skip-gram模型的训练，并对比了更复杂的分层softmax。

负采样顾名思义属于基于采样的方法族，这个方法族还包括重要性采样（importance sampling）和目标采样（target sampling）。负采样是一种叫做Noise Contrastive Estimation（NCE）方法的简化模型。负采样对产生的噪声样本数量\\(k\\)和噪声样本分布\\(Q\\)作了一定的假设，例如\\(kQ(w)=1\\)，这样可以简化计算。更多的细节可以看Sebastian Rudder的[On word embeddings - Part 2: Approximating the Softmax](http://ruder.io/word-embeddings-softmax/)和Chris Dyer的[Notes on Noise Contrastive Estimation and Negative Sampling](http://demo.clab.cs.cmu.edu/cdyer/nce_notes.pdf)。

虽然负采样对于学习词嵌入是有用的，但它并不能保证其导数趋向于softmax函数的梯度。相对来说NCE在噪声样本增加时就能提供这个保证。[Mnih and Teh(2012)](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf)表明25个噪声样本足以使其性能达到正规的softmax，且伴随45%的加速。因为这个原因，我们将会使用NCE。

注意例如负采样和NCE等基于采样的方法只在训练时有用，在预测时仍然需要计算完整的softmax以获得规范的概率。

## 数据集（Dataset）
text8是2006年3月3日英语维基百科的文本的前100 MB，我们使用的文本已经花费大量时间进行预处理过，因为在这门课中主要的学习目标是TensorFlow。我们可以在[这里](http://mattmahoney.net/dc/text8.zip)下载这个数据集，课程的GitHub中的word_utils.py能够下载和读取这个文本。

100MB的文本不足以训练好的词嵌入，但是足够看到一些有趣的联系。如果你用空格分隔这个文本可以获得17,005,207个标记，如果想获得更好的结果你应该使用fil9（维基百科的前\\(10^9\\)个字节），就像[Matt Mahoney的网站](https://cs.fit.edu/~mmahoney/compression/textdata.html)上描述的一样。

## 实现word2vec

### 第1阶段：装配计算图

- 建立数据集并用它生成样本

输入是中心词，输出是上下文词。我们创建一个最常用词字典，用这些词的索引输入模型从而替代输入词。比如我们的中心词是字典中的第1000个词就输入999。

每个样本输入是一个标量，所以BATCH_SIZE个样本输入的形状为[BATCH_SIZE]，BATCH_SIZE个样本的输出形状为[BATCH_SIZE,1]。

	dataset = tf.data.Dataset.from_generator(gen, 
	                            (tf.int32, tf.int32), 
	                            (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
	iterator = dataset.make_initializable_iterator()
	center_words, target_words = iterator.get_next()

- 定义权重（嵌入矩阵，embedding matrix）

每一行对应一个词向量，如果一个词被表示为一个大小为EMBED_SIZE的向量，那么嵌入矩阵的形状为[VOCAB_SIZE,EMBED_SIZE]。我们用随机的均匀分布初始化嵌入矩阵。

	embed_matrix = tf.get_variable('embed_matrix', 
	                                shape=[VOCAB_SIZE, EMBED_SIZE],
	                                initializer=tf.random_uniform_initializer())

- 计算图的正向传播

我们的目的是获得我们字典中词的向量表示（嵌入矩阵），记住嵌入矩阵的维度为VOCAB_SIZExEMBED_SIZE，每一行都对应一个词的向量表示。所以要获得batch中所有中心词的向量，只需要对嵌入矩阵相应行进行切片，TensorFlow提供了一个很方便的方法去做这个。

	tf.nn.embedding_lookup(
	    params,
	    ids,
	    partition_strategy='mod',
	    name=None,
	    validate_indices=True,
	    max_norm=None
	)

这个方法在涉及到和独热码的矩阵相乘时十分有用，因为它避免了我们在无论如何都会返回0的地方做一堆不必要的计算。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_embedding_lookup.png)

所以我们在获得输入的中心词的向量表示时用这个方法：

    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

- 定义损失函数

NCE很难用纯Python实现，TensorFlow已经为我们实现了：

tf.nn.nce_loss(
    weights,
    biases,
    labels,
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=False,
    partition_strategy='mod',
    name='nce_loss'
)

**注意函数已经实现了，但是第四个参数是输入（input），第三个参数是标签（label）。**这在有些时候带来了很多麻烦，但是TensorFlow还是一个正在成长的平台，现在还不是很完美。NCE损失的源代码可以在[这里](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py)找到。

为了计算NCE损失我们需要隐层中的`weights`和`biases`，它们在训练时被`optimizer`更新。在采样之后，最后的输出评分会被计算，这些计算会在`tf.nn.nce_loss`中完成。

	tf.matmul(embed, tf.transpose(nce_weight)) + nce_bias

	nce_weight = tf.get_variable('nce_weight', 
	       shape=[VOCAB_SIZE, EMBED_SIZE],
	       initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBED_SIZE ** 0.5)))
	nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))

之后我们定义损失loss：

	loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
						biases=nce_bias, 
						labels=target_words, 
						inputs=embed, 
						num_sampled=NUM_SAMPLED, 
						num_classes=VOCAB_SIZE))

- 定义optimizer

我们还是使用梯度下降：

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

### 第2阶段：执行运算
我们将会创建一个session来运行optimizer去最小化损失，然后为我们输出损失值。别忘了重新初始化你的迭代器！

	with tf.Session() as sess:
	        sess.run(iterator.initializer)
	        sess.run(tf.global_variables_initializer())
	
	        writer = tf.summary.FileWriter('graphs/word2vec_simple', sess.graph)
	
	        for index in range(NUM_TRAIN_STEPS):
	            try:
	                loss_batch, _ = sess.run([loss, optimizer])
	            except tf.errors.OutOfRangeError:
	                sess.run(iterator.initializer)
	        writer.close()

你可以在课程的GitHub上的[word2vec.py](https://github.com/cnscott/Stanford-CS20si/blob/master/examples/04_word2vec.py)中看到完整的模型。

## 接口：怎样构建你的TensorFlow模型
至今我们建立的所有的模型或多或少都有着相同的结构。

### 第1阶段：组装你的计算图
1. 导入数据（用tf.data或者placeholder）
2. 定义权重
3. 定义预测模型
4. 定义损失函数loss
5. 定义优化器optimizer

### 第2阶段：执行运算
1. 初始化所有的模型变量
2. 初始化迭代器/feed_dict
3. 执行预测模型
4. 计算损失loss
5. 调整参数最小化loss

下面的图片是训练循环的可视化表示，摘自TensorFlow for Machine Intelligence (Abrahams et al., 2016)。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_Training-loop.jpg)

**问题**: 怎样使我们的模型可以重用？ 

**提示**: 利于Python的面向对象功能。

**回答**: 将我们的模型写成一个类!

