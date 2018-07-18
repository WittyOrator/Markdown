# 第5课: Word2vec和实验管理(上)

> [CS20si课程资料和代码Github地址](https://github.com/cnscott/Stanford-CS20si)

<!-- TOC -->

- [第5课: Word2vec和实验管理(上)](#第5课-word2vec和实验管理上)
    - [Word2vec](#word2vec)
    - [Softmax, 负采样(Negative Sampling)和NCE(Noise Contrastive Estimation)](#softmax-负采样negative-sampling和ncenoise-contrastive-estimation)
    - [数据集(Dataset)](#数据集dataset)
    - [实现word2vec](#实现word2vec)
        - [第1阶段: 装配计算图](#第1阶段-装配计算图)
        - [第2阶段: 执行运算](#第2阶段-执行运算)
    - [接口: 怎样构建你的TensorFlow模型](#接口-怎样构建你的tensorflow模型)
        - [第1阶段: 组装你的计算图](#第1阶段-组装你的计算图)
        - [第2阶段: 执行运算](#第2阶段-执行运算-1)
    - [可视化词嵌入](#可视化词嵌入)
    - [变量共享](#变量共享)
        - [命名空间(Name Scope)](#命名空间name-scope)
        - [变量空间(Variable scope)](#变量空间variable-scope)
        - [计算图集合(Graph collections)](#计算图集合graph-collections)

<!-- /TOC -->

我们已经建立了几个非常简单的模型，它们只需要几分钟就能训练完毕。如果要训练更复杂的模型，我们需要一些更多的工具。在这节课中，我们将介绍模型库、变量共享、模型共享以及如何管理你的实验。我们将会用word2vec作为例子演示这些。

## Word2vec
你也许还不了解词嵌入（word embedding），那么你应该看看[Stanford CS 224N的词向量课程](http://web.stanford.edu/class/cs224n/lectures/lecture2.pdf)。了解之后，跟一下这两篇论文是一个好主意：

- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)(Mikolov et al., 2013)
- ,
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

## Softmax, 负采样(Negative Sampling)和NCE(Noise Contrastive Estimation)

获得可能的邻近词的分布，在理论上，我们经常用softmax。Softmax将一组随机值$x_i$映射成一组和为1的概率值$p_i$。在这种情况下，$softmax(x_i)$表示$x_i$是指定的中心词的邻近词的概率。

$$softmax(x_i) = exp(x_i) / ∑_i exp(x_i)$$

然而，分母的标准形式需要我们计算字典中所有词（可能有几百万个）的exp并求和。就算去掉不常用的词，一个自然语言模型也必须考虑至少成千上万个最常用词，标准形式的softmax还是不好计算。

这里有两个主要的方法可以规避这个瓶颈：分层softmax（hierarchical softmax）和基于采样的softmax（sample-based softmax）。Mikolov团队在他们的论文中展示了负采样加速了skip-gram模型的训练，并对比了更复杂的分层softmax。

负采样顾名思义属于基于采样的方法族，这个方法族还包括重要性采样（importance sampling）和目标采样（target sampling）。负采样是一种叫做Noise Contrastive Estimation（NCE）方法的简化模型。负采样对产生的噪声样本数量$k$和噪声样本分布$Q$作了一定的假设，例如$kQ(w)=1$，这样可以简化计算。更多的细节可以看Sebastian Rudder的[On word embeddings - Part 2: Approximating the Softmax](http://ruder.io/word-embeddings-softmax/)和Chris Dyer的[Notes on Noise Contrastive Estimation and Negative Sampling](http://demo.clab.cs.cmu.edu/cdyer/nce_notes.pdf)。

虽然负采样对于学习词嵌入是有用的，但它并不能保证其导数趋向于softmax函数的梯度。相对来说NCE在噪声样本增加时就能提供这个保证。[Mnih and Teh(2012)](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf)表明25个噪声样本足以使其性能达到正规的softmax，且伴随45%的加速。因为这个原因，我们将会使用NCE。

注意例如负采样和NCE等基于采样的方法只在训练时有用，在预测时仍然需要计算完整的softmax以获得规范的概率。

## 数据集(Dataset)
text8是2006年3月3日英语维基百科的文本的前100 MB，我们使用的文本已经花费大量时间进行预处理过，因为在这门课中主要的学习目标是TensorFlow。我们可以在[这里](http://mattmahoney.net/dc/text8.zip)下载这个数据集，课程的GitHub中的word_utils.py能够下载和读取这个文本。

100MB的文本不足以训练好的词嵌入，但是足够看到一些有趣的联系。如果你用空格分隔这个文本可以获得17,005,207个标记，如果想获得更好的结果你应该使用fil9（维基百科的前$10^9$个字节），就像[Matt Mahoney的网站](https://cs.fit.edu/~mmahoney/compression/textdata.html)上描述的一样。

## 实现word2vec

### 第1阶段: 装配计算图

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

- 预测（计算图的正向传播）

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

### 第2阶段: 执行运算
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

## 接口: 怎样构建你的TensorFlow模型
至今我们建立的所有的模型或多或少都有着相同的结构。

### 第1阶段: 组装你的计算图
1. 导入数据（用tf.data或者placeholder）
2. 定义权重
3. 定义预测模型
4. 定义损失函数loss
5. 定义优化器optimizer

### 第2阶段: 执行运算
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

我们的模型类应该实现下面的接口，我们合并了第3步和第4步是因为我们想把embed放到命名空间“NCE loss”下。

	class SkipGramModel:
	    """ Build the graph for word2vec model """
	    def __init__(self, params):
	        pass
	
	    def _import_data(self):
	        """ Step 1: import data """
	        pass
	
	    def _create_embedding(self):
	        """ Step 2: in word2vec, it's actually the weights that we care about """
	        pass
	
	    def _create_loss(self):
	        """ Step 3 + 4: define the inference + the loss function """
	        pass
	
	    def _create_optimizer(self):
	        """ Step 5: define optimizer """
	        pass

## 可视化词嵌入

> **t-SNE**(维基百科)

> t-SNE（t-distributed stochastic neighbor embedding，t-分布随机近邻嵌入）是一种Geoffrey Hinton等人发明的用于降维的机器学习算法。他是一种非线性降维技术，特别适合在将高维数据嵌入的二维或三维空间中，然后放到散点图中进行可视化。具体地说，它将每一个高维对象建模为一个二维或三维点，其方式是相似的对象建模为邻近的点，而不相似的对象建模为较远的点。

> t-SNE算法包含两个主要步骤：
> 
> 1. 首先对成对的高维对象构建一个概率分布，相似的对象拥有高概率被选中，不同的对象拥有极地的概率被选中。
> 
> 2. t-SNE在低维映射中对点定义了相似的概率分布，然后相对于映射中个点的位置最小化两个分布之间的Kullback-Leibler散度。
> 
> 注意，虽然原始算法使用对象之间的欧几里得距离作为其相似性度量的基础，但这应该根据需要进行修改。

你可以用它来可视化词嵌入，你可以可视化任何东西的任何向量表示！在Olah的博客中可以看到[可视化MNIST](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)的例子(需要科学上网)。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_t-SNE-MNIST.jpg)

我们也可以使用PCA来可视化词嵌入。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_skip-gram-PCA.jpg)

而且我们用TensorFlow projector和TensorBoard只用不到10行代码就可以做所有这些可视化。这些可视化文件会被存储在visualization目录中，在命令行运行`tensorboard --logdir visualization`进行查看。

	from tensorflow.contrib.tensorboard.plugins import projector
	
	def visualize(self, visual_fld, num_visualize):
	        # create the list of num_variable most common words to visualize
	        word2vec_utils.most_common_words(visual_fld, num_visualize)
	
	        saver = tf.train.Saver()
	        with tf.Session() as sess:
	            sess.run(tf.global_variables_initializer())
	            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
	
	            # if that checkpoint exists, restore from checkpoint
	            if ckpt and ckpt.model_checkpoint_path:
	                saver.restore(sess, ckpt.model_checkpoint_path)
	
	            final_embed_matrix = sess.run(self.embed_matrix)
	            
	            # you have to store embeddings in a new variable
	            embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embeded')
	            sess.run(embedding_var.initializer)
	
	            config = projector.ProjectorConfig()
	            summary_writer = tf.summary.FileWriter(visual_fld)
	
	            # add embedding to the config file
	            embedding = config.embeddings.add()
	            embedding.tensor_name = embedding_var.name
	            
	            # link this tensor to the file with the first NUM_VISUALIZE words of vocab
	            embedding.metadata_path = os.path.join(visual_fld,[file_of_most_common_words])
	
	            # saves a configuration file that TensorBoard will read during startup.
	            projector.visualize_embeddings(summary_writer, config)
	            saver_embed = tf.train.Saver([embedding_var])
	            saver_embed.save(sess, os.path.join(visual_fld, 'model.ckpt'), 1)

请到课程GitHub的examples/04_word2vec_visualize.py中查看完整代码。

## 变量共享
### 命名空间(Name Scope)
让我们给tensors命名然后看看在TensorBoard中我们的word2vec模型长什么样。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_NameScope-Word2vec.jpg)

就像你在图中看到的，节点散落的到处都是，使图非常难读。TensorFlow并不知道哪些节点应该分到一组，当您构建具有数百个运算的复杂模型时，这可能会使调试你的计算图变得十分困难。

TensorFlow使用命名空间（Name Scope）来将运算节点分组：

	with tf.name_scope(name_of_that_scope):
		# declare op_1
		# declare op_2
		# ...

比如你的计算图有4个命名空间：“data”、“embed”、“loss”和“optimizer”

	with tf.name_scope('data'):
	    iterator = dataset.make_initializable_iterator()
	    center_words, target_words = iterator.get_next()
	
	with tf.name_scope('embed'):
	    embed_matrix = tf.get_variable('embed_matrix', 
	                                    shape=[VOCAB_SIZE, EMBED_SIZE],
	                                    initializer=tf.random_uniform_initializer())
	    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embedding')
	
	with tf.name_scope('loss'):
	    nce_weight = tf.get_variable('nce_weight', shape=[VOCAB_SIZE, EMBED_SIZE],
	                                initializer=tf.truncated_normal_initializer())
	    nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCAB_SIZE]))
	
	    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
	                                        biases=nce_bias, 
	                                        labels=target_words, 
	                                        inputs=embed, 
	                                        num_sampled=NUM_SAMPLED, 
	                                        num_classes=VOCAB_SIZE), name='loss')
	
	with tf.name_scope('optimizer'):
	    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

在TensorBoard中查看计算图时，你会看到整洁的分组：

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_NameScope-Grouped.jpg)

你可以双击每个命名空间块展开查看内部的运算。

TensorBoard有三种类型的边：

- 灰色实线箭头，表示数据流 - 比如tf.add(x,y)
- 橙色实线箭头，表示哪个运算可以改变哪个运算 - 比如optimizer在BP中改变nce_weight、nce_bias和embed_matrix。
- 虚线箭头.表示控制依赖 - 比如 nce_weight只能在init之后被执行。控制依赖还可以用tf.Graph.control_dependencies(control_inputs)声明。

### 变量空间(Variable scope)
一个人们常问的问题是：“命名空间和变量空间有什么不同？”。它们全都是创建命名空间，而变量空间做的是有利于参数共享。让我们看看为什么我们需要变量共享。

假设我们需要创建一个两个隐层的神经网络，然后我们用两个不同的输入x1和x2去调用这个神经网络。

	x1 = tf.truncated_normal([200, 100], name='x1')
	x2 = tf.truncated_normal([200, 100], name='x2')
	
	def two_hidden_layers(x):
	    assert x.shape.as_list() == [200, 100]
	    w1 = tf.Variable(tf.random_normal([100, 50]), name="h1_weights")
	    b1 = tf.Variable(tf.zeros([50]), name="h1_biases")
	    h1 = tf.matmul(x, w1) + b1
	    assert h1.shape.as_list() == [200, 50]  
	    w2 = tf.Variable(tf.random_normal([50, 10]), name="h2_weights")
	    b2 = tf.Variable(tf.zeros([10]), name="h2_biases")
	    logits = tf.matmul(h1, w2) + b2
	    return logits
	
	logits1 = two_hidden_layers(x1)
	logits2 = two_hidden_layers(x2)

查看TensorBoard中的计算图：

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_Graph-TwoLayers.jpg)

每次你调用两个网络时，TensorFlow都会创建两组变量，而事实上，你想要网络为所有的输入共享相同的变量。要做这个，你首先需要用tf.get_variable()创建变量。当我们用tf.get_variable()创建变量时，它会先检查这个变量是否存在，如果存在就使用它，否则创建一个新的变量。

	def two_hidden_layers_2(x):
	    assert x.shape.as_list() == [200, 100]
	    w1 = tf.get_variable("h1_weights", [100, 50], initializer=tf.random_normal_initializer())
	    b1 = tf.get_variable("h1_biases", [50], initializer=tf.constant_initializer(0.0))
	    h1 = tf.matmul(x, w1) + b1
	    assert h1.shape.as_list() == [200, 50]  
	    w2 = tf.get_variable("h2_weights", [50, 10], initializer=tf.random_normal_initializer())
	    b2 = tf.get_variable("h2_biases", [10], initializer=tf.constant_initializer(0.0))
	    logits = tf.matmul(h1, w2) + b2
	    return logits

我们运行会得到下列错误：

    ValueError: Variable h1_weights already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?

要避免错误，我们需要将我们要用的所有变量放到变量空间中，然后设置变量空间为可重用的（reusable）。
	
	def fully_connected(x, output_dim, scope):
	    with tf.variable_scope(scope) as scope:
	        w = tf.get_variable("weights", [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
	        b = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(0.0))
	        return tf.matmul(x, w) + b
	
	def two_hidden_layers(x):
	    h1 = fully_connected(x, 50, 'h1')
	    h2 = fully_connected(h1, 10, 'h2')
	
	with tf.variable_scope('two_layers') as scope:
	    logits1 = two_hidden_layers(x1)
	    scope.reuse_variables() # 设置重用变量
	    logits2 = two_hidden_layers(x2)

让我们看看TensorBoard：

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_Graph-TwoLayers-VariableShare.jpg)

现在只有一组变量了，都在变量空间呢`two_layers`中，它们接受了两个不同的输入x1和x2。`tf.variable_scope("name")`隐式的打开了`tf.name_scope("name")`。

### 计算图集合(Graph collections)
当你创建模型时，你可能想将你们的变量放在计算图的不同部分中，有时你想要一种简单的方法存取它们。`tf.get_collection`使你能够使用集合的名字作为关键字存取特定的变量集合，空间是变量空间。

	tf.get_collection(
	    key,
	    scope=None
	)

默认情况下，所有的变量都被放在集合`tf.GraphKeys.GLOBAL_VARIABLES`中，要获取变量空间“my_scope”中的所有的变量，只需要简单的调用：

    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='my_scope')

如果你在建立变量时设置`trainable=True`（这是默认值），那么这个变量将会被放在集合`tf.GraphKeys.TRAINABLE_VARIABLES`中。

你可以创建不包含变量的运算的集合，你可以使用`tf.add_to_collection(name,value)`来创建你自己的集合，例如你可以创建一个initializer的集合然后把所有的init运算都放在里面。

标准库使用各种众所周知的名称来收集和检索与计算图相关的值。 `tf.train.Optimizer`的子类默认优化变量集合`tf.GraphKeys.TRAINABLE_VARIABLES`中的变量，但是也可以显示设置需要优化的变量列表。