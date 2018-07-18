# 第5课: Word2vec和实验管理(下)

> [CS20si课程资料和代码Github地址](https://github.com/cnscott/Stanford-CS20si)

## 管理实验
我们已经建立了一个word2vec模型，在使用一个小数据集时它看起来工作的很好。我们知道在一个大数据集上会消耗更长的时间，而且我们也知道训练更复杂的模型也会消耗大量的时间。例如，一个机器翻译模型可能需要几天时间，如果在一个单GPU上可能要几个月，很多计算机视觉和强化学习任务需要很长的时间和耐心。

让我们的模型跑几天然后做调整是很困难的，如果计算器或者计算集群崩溃了，训练中断，我们不得不重新跑！**所以能够在任何时间点停止训练并能恢复运行十分关键。**这有助于分析我们的模型，因为这允许我们在经过任意训练步骤后检查我们的模型。

另一个研究人员经常面临的问题是如何复制研究成果。在建立和训练神经网络时我们经常用随机化。例如我们为模型随机初始化weights，或者随机打乱训练样本。学习如何控制模型中的这个随机因素是很重要的。

在这一部分，我们将介绍TensorFlow提供的一组非常棒的工具来帮助我们管理实验，包括`tf.train.Saver()`类，TensorFlow的随机状态和可视化训练过程（用TensorBoard）。

### tf.train.Saver()
每隔一定的步骤或epoch周期性的保存模型参数，我们可以在这些节点上恢复/重新训练我们的模型。`tf.train.Saver()`类允许我们将计算图的变量保存到二进制文件中。

	tf.train.Saver.save(
	    sess,
	    save_path,
	    global_step=None,
	    latest_filename=None,
	    meta_graph_suffix='meta',
	    write_meta_graph=True,
	    write_state=True
	)

例如，如果我们每隔1000步保存一次计算图的变量：

	# define model
	
	# create a saver object
	saver = tf.train.Saver()
	
	# launch a session to execute the computation
	with tf.Session() as sess:
	    # actual training loop
	    for step in range(training_steps): 
		sess.run([optimizer])
		if (step + 1) % 1000 == 0:
		   saver.save(sess, 'checkpoint_directory/model_name', global_step=global_step)

在TensorFlow中，你保存计算图变量的那一步叫一个检查点（checkpoint）。因为我们会建立很多个检查点，在我们的模型中添加了一个名为`global_step`的变量有助于记录训练步骤。你会在很多TensorFlow程序中看到这个变量，我们首先会创建它并初始化为0，然后将它设置成不用被训练（因为我们不希望TensorFlow优化它）。

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

我能需要将`global_step`作为参数传递给optimizer，让它知道在每个训练步骤对`global_step`进行累加。

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)

要将变量值保存到checkpoints目录中，我们使用：

    saver.save(sess, 'checkpoints/model-name', global_step=global_step)

要恢复变量，我们用`tf.train.Saver.restore(sess, save_path)`，例如用第10000步的checkpoint进行恢复：

    saver.restore(sess, 'checkpoints/skip-gram-10000')

但是当然，我们只能在有checkpoint的时候才能加载变量，当没有时重新训练。TensorFlow允许我们使用`tf.train.get_checkpoint_state(‘directory-name/checkpoint’)`从一个文件夹读取checkpoint。

	ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
	if ckpt and ckpt.model_checkpoint_path:
	     saver.restore(sess, ckpt.model_checkpoint_path)

'checkpoint'文件会自动的跟踪时间最近的checkpoint，它的内容像这样：

	model_checkpoint_path: "skip-gram-21999"
	all_model_checkpoint_paths: "skip-gram-13999"
	all_model_checkpoint_paths: "skip-gram-15999"
	all_model_checkpoint_paths: "skip-gram-17999"
	all_model_checkpoint_paths: "skip-gram-19999"
	all_model_checkpoint_paths: "skip-gram-21999"

所以word2vec的训练循环像这样：

	saver = tf.train.Saver()
	
	initial_step = 0
	utils.safe_mkdir('checkpoints')
	
	with tf.Session() as sess:
	    sess.run(self.iterator.initializer)
	    sess.run(tf.global_variables_initializer())
	
	    # if a checkpoint exists, restore from the latest checkpoint
	    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
	    if ckpt and ckpt.model_checkpoint_path:
	        saver.restore(sess, ckpt.model_checkpoint_path)
	
	    writer = tf.summary.FileWriter('graphs/word2vec' + str(self.lr), sess.graph)
	
	    for index in range(num_train_steps):
	        try:
	            sess.run(self.optimizer)
	            # save the model every 1000 steps
	            if (index + 1) % 1000 == 0: 
	                saver.save(sess, 'checkpoints/skip-gram', index)
	        except tf.errors.OutOfRangeError:
	            sess.run(self.iterator.initializer)
	            
	    writer.close()

查看'checkpoints'目录，你会看到这些文件：

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_files-checkpoints.jpg)

默认情况下，`saver.save()`保存计算图的所有变量，这是TensorFlow推荐的。然而你也可以选择保存什么变量，在我们创建saver对象时将它们以list或dict传入。

	v1 = tf.Variable(..., name='v1') 
	v2 = tf.Variable(..., name='v2') 
	
	# pass the variables as a dict: 
	saver = tf.train.Saver({'v1': v1, 'v2': v2}) 
	
	# pass them as a list
	saver = tf.train.Saver([v1, v2]) 
	
	# passing a list is equivalent to passing a dict with the variable op names # as keys
	saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})

注意saver只能保存变量，不是整个计算图，所以我们仍然需要自己建立计算图，然后在加载变量。检查点指定从变量名映射到tensor的方式。

人们通常所做的不仅仅是保存最近一次迭代的参数，还保存到目前为止得到最佳结果的参数，以便您可以使用迄今为止的最佳参数对模型进行的评估。

### tf.summary
我们曾经用matplotlib可视化我们的loss和accuracy，这些都是不必要的，因为TensorBoard为我们提供了一组非常好的工具来在训练过程中可视化我们的summary统计。一些流行的统计量是loss、平均loss和accuracy，你可以将它们可视化为散点图、柱状图甚至图片，所以在我们的计算图中需要新的命名空间保存summary运算。

	def _create_summaries(self):
	     with tf.name_scope("summaries"):
	            tf.summary.scalar("loss", self.loss)
	            tf.summary.scalar("accuracy", self.accuracy)            
	            tf.summary.histogram("histogram loss", self.loss)
	            # because you have several summaries, we should merge them all
	            # into one op to make it easier to manage
	            self.summary_op = tf.summary.merge_all()

因为这是一个运算，所以你必须用`sess.run()`执行它。

	loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], 
	                                  feed_dict=feed_dict)

现在你已经得到了summary，还需要将summary用FileWriter对象写入文件中来进行可视化。

    writer.add_summary(summary, global_step=step)

现在查看TensorBoard，在Scalars页面你可以看到标量摘要图，这是你的loss的摘要图。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_Scalar-Loss.jpg)

loss柱状图

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_Histogram-Loss.png)

如果你将几个summaries存入graph目录的不同的子文件夹中，你可以比较它们。例如第一次用学习率为1.0跑我们的模型，保存为'graph/lr1.0'，然后第二次用学习率为0.5跑我们的模型，保存为'graph/lr0.5'，在Scalars页面的左上角我们可以切换查看这两次运行的结果来比较它们。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_Scalar-Two-Loss.jpg)

你也可以用`tf.summary.image`将统计结果作为图片可视化：

    tf.summary.image(name, tensor, max_outputs=3, collections=None)

### 控制随机性
直到我把它写下来，我才意识到什么是一个矛盾的词“控制随机化”，但是事实上你经常需要控制随机过程来在实验中得到稳定的结果。你可能对Numpy中的随机种子和随机状态很熟悉。TensorFlow允许你用两种方法从随机中获得稳定的结果。

1. 在计算层面设置随机种子。所有的随机tensor允许在初始化时传入随机种子。

	`var = tf.Variable(tf.truncated_normal((-1.0,1.0), stddev=0.1, seed=0))`

	注意Session是记录随机状态的东西，所以每个新的Session都会开始新的随机状态。

2. 使用tf.Graph.seed在计算图层面设置随机种子。

	`tf.set_random_seed(seed)`

	如果你不在乎计算图中每个运算的随机性，而只是想在另一个计算图中复制结果，你可以使用`tf.set_random_seed`。设置当前的TensorFlow随机种子只对当前的默认计算图有效。

### Autodiff（TensorFlow是怎样计算梯度的）
在目前我们所有建立的模型中，我们没有计算过一个梯度。我们所做的是建立一个前向传播模型，然后TensorFlow为我们做反向传播。

TensorFlow使用所谓的反向模式来自动微分，它允许你用与计算原始函数大致相同的代价对一个函数求导，通过在图中创建额外的节点和边来计算梯度。例如你需要计算C对I的梯度，首先TensorFlow查看两个节点间的路径，然后从C反向移动到I。对于这条路径上的每个运算，都有一个节点被添加到计算图中，通过链式法则组合每个新加入节点的局部梯度。这个过程在TensorFlow白皮书中显示为：

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_AutoDiff-Whitebook.jpg)

要计算局部梯度，我们可以使用`tf.gradients()`

    tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)


`tf.gradients(ys, [xs])`代表你想计算ys相对于[xs]中每个x的梯度。

	x = tf.Variable(2.0)
	y = 2.0 * (x ** 3)
	z = 3.0 + y ** 2
	
	grad_z = tf.gradients(z, [x, y])
	with tf.Session() as sess:
		sess.run(x.initializer)
		print sess.run(grad_z) # >> [768.0, 32.0]
	# 768 is the gradient of z with respect to x, 32 with respect to y

你可以手算下是否正确。

所以问题是：为什么我们还要学习如何计算梯度？为什么Chris Manning和Richard Socher还要我们计算cross entropy and softmax的梯度？用手算梯度会不会到某一天就像因为发明计算器而使用手算平方根一样过时吗?

也许。但是现在，TensorFlow可以为我们计算梯度，但它不能让我们直观地知道要使用什么函数。它不能告诉我们函数是否将会遭受梯度爆炸或梯度消失。我们仍然需要了解梯度以便理解为什么一个模型可以工作但是另一个不行。
