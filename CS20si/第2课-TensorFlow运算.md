# 第2课: TensorFlow运算

> [CS20si课程资料和代码Github地址](https://github.com/cnscott/Stanford-CS20si)

注意：运算的介绍可能比较枯燥，但是是基础，之后会更有趣。

## 1. TensorBoard
TensorFlow不仅仅是一个软件包，它是一个包括TensorFlow，TensorBoard和Tensor Serving的套件。为了充分利用TensorFlow，我们应该知道如何将它们结合起来使用，所以首先我们来了解TensorBoard。

Tensor是TensorFlow安装程序自带的图形可视化工具，用Google自己的话说：
> “你使用TensorFlow进行的例如训练深度神经网络的计算可能会非常复杂和难懂。为了更简单的理解，调试和优化TensorFlow程序，我们提供了一个名叫TensorBoard的可视化套件。”

TensorBoard配置好后，大概是这个样子：
![](https://tensorflow.google.cn/images/mnist_tensorboard.png)

当用户在一个激活了TensorBoard的TensorFlow程序上进行运算时，这些运算都会被导出到事件日志（event log）文件中。TensorBoard能够将这些事件日志可视化以便可以深入了解模型的计算图和它的运行时行为。越早和越经常的使用TensorBoard会使TensorFlow上的工作更有趣且更有成效。

接下来让我们编写第一个TensorFlow程序，然后使用TensorBoard可视化程序的计算图。为了使用TensorBoard进行可视化，我们需要写入程序的事件日志。

    import tensorflow as tf
    a = tf.constant(2)
    b = tf.constant(3)
    x = tf.add(a, b)
	writer = tf.summary.FileWriter([logdir], [graph])
    with tf.Session() as sess:
    	print(sess.run(x))
	writer.close()

[graph]是程序的计算图，你可以用`tf.get_default_graph()`获得程序默认的计算图，也可以用`sess.graph`获得Session处理的计算图。（当然你需要创建一个Session）不管怎样，确定在创建writer之前你已经定义了完整的计算图，不然TensorBoard可视化的结果将会不完整。
[logdir]是你希望存储事件日志的目录。

接下来，打开终端，先运行刚才的Tensorflow程序，再以刚才写入的日志目录作为参数运行TensorBoard。

	$ python3 [my_program.py] 
	$ tensorboard --logdir [logdir] --port [port]

最后打开浏览器，输入http://localhost:[port]/（端口号自己选择），你将会看到TensorBoard页面。点击Graph标签你可以查看计算图中有3个节点：2个常量运算和一个add运算。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_graph_add.jpg)

“Const”和“Const_1"代表a和b，节点”Add“对应x。我们可以在代码中给a，b和x命名让TensorBoard了解这些运算的名字。

	a = tf.constant(2, name="a")
	b = tf.constant(3, name="b")
	x = tf.add(a, b, name="add")

现在如果你再次运行程序和TensorBoard，你会看到：

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_graph_name_add.jpg)

计算图自己定义了运算和运算间依赖关系，只要简单的点击节点就可以查看值和节点类型。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_graph_node_add.jpg)

**Note**：如果你运行了程序很多次，在日志目录会有多个事件日志文件。TensorBoard只会显示最后一个计算图并且警告有多个日志文件，如果想避免警告就删除不需要的日志文件。

当然，TensorBoard能做的远远不止于可视化计算图，这里我们将介绍它最重要的一些功能。

## 2. 常量运算（Constant op）
创建常量运算很直接。

	tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
	# constant of 1d tensor (vector)
	a = tf.constant([2, 2], name="vector")
	# constant of 2x2 tensor (matrix)
	b = tf.constant([[0, 1], [2, 3]], name="matrix")

你可以用指定值初始化一个特定维度的tensor，就像Numpy一样。

	tf.zeros(shape, dtype=tf.float32, name=None)
	# create a tensor of shape and all elements are zeros
	tf.zeros([2, 3], tf.int32) ==> [[0, 0, 0], [0, 0, 0]]

---
	tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
	# create a tensor of shape and type (unless type is specified) as the input_tensor but all elements are zeros.
	# input_tensor [[0, 1], [2, 3], [4, 5]]
	tf.zeros_like(input_tensor) ==> [[0, 0], [0, 0], [0, 0]]

---
	tf.ones(shape, dtype=tf.float32, name=None)
	# create a tensor of shape and all elements are ones
	tf.ones([2, 3], tf.int32) ==> [[1, 1, 1], [1, 1, 1]]

---
	tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
	# create a tensor of shape and type (unless type is specified) as the input_tensor but all elements are ones.
	# input_tensor is [[0, 1], [2, 3], [4, 5]]
	tf.ones_like(input_tensor) ==> [[1, 1], [1, 1], [1, 1]]

---
	tf.fill(dims, value, name=None) 
	# create a tensor filled with a scalar value.
	tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]

你可以创建一个常量序列

	tf.lin_space(start, stop, num, name=None)
	# create a sequence of num evenly-spaced values are generated beginning at start. If num > 1, the values in the sequence increase by (stop - start) / (num - 1), so that the last one is exactly stop.
	# comparable to but slightly different from numpy.linspace
	tf.lin_space(10.0, 13.0, 4, name="linspace") ==> [10.0 11.0 12.0 13.0]

---
	tf.range([start], limit=None, delta=1, dtype=None, name='range')
	# create a sequence of numbers that begins at start and extends by increments of delta up to but not including limit
	# slight different from range in Python
	tf.range(3, 18, 3) ==> [3, 6, 9, 12, 15]
	tf.range(3, 1, -0.5) ==> [3, 2.5, 2, 1.5]
	tf.range(5) ==> [0, 1, 2, 3, 4]

需要注意的是和Numpy的序列不同，TensorFlow的序列是不能迭代的。

	for _ in np.linspace(0, 10, 4): # OK
	for _ in tf.linspace(0.0, 10.0, 4): # TypeError: 'Tensor' object is not iterable.

	for _ in range(4): # OK
	for _ in tf.range(4): # TypeError: 'Tensor' object is not iterable.
你也可以创建服从指定分布的随机常量。

	tf.random_normal
	tf.truncated_normal
	tf.random_uniform
	tf.random_shuffle
	tf.random_crop
	tf.multinomial
	tf.random_gamma
	tf.set_random_seed

## 3. 数学运算
TensorFlow的数学运算符很标准，你可以在[这里](https://tensorflow.google.cn/api_guides/python/math_ops)找到完整的列表。

- TensorFlow大量的除法运算

TensorFlow至少支持7种除法运算，做着或多或少一样的事情：`tf.div, tf.divide, tf.truediv, tf.floordiv, tf.realdiv, tf.truncateddiv, tf.floor_div`。创建这些运算的人一定非常喜欢除法。。。

	a = tf.constant([2, 2], name='a')
	b = tf.constant([[0, 1], [2, 3]], name='b')
	with tf.Session() as sess:
		print(sess.run(tf.div(b, a)))             ⇒ [[0 0] [1 1]]
		print(sess.run(tf.divide(b, a)))          ⇒ [[0. 0.5] [1. 1.5]]
		print(sess.run(tf.truediv(b, a)))         ⇒ [[0. 0.5] [1. 1.5]]
		print(sess.run(tf.floordiv(b, a)))        ⇒ [[0 0] [1 1]]
		print(sess.run(tf.realdiv(b, a)))         ⇒ # Error: only works for real values
		print(sess.run(tf.truncatediv(b, a)))     ⇒ [[0 0] [1 1]]
		print(sess.run(tf.floor_div(b, a)))       ⇒ [[0 0] [1 1]]

- tf.add_n

将多个tensor相加。

    tf.add_n([a, b, b])  => equivalent to a + b + b

- 点积

注意`tf.matmul`不是点积，而是使用`tf.tensordot`。

	a = tf.constant([10, 20], name='a')
	b = tf.constant([2, 3], name='b')
	with tf.Session() as sess:
		print(sess.run(tf.multiply(a, b)))           ⇒ [20 60] # element-wise multiplication
		print(sess.run(tf.tensordot(a, b, 1)))       ⇒ 80

下面是Python中的运算，摘自Fundamentals of DeepLearning。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_python_ops.jpg)

## 4. 数据类型
- Python原生类型

TensorFlow兼容Python的原生数据类型，例如：boolean，integer，float和string等。单独的值转换为0维tensor（标量，scalar），列表转换为1维tensor（向量，vector），列表的列表转换为2维tensor（矩阵，matrix），以此类推。

	t_0 = 19 # Treated as a 0-d tensor, or "scalar" 
	tf.zeros_like(t_0)                   # ==> 0
	tf.ones_like(t_0)                    # ==> 1

	t_1 = [b"apple", b"peach", b"grape"] # treated as a 1-d tensor, or "vector"
	tf.zeros_like(t_1)                   # ==> [b'' b'' b'']
	tf.ones_like(t_1)                    # ==> TypeError
	
	t_2 = [[True, False, False],
	       [False, False, True],
	       [False, True, False]]         # treated as a 2-d tensor, or "matrix"
	
	tf.zeros_like(t_2)                   # ==> 3x3 tensor, all elements are False
	tf.ones_like(t_2)                    # ==> 3x3 tensor, all elements are True

- TensorFlow原生类型

TensorFlow也有自己的原生类型：`tf.int32, tf.float32`,还有更令人兴奋的类型：`tf.bfloat, tf.complex, tf.quint`，完整的类型列表在[这里](https://tensorflow.google.cn/api_docs/python/tf/DType)。

- Numpy数据类型

到此为止，你可能已经注意到Numpy和TensorFlow数据类型的相似性。**TensorFlow被设计为和Numpy无缝集成**，这个软件包已经成为数据科学的通用语。

TensorFlow的数据类型是基于Numpy的，实际上`np.int32 == tf.int32`返回`True`。你可以将Numpy类型传给TensorFlow函数。

	tf.ones([2, 2], np.float32) ==> [[1.0 1.0], [1.0 1.0]]

## 5. 变量
- 创建变量

使用`tf.Variable`创建变量，应该注意的是`tf.constant`是小写的而`tf.Variable`是大写的，这是因为**`tf.constant`是一个运算而`tf.Variable`是一个含有多个运算的类**。

	x = tf.Variable(...) 
	x.initializer # init 
	x.value() # read op 
	x.assign(...) # write op 
	x.assign_add(...) 
	# and more

传统的创建变量方式为

	tf.Variable(<initial-value>, name=<optional-name>)
	s = tf.Variable(2, name="scalar") 
	m = tf.Variable([[0, 1], [2, 3]], name="matrix") 
	W = tf.Variable(tf.zeros([784,10]))

而**TensorFlow推荐使用`tf.get_variable`来创建变量，这样有利于变量的共享**。

	tf.get_variable(
	    name,
	    shape=None,
	    dtype=None,
	    initializer=None,
	    regularizer=None,
	    trainable=True,
	    collections=None,
	    caching_device=None,
	    partitioner=None,
	    validate_shape=True,
	    use_resource=None,
	    custom_getter=None,
	    constraint=None
	)

---
	s = tf.get_variable("scalar", initializer=tf.constant(2)) 
	m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
	W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())

- 初始化变量

你必须在使用变量之前初始化它们，否则将会报`FailedPreconditionError: Attempting to use uninitialized value`的错误。想查看所有没初始化的变量，你可以将它们打印出来：

	print(session.run(tf.report_uninitialized_variables()))

简单的将所有变量一次性初始化的方法为：

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

在这种情况下你用`tf.Session.run()`获得的是一个initializer而不是之前的tensor运算。

如果只初始化一部分变量，你可以使用`tf.variables_initializer()`：

	with tf.Session() as sess:
		sess.run(tf.variables_initializer([a, b]))

你也可以用`tf.Variable.initializer()`一个个的初始化变量：

	with tf.Session() as sess:
		sess.run(W.initializer)

还有一种初始化变量方式是从一个文件读取，我们会在后面的课程提到。

- 计算（Evaluate）变量的值

和tensor类似，可以用session获取变量的值。

	# V is a 784 x 10 variable of random values
	V = tf.get_variable("normal_matrix", shape=(784, 10), 
	                     initializer=tf.truncated_normal_initializer())
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(V))

也可以通过`tf.Variable.eval()`来获取变量的值。

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(V.eval())

- 给变量赋值

我们可以通过`tf.Variable.assign()`来给变量赋值。

    W = tf.Variable(10)
    W.assign(100)
    with tf.Session() as sess:
    	sess.run(W.initializer)
    	print(W.eval()) # >> 10

为什么输出的10而不是100？`W.assign(100)`并没有给W赋值而是创建了一个assign运算，要想使这个运算生效我们可以在session中运行这个运算。

	W = tf.Variable(10)

	assign_op = W.assign(100)
	with tf.Session() as sess:
		sess.run(assign_op)
		print(W.eval()) # >> 100

注意这次我们没有初始化W，因为assign为我们做了。**实际上initializer就是一个assign运算，它用初始值来初始化变量。**

	# in the source code
	self._initializer_op = state_ops.assign(self._variable, self._initial_value, validate_shape=validate_shape).op

为了简化变量的加减运算，TensorFlow提供了`tf.Variable.assign_add()`和`tf.Variable.assign_sub()`方法。不同于`tf.Variable.assign()`，这两个方法不会初始化变量，因为它们依赖变量的初始值。

	W = tf.Variable(10)
	
	with tf.Session() as sess:
		sess.run(W.initializer)
		print(sess.run(W.assign_add(10))) # >> 20
		print(sess.run(W.assign_sub(2)))  # >> 18

因为TensorFlow的Session们维护着各自的值，**每个Session拥有变量属于Session自己的当前值**。

	W = tf.Variable(10)
	sess1 = tf.Session()
	sess2 = tf.Session()
	sess1.run(W.initializer)
	sess2.run(W.initializer)
	print(sess1.run(W.assign_add(10)))		# >> 20
	print(sess2.run(W.assign_sub(2)))		# >> 8
	print(sess1.run(W.assign_add(100)))		# >> 120
	print(sess2.run(W.assign_sub(50)))		# >> -42
	sess1.close()
	sess2.close()

当你有一个依赖其它变量的变量时，假设你声明了`U = W * 2`

	# W is a random 700 x 10 tensor
	W = tf.Variable(tf.truncated_normal([700, 10]))
	U = tf.Variable(W * 2)

在这种情况下，**你应该使用`initialized_value()`方法去保证在使用W的值去初始化U之前W已经被初始化**。

	U = tf.Variable(W.initialized_value() * 2)

## 6. 交互Session（Interactive Session）
你有时候会看到`InteractiveSession`代替`Session`，它们之间**唯一的不同是`InteractiveSession`会把自己设置为默认的Session**，这样你就可以直接调用`run()`和`eval()`方法。这样方便了在Shell和IPython Notebook中使用，但是当有多个Session时使问题变得复杂。

	sess = tf.InteractiveSession()
	a = tf.constant(5.0)
	b = tf.constant(6.0)
	c = a * b
	print(c.eval()) # we can use 'c.eval()' without explicitly stating a session
	sess.close()

`tf.get_default_session()`方法返回当前线程的默认Session。

## 7. 控制依赖关系（Control Dependencies）
有时候，我们拥有两个或者更多独立的运算，而我们希望指定哪些运算应该先运行。这个情况下，我们可以使用`tf.Graph.control_dependencies([control_inputs])`。

	# your graph g have 5 ops: a, b, c, d, e
	with g.control_dependencies([a, b, c]):
		# `d` and `e` will only run after `a`, `b`, and `c` have executed.
		d = ...
		e = …

## 8. 导入数据
### 8.1 传统的方法：`placehoder`和`feed_dict`

TensorFlow 程序一般由两个阶段：

1. 组装一个计算图
2. 用Session在计算图中执行运算和评估变量

我们可以在不管计算所需的数值的情况下组装计算图，这个在不知道输入数据的情况下定义函数是一样的。

在计算图组装完成后，我们可以用`placeholder`将自己的数据灌入计算图中：

    tf.placeholder(dtype, shape=None, name=None)
	a = tf.placeholder(tf.float32, shape=[3]) # a is placeholder for a vector of 3 elements
	b = tf.constant([5, 5, 5], tf.float32)
	c = a + b # use the placeholder as you would any tensor
	with tf.Session() as sess:
		print(sess.run(c)) 

当我们尝试通过Session计算`c`的值时，我们将会获得一个错误，因为我们需要获得`a`的值。我们可以通过`feed_dict`来向`placeholder`灌数据，它是一个字典。

	with tf.Session() as sess:
		# compute the value of c given the value of a is [1, 2, 3]
		print(sess.run(c, {a: [1, 2, 3]})) 		# [6. 7. 8.]

这时我们再查看TensorBoard，计算图如下：

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_graph_placeholder_add.jpg)

我们可以向`placeholder`中多次灌入不同的值。

	with tf.Session() as sess:
		for a_value in list_of_a_values:
			print(sess.run(c, {a: a_value}))

你也可以向不是`placeholder`的tensor灌入数值，**所有的tensor都是可以灌值的**，**可以用`tf.Graph.is_feedable(tensor)`方法查看一个tensor是否是可以灌值的**。

	a = tf.add(2, 5)
	b = tf.multiply(a, 3)
	
	with tf.Session() as sess:
		print(sess.run(b)) 						# >> 21
		# compute the value of b given the value of a is 15
		print(sess.run(b, feed_dict={a: 15})) 			# >> 45

`feed_dict`对测试你的模型非常有用，当你有一个很大的计算图但只想测试其中某一块时，你可以灌入假值来避免在不必要的运算上浪费时间。

### 8.2 新的方法：`tf.data`

这个方法需要配合例子来讲，所以我们会在下一课线性和逻辑回归中涉及。

## 9.lazy loading的陷阱
现在TensorFlow中最常见的不是bug的bug叫做“lazy loading”。Lazy loading是一种设计模式，即在你要使用一个对象时才初始化对象。在TensorFlow的场景中，它的含义是在要执行一个运算时才创建这个运算。例如：

	x = tf.Variable(10, name='x')
	y = tf.Variable(20, name='y')
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('graphs/lazy_loading', sess.graph)
		for _ in range(10):
			sess.run(tf.add(x, y))
		print(tf.get_default_graph().as_graph_def()) 
		writer.close()

现在我们看看TensorBoard：

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_graph_lazy_loading.jpg)

打印计算图的定义：

    print(tf.get_default_graph().as_graph_def())

得到如下结果：

	node {
	  name: "Add_1"
	  op: "Add"
	  input: "x_1/read"
	  input: "y_1/read"
	  attr {
	    key: "T"
	    value {
	      type: DT_INT32
	    }
	  }
	}
	…
	…
	…
	node {
	  name: "Add_10"
	  op: "Add"
	  ...
	}

你可能回想：“这很愚蠢，为什么我要在一个相同的值上计算两次？”，然后认为这是一个bug。这种情况会经常发生，比如在你想在训练集的每个batch上计算损失函数或做预测时，**如果你不注意，可能会添加巨量的无用运算。**

### **Note**：**在翻译这篇文章时，译者用TensorFlow 1.8版本做了实验，这个bug应该没有了。**