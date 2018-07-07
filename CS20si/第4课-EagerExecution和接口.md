# 第4课:Eager Execution和接口
---
到现在为止我们在TensorFlow中实现了两个简单的模型：用出生率预测平均寿命的线性回归和MNIST上手写数字识别的逻辑回归。我们学习了TensorFlow程序的两个基本阶段：组装计算图和执行计算图。但是你如何能够直接使用Python用命令的方式执行TensorFlow运算呢？这可以大大降低调试我们TensorFlow模型的难度。

在这一课中，我们介绍Eager execution，用Eager模型重写我们的线性回归。

## Eager execution
Eager execution是一个支持GPU加速和自动微分的类Numpy数值计算库，而且是一个用于机器学习研究和实验的灵活平台。它 是从TensorFlow 1.5版本开始在`tf.contrib.eager`
中提供的。

- 动机
	- 今天的TensorFlow：构建计算图然后执行它
		- 这是声明式编程。它的好处是高效且易于转换到其它平台；缺点是不是Python风格的且难以调试。
	- 如果你可以直接执行运算呢？
		- Eager execution提供它：它是TensorFlow的命令前端。
- 关键优势：Eager execution
	- 和Python调试工具兼容
		- pdb.set_trace()让你心满意足。
	- 提供即时的错误报告。
	- 允许使用Python数据结构。
		- 例如使用结构化输入
	- 能让你用Python控制流进行使用和微分。
- 开启Eager execution只需要两行代码。

    	import tensorflow.contrib.eager as tfe
    	tfe.enable_eager_execution() # Call this at program start-up

使用Eager execution你就可以简单的在一个REPL（交互编程环境，Read-eval-print-loop）中执行你的代码，就像这样：
	
	x = [[2.]]  # No need for placeholders!
	m = tf.matmul(x, x)
	
	print(m)  # No sessions!
	# tf.Tensor([[4.]], shape=(1, 1), dtype=float32)

你不用再担心这些：

1. placeholder
2. session
3. control dependencies
4. lazy loading
5. {name，variable，op} scopes

声明式TensorFlow代码：

	x = tf.placeholder(tf.float32, shape=[1, 1])
	m = tf.matmul(x, x)
	
	print(m)
	# Tensor("MatMul:0", shape=(1, 1), dtype=float32)

	with tf.Session() as sess:
	  	m_out = sess.run(m, feed_dict={x: [[2.]]})
		print(m_out)
		# [[4.]]

变成了：

    x = [[2.]]  # No need for placeholders!
	m = tf.matmul(x, x)
	
	print(m)  # No sessions!
	# tf.Tensor([[4.]], shape=(1, 1), dtype=float32)

### 梯度
Eager execution已经内建自动微分功能。

Eager模式下：

- 每个运算都被记录
- 通过回放这些记录来计算梯度
	- 反向传播

例子：

    def square(x):
		return x ** 2
    
    	grad = tfe.gradients_function(square)
    
    	print(square(3.))# tf.Tensor(9., shape=(), dtype=float32)
    	print(grad(3.))  # [tf.Tensor(6., shape=(), dtype=float32))]


### 一个运算的集合

**TensorFlow = 运算内核 + 执行**

- 原来的计算图构建方式： 使用Session执行运算的集合
- Eager execution方式：用Python执行运算的集合

## Eager模式的Huber回归

 Huber回归的代码在[这里](https://github.com/cnscott/Stanford-CS20si/blob/master/examples/04_linreg_eager.py)查看。

	""" Starter code for a simple regression example using eager execution.
	Created by Akshay Agrawal (akshayka@cs.stanford.edu)
	CS20: "TensorFlow for Deep Learning Research"
	cs20.stanford.edu
	Lecture 04
	"""
	import time
	
	import tensorflow as tf
	import tensorflow.contrib.eager as tfe
	import matplotlib.pyplot as plt
	
	import utils
	
	DATA_FILE = 'data/birth_life_2010.txt'
	
	# In order to use eager execution, `tfe.enable_eager_execution()` must be
	# called at the very beginning of a TensorFlow program.
	tfe.enable_eager_execution()
	
	# Read the data into a dataset.
	data, n_samples = utils.read_birth_life_data(DATA_FILE)
	dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
	
	# Create variables.
	w = tfe.Variable(0.0)
	b = tfe.Variable(0.0)
	
	# Define the linear predictor.
	def prediction(x):
	  return x * w + b
	
	# Define loss functions of the form: L(y, y_predicted)
	def squared_loss(y, y_predicted):
	  return (y - y_predicted) ** 2
	
	def huber_loss(y, y_predicted, m=1.0):
	  """Huber loss."""
	  t = y - y_predicted
	  # Note that enabling eager execution lets you use Python control flow and
	  # specificy dynamic TensorFlow computations. Contrast this implementation
	  # to the graph-construction one found in `utils`, which uses `tf.cond`.
	  return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)
	
	def train(loss_fn):
	  """Train a regression model evaluated using `loss_fn`."""
	  print('Training; loss function: ' + loss_fn.__name__)
	  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
	
	  # Define the function through which to differentiate.
	  def loss_for_example(x, y):
	    return loss_fn(y, prediction(x))
	
	  # `grad_fn(x_i, y_i)` returns (1) the value of `loss_for_example`
	  # evaluated at `x_i`, `y_i` and (2) the gradients of any variables used in
	  # calculating it.
	  grad_fn = tfe.implicit_value_and_gradients(loss_for_example)
	
	  start = time.time()
	  for epoch in range(100):
	    total_loss = 0.0
	    for x_i, y_i in tfe.Iterator(dataset):
	      loss, gradients = grad_fn(x_i, y_i)
	      # Take an optimization step and update variables.
	      optimizer.apply_gradients(gradients)
	      total_loss += loss
	    if epoch % 10 == 0:
	      print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))
	  print('Took: %f seconds' % (time.time() - start))
	  print('Eager execution exhibits significant overhead per operation. '
	        'As you increase your batch size, the impact of the overhead will '
	        'become less noticeable. Eager execution is under active development: '
	        'expect performance to increase substantially in the near future!')
	
	train(huber_loss)
	plt.plot(data[:,0], data[:,1], 'bo')
	# The `.numpy()` method of a tensor retrieves the NumPy array backing it.
	# In future versions of eager, you won't need to call `.numpy()` and will
	# instead be able to, in most cases, pass Tensors wherever NumPy arrays are
	# expected.
	plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r',
	         label="huber regression")
	plt.legend()
	plt.show()