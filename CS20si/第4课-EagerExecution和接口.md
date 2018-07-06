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

更多的细节请看第4课的PPT。

