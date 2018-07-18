# 第7课: TensorFlow中的卷积

> [CS20si课程资料和代码Github地址](https://github.com/cnscott/Stanford-CS20si)

## 没有训练的卷积
你有可能已经熟悉数学或物理中的“卷积”，牛津词典对“卷积”在数学领域的定义如下：

> 一个由两个给定的函数集成的函数,表达一个函数的形状是如何被另一个函数修改的。

这就是卷积在机器学习中的含义，卷积是一个原始输入Input被一个核Kernel（或者叫滤波器Filter/特征图Feature Map）修改。更多的细节参照CS231n课程。

事实上，我们可以不用训练而直接使用卷积。比如我们可以用一个3x3的卷积来模糊一张图片。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_Conv-Blur.jpg)

在TensorFlow中去做卷积，我们有很多内建的层可以使用。你可以输入2维数据做1维卷积，输入3维数据做2维卷积，输入4维数据做3维卷积，最常用的是2维卷积。

	tf.nn.conv2d(
	    input,
	    filter,
	    strides,
	    padding,
	    use_cudnn_on_gpu=True,
	    data_format='NHWC',
	    dilations=[1, 1, 1, 1],
	    name=None
	)
	
	Input: Batch size (N) x Height (H) x Width (W) x Channels (C)
	Filter: Height x Width x Input Channels x Output Channels
	(e.g. [5, 5, 3, 64])
	Strides: 4 element 1-D tensor, strides in each direction
	(often [1, 1, 1, 1] or [1, 2, 2, 1])
	Padding: 'SAME' or 'VALID'
	Dilations: The dilation factor. If set to k > 1, there will be k-1 skipped cells between each filter element on that dimension.
	Data_format: default to NHWC

这儿还有一些其它的内建卷积：

	depthwise_conv2d: 单独处理每个channel的数据。
	separable_conv2d: 一个depthwise的空间滤波器后面跟一个逐点滤波器。

作为一个有趣的练习，你可以在课程的GitHub中的kernes.py文件中看到一些著名的核的值，在07_basic_kernels.py中看到它们的用法。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_Basic-Kernels.jpg)

在练习中，我们硬编码这些核的值，但是在训练一个CNN时，我们不知道核的最优值而是学习出它们。我们会用老朋友MNIST来做学习核的练习。

## 用CNN处理MNIST
我们曾经用含有一个全连接层的逻辑回归处理MNIST，结果是糟糕的。现在让我们看看用局部连接的CNN是否会好一些。

我们将会用两个步长为1的卷积层，每个跟随一个relu激活和最大池化层maxpool，最后加上两个全连接层。

![](http://images.cnblogs.com/cnblogs_com/tech0ne/1247403/o_CNN-MNIST.jpg)

### 卷积层

	def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
	    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
	        in_channels = inputs.shape[-1]
	        kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters], 
	                                initializer=tf.truncated_normal_initializer())
	        biases = tf.get_variable('biases', [filters],
	                            initializer=tf.random_normal_initializer())
	        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
	    return tf.nn.relu(conv + biases, name=scope.name)

### 池化层

	def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
	    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
	        pool = tf.nn.max_pool(inputs, 
	                            ksize=[1, ksize, ksize, 1], 
	                            strides=[1, stride, stride, 1],
	                            padding=padding)
	    return pool

### 全连接层

	def fully_connected(inputs, out_dim, scope_name='fc'):
	    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
	        in_dim = inputs.shape[-1]
	        w = tf.get_variable('weights', [in_dim, out_dim],
	                            initializer=tf.truncated_normal_initializer())
	        b = tf.get_variable('biases', [out_dim],
	                            initializer=tf.constant_initializer(0.0))
	        out = tf.matmul(inputs, w) + b
	    return out

### 放在一起
有了上面这些层，我们可以简单地建立我们的模型

	def inference(self):
	        conv1 = conv_relu(inputs=self.img,
	                        filters=32,
	                        k_size=5,
	                        stride=1,
	                        padding='SAME',
	                        scope_name='conv1')
	        pool1 = maxpool(conv1, 2, 2, 'VALID', 'pool1')
	        conv2 = conv_relu(inputs=pool1,
	                        filters=64,
	                        k_size=5,
	                        stride=1,
	                        padding='SAME',
	                        scope_name='conv2')
	        pool2 = maxpool(conv2, 2, 2, 'VALID', 'pool2')
	        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
	        pool2 = tf.reshape(pool2, [-1, feature_dim])
	        fc = tf.nn.relu(fully_connected(pool2, 1024, 'fc'))
	        dropout = tf.layers.dropout(fc, self.keep_prob, training=self.training, name='dropout')
	        
	        self.logits = fully_connected(dropout, self.n_classes, 'logits')

在训练时，需要评估每个epoch的准确率。

	def eval(self):
	        '''
	        Count the number of right predictions in a batch
	        '''
	        with tf.name_scope('predict'):
	            preds = tf.nn.softmax(self.logits)
	            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
	            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

可以在课程Github的07_convnet_mnist.py中查看完整代码。

> 译者注：这篇略过了很多和CS231n重复的东西，CS20si中的CNN相关课程主要看看Tensorflow的代码组织，CNN的知识还是推荐看[CS231n系列课程](https://mp.csdn.net/mdeditor/80964012)。