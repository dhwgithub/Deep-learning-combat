# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import cnn_learn.alexnet_learn.alexnet_inference as inference

####配置实现指数衰减学习率的相关参数######
BATCH_SIZE = 50  # 一个训练batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.01  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减系数

####配置实现正则化的相关参数######
REGULARAZTION_RATE = 0.0001  # 正则化项的权重

####配置实现滑动平均模型的相关参数######
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均模型的衰减率

#############训练迭代轮数#########
TRAINING_STEPS = 8000


def train(mnist):
    """
    定义输入输出placeholder，其中mnist_inference.INPUT_NODE为784，mnist_inference.OUTPUT_NODE为10。
    TensorFlow提供了tf.contrib.layers.l2_regularizer函数，它可以返回一个函数，
    这个函数可以计算一个给定参数的l2正则化项的值。类似的，
    tf.contrib.layers.l1_regularizer可以计算L1正则化项的值
    """
    # 调整输入数据placeholder的格式，输入为一个四维矩阵。第一维表示一个batch中样例的个数；
    # 第二维和第三维表示图片的尺寸；第四维表示图片的深度
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    inference.IMAGE_SIZE,
                                    inference.IMAGE_SIZE,
                                    inference.NUM_CHANNELS], name='x-input1')
    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)  # 返回一个可以计算l2正则化项的函数

    # 前向传播过程
    y = inference.inference(x, True, regularizer)

    # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量(trainable=False)。
    # 在使用tensorflow训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类，ExponentialMovingAverage还提供了num_updates参数
    # 来动态设置decay的大小，因此，通过给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如global_step）就不需要了。
    # tf.trainable_variables()返回的就是图上集合GraphKeys.TRAINABLE_VARIABLES中的元素。
    # 这个集合的元素就是所有没有指定trainable=False的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 因为交叉熵一般会与softmax回归一起使用，所以TensorFlow对这两个功能进行了统一封装，
    # 并提供了tf.nn.softmax_cross_entropy_with_logits函数。比如：
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits（y,y_），其中y代表了原始神经网络的输出结果，而y_代表标准答案。
    # 这样通过一个命令就可以得到使用了softmax回归之后的交叉熵。在只有一个正确答案的分类问题中，TensorFlow还提供了
    # tf.nn.sparse_softmax_cross_entropy_with_logits函数来进一步加速计算过程。
    # 注意，tf.argmax(vector, axis=1)，其中axis：0表示按列，1表示按行。返回的是vector中的最大值的索引号，
    # 如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，
    # 这个向量的每一个元素都是相对应矩阵行的最大值元素的索引号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # get_collection返回一个列表，这个列表包含所有这个losses集合中的元素，这些元素就是损失函数的不同部分，
    # 将它们加起来就可以得到最终的损失函数。
    # 其中tf.add_n([p1, p2, p3....])函数是实现一个列表的元素的相加。输入的对象是一个列表，列表里的元素可以是向量、矩阵等
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 通过exponential_decay函数生成学习率，使用呈指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)

    # 在minimize函数中传入global_step将自动更新global_step参数，从而使得学习率learning_rate也得到相应更新。
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，
    # 又要更新每一个参数的滑动平均值。为了一次完成多个操作，TensorFlow提供了tf.control_dependencies机制
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)

    with tf.Session() as sess:
        # 初始化所有变量
        sess.run(tf.global_variables_initializer())

        # operation to merge all summary
        merge_op = tf.summary.merge_all()
        # write to file
        writer = tf.summary.FileWriter('./logs', sess.graph)

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成。
        print("****************开始训练************************")
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 类似地将输入的训练数据格式调整为一个四维矩阵,并将这个调整后的数据传入sess.run过程
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          inference.IMAGE_SIZE,
                                          inference.IMAGE_SIZE,
                                          inference.NUM_CHANNELS))

            result, train_op_renew, loss_value, step = sess.run([merge_op, train_op, loss, global_step],
                                                                feed_dict={x: reshaped_xs, y_: ys})
            writer.add_summary(result, step)

            if i % 1000 == 0:
                # 每1000轮保存一次模型。
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。
                # 通过损失函数的大小可以大概了解训练的情况。
                # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                print("After %d training step (s) , loss on training batch is %g." % (step, loss_value))

                # 持久化一个简单的tensorflow模型。注意这里给出了global_step参数，这样可以让每个被
                # 保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000” 表示训练1000轮之后得到的模型。
                # 通过 saver.save函数将tensorflow模型保存到了model.ckpt文件中。
                # 每次保存操作会生成三个文件，这是因为tensorflow会将计算图的结构和图上参数取值分开保存。
                # 第一个文件为model.ckpt.meta，它保存了tensorflow计算图的结构；第二个文件为model.ckpt，这个文件中保存了
                # tensorflow程序中每一个变量的取值；最后一个文件为checkpoint文件，这个文件中保存了一个目录下所有的模型文件列表。
                saver.save(sess, r".\model\model.ckpt", global_step=global_step)

        print("*******************训练结束****************************")


def main(argv=None):
    """
    主程序入口
    声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    """
    data_path = r'E:\pycharm\tensorflow-learn\datasets\MNIST'
    mnist = input_data.read_data_sets(data_path, False, one_hot=True)
    if mnist == None:
        print("*************数据加载失败*****************")
    else:
        print("*************数据加载成功*****************")
        train(mnist)


# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()

'''
启动可视化界面命令：
tensorboard --logdir=E:\pycharm\tensorflow-learn\cnn_learn\alexnet_learn\logs
'''
'''
****************开始训练************************
After 1 training step (s) , loss on training batch is 62.8283.
After 1001 training step (s) , loss on training batch is 1.18844.
After 2001 training step (s) , loss on training batch is 0.913694.
After 3001 training step (s) , loss on training batch is 0.539359.
After 4001 training step (s) , loss on training batch is 0.40446.
After 5001 training step (s) , loss on training batch is 0.296183.
After 6001 training step (s) , loss on training batch is 0.471341.
After 7001 training step (s) , loss on training batch is 0.361297.
*******************训练结束****************************
'''