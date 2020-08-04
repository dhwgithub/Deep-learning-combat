# -*- coding: utf-8 -*-
# https://www.cnblogs.com/daremosiranaihana/p/11655343.html
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import cnn_learn.resnet50_learn.resnet50_inference as inference

####配置实现指数衰减学习率的相关参数######
BATCH_SIZE = 32  # 一个训练batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.001  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减系数

####配置实现正则化的相关参数######
REGULARAZTION_RATE = 0.0001  # 正则化项的权重

####配置实现滑动平均模型的相关参数######
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均模型的衰减率

#############训练迭代轮数#########
TRAINING_STEPS = 8000
SHOE_STEPS = 1000

def train(mnist):
    # 调整输入数据placeholder的格式，输入为一个四维矩阵。第一维表示一个batch中样例的个数；
    # 第二维和第三维表示图片的尺寸；第四维表示图片的深度
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    inference.IMAGE_SIZE,
                                    inference.IMAGE_SIZE,
                                    inference.NUM_CHANNELS], name='x-input1')
    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

    # regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)  # 返回一个可以计算l2正则化项的函数

    # 前向传播过程
    y = inference.inference(x, True)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    loss = cross_entropy_mean

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

            if i % SHOE_STEPS == 0:
                print("After %d training step (s) , loss on training batch is %g." % (step, loss_value))

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
tensorboard --logdir=E:\pycharm\tensorflow-learn\cnn_learn\resnet50_learn\logs
'''
'''

'''