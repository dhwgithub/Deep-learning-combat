# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data

import cnn_learn.make_my_dataset.alexnet_inference as inference
from cnn_learn.make_my_dataset import dog_and_cat_train

####配置实现指数衰减学习率的相关参数######
BATCH_SIZE = 64  # 一个训练batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.001  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减系数

####配置实现正则化的相关参数######
REGULARAZTION_RATE = 0.0001  # 正则化项的权重

####配置实现滑动平均模型的相关参数######
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均模型的衰减率

#############训练迭代轮数#########
TRAINING_STEPS = 8000

TRAIN_NUM = 263
TEST_NUM = 64

def train(img_batch, label_batch, img_test, label_test):
    # 调整输入数据placeholder的格式，输入为一个四维矩阵。第一维表示一个batch中样例的个数；
    # 第二维和第三维表示图片的尺寸；第四维表示图片的深度
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    inference.IMAGE_SIZE,
                                    inference.IMAGE_SIZE,
                                    inference.NUM_CHANNELS], name='x-input1')
    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)  # 返回一个可以计算l2正则化项的函数
    is_train = tf.placeholder(tf.string, name='is_train')

    # 前向传播过程
    y = inference.inference(x, is_train, regularizer)

    # ####################################  测  试   ####################################################
    # x_test = tf.placeholder(tf.float32, [TEST_NUM,
    #                                      inference.IMAGE_SIZE,
    #                                      inference.IMAGE_SIZE,
    #                                      inference.NUM_CHANNELS], name='x-input1')
    # y_test_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')
    # y_test = inference.inference(x_test, False, None)
    # y_test = inference.inference(x, False, None)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    # ####################################  测  试   ####################################################

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 通过exponential_decay函数生成学习率，使用呈指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               TRAIN_NUM / BATCH_SIZE,
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

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成。
        print("****************开始训练************************")
        for i in range(TRAINING_STEPS):
            xs, ys = sess.run([img_batch, label_batch])
            ys = dog_and_cat_train.one_hot(ys, inference.NUM_LABELS)

            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          inference.IMAGE_SIZE,
                                          inference.IMAGE_SIZE,
                                          inference.NUM_CHANNELS))

            result, train_op_renew, loss_value, step, acc = sess.run([merge_op, train_op, loss, global_step, accuracy],
                                                                feed_dict={x: reshaped_xs, y_: ys,
                                                                           is_train: 'True'})
            writer.add_summary(result, step)

            if i % 1000 == 0:
                print("After %d training step (s) , loss on training batch is %g, acc is %g" % (step, loss_value, acc))
                saver.save(sess, r".\model\model.ckpt", global_step=global_step)

        ########################################## 测 试  ################################
        print("*******************开始测试****************************")
        # tf.reset_default_graph()
        ckpt = tf.train.get_checkpoint_state(r".\model")
        if ckpt and ckpt.model_checkpoint_path:
            # 加载模型
            saver.restore(sess, ckpt.model_checkpoint_path)

            # operation to merge all summary
            merge_op = tf.summary.merge_all()
            # write to file
            writer = tf.summary.FileWriter('./logs', sess.graph)

            global_step = ckpt.model_checkpoint_path.split('-')[-1]

            xs, ys = sess.run([img_test, label_test])
            # xs, ys = sess.graph.get_tensor_by_name('test_dataset')  # 训练时运行过
            ys = dog_and_cat_train.one_hot(ys, inference.NUM_LABELS)

            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          inference.IMAGE_SIZE,
                                          inference.IMAGE_SIZE,
                                          inference.NUM_CHANNELS))

            result, accuracy_score = sess.run([merge_op, accuracy], feed_dict={x: reshaped_xs, y_: ys,
                                                                               is_train: 'False'})  # 使用此模型检验
            # 没有初始化滑动平均值，只是调用模型的值，inference只是提供了一个变量的接口，完全没有赋值
            print("After %s training steps, validation accuracy = %g" % (global_step, accuracy_score))

            writer.add_summary(result, global_step)
        else:
            print("No checkpoint file found")
            return
        ########################################### 测 试 ####################################

        coord.request_stop()
        coord.join(threads)
        print("*******************训练结束****************************")


def main(argv=None):
    train_tfrecords_path = r".\my_dataset\tf_files\dog_and_cat_train.tfrecords"
    test_tfrecords_path = r".\my_dataset\tf_files\dog_and_cat_test.tfrecords"

    img_batch, label_batch, img_test, label_test = dog_and_cat_train.get_datasets(train_tfrecords_path, test_tfrecords_path, BATCH_SIZE)
    train(img_batch, label_batch, img_test, label_test)


# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()

'''
启动可视化界面命令：
tensorboard --logdir=E:\pycharm\tensorflow-learn\cnn_learn\make_my_dataset\logs
'''
'''
****************开始训练************************
After 1 training step (s) , loss on training batch is 0.700601, acc is 0.609375
After 1001 training step (s) , loss on training batch is 0.693418, acc is 0.59375
After 2001 training step (s) , loss on training batch is 0.692839, acc is 0.59375
After 3001 training step (s) , loss on training batch is 0.672396, acc is 0.734375
After 4001 training step (s) , loss on training batch is 0.685288, acc is 0.65625
After 5001 training step (s) , loss on training batch is 0.689638, acc is 0.640625
After 6001 training step (s) , loss on training batch is 0.689434, acc is 0.625
After 7001 training step (s) , loss on training batch is 0.688619, acc is 0.609375
*******************开始测试****************************
After 7001 training steps, validation accuracy = 0.671875
*******************训练结束****************************
'''