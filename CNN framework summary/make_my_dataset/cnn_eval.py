# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data

import cnn_learn.make_my_dataset.alexnet_inference as inference
import cnn_learn.make_my_dataset.cnn_train as cnn_train
from cnn_learn.make_my_dataset import dog_and_cat_train

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 0

TEST_NUM = 64

def evaluate(img_test, label_test):
    with tf.Graph().as_default() as g:  # 将默认图设为g
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [TEST_NUM,
                                        inference.IMAGE_SIZE,
                                        inference.IMAGE_SIZE,
                                        inference.NUM_CHANNELS], name='x-input1')
        y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

        # xs = mnist.validation.images
        # # print(xs.shape)  # (5000, 784)
        # # 类似地将输入的测试数据格式调整为一个四维矩阵
        # reshaped_xs = np.reshape(xs, (mnist.validation.images.shape[0],
        #                               my_lenet5.IMAGE_SIZE,
        #                               my_lenet5.IMAGE_SIZE,
        #                               my_lenet5.NUM_CHANNELS))
        # validate_feed = {x: reshaped_xs, y_: mnist.validation.labels}

        # 直接通过调用封装好的函数来计算前向传播的结果
        # 测试时不关注过拟合问题，所以正则化输入为None
        y = inference.inference(x, False, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型
        variable_averages = tf.train.ExponentialMovingAverage(cnn_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)  # 这些值要从模型中提取

        # Create a summary to monitor cost tensor
        tf.summary.scalar("accuracy", accuracy)

        # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        for i in range(1):  # 为了降低个人电脑的压力，此处只利用最后生成的模型对测试数据集做测试
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                ckpt = tf.train.get_checkpoint_state(r".\model")
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    # operation to merge all summary
                    merge_op = tf.summary.merge_all()
                    # write to file
                    writer = tf.summary.FileWriter('./logs', sess.graph)

                    global_step = ckpt.model_checkpoint_path.split('-')[-1]

                    # xs, ys = sess.run([img_test, label_test])
                    xs, ys = sess.graph.get_tensor_by_name('test_dataset')  # 训练时运行过
                    ys = dog_and_cat_train.one_hot(ys, inference.NUM_LABELS)

                    reshaped_xs = np.reshape(xs, (TEST_NUM,
                                                  inference.IMAGE_SIZE,
                                                  inference.IMAGE_SIZE,
                                                  inference.NUM_CHANNELS))

                    result, accuracy_score = sess.run([merge_op, accuracy], feed_dict={x: reshaped_xs, y_: ys})  # 使用此模型检验
                    # 没有初始化滑动平均值，只是调用模型的值，inference只是提供了一个变量的接口，完全没有赋值
                    print("After %s training steps, validation accuracy = %g" % (global_step, accuracy_score))

                    writer.add_summary(result, global_step)
                else:
                    print("No checkpoint file found")
                    return
                # time sleep()函数推迟调用线程的运行，可通过参数secs指秒数，表示进程挂起的时间。
                time.sleep(EVAL_INTERVAL_SECS)

                coord.request_stop()
                coord.join(threads)

def main(argv=None):
    train_tfrecords_path = r".\my_dataset\tf_files\dog_and_cat_train.tfrecords"
    test_tfrecords_path = r".\my_dataset\tf_files\dog_and_cat_test.tfrecords"

    img_batch, label_batch, img_test, label_test = dog_and_cat_train.get_datasets(train_tfrecords_path,
                                                                                  test_tfrecords_path, cnn_train.BATCH_SIZE)
    evaluate(img_test, label_test)


if __name__ == '__main__':
    tf.app.run()

'''
启动可视化界面命令：
tensorboard --logdir=E:\pycharm\tensorflow-learn\cnn_learn\make_my_dataset\logs
'''
'''

'''