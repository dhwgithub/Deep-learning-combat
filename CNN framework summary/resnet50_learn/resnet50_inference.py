# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

####定义神经网络结构相关的参数########
INPUT_NODE = 784  # 输入层的节点数。对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 10  # 输出层的节点数。这个等于类别的数目。

####定义与样本数据相关的参数########
IMAGE_SIZE = 28  # 像素尺寸
NUM_CHANNELS = 1  # 通道数
NUM_LABELS = 10  # 手写数字类别数目

########## 第1阶段 ##############
CONV_SIZE = 5  # 7
CONV_NUM = 64  # 64
########## 第2阶段-残差块组 ##############
NET_OUT_NUM1 = 64  # 256
NET_OUT_NUM2 = 86  # 512
NET_OUT_NUM3 = 64  # 1024
NET_OUT_NUM4 = 32  # 2048


def conv2d(scope_name, input_tensor, conv_size, conv_num, stride, is_train=True, normalizer_fc=True, activation_fn=True):
    with tf.variable_scope(scope_name):
        conv_weights = tf.get_variable("weight",
                                       [conv_size, conv_size, input_tensor.shape[-1], conv_num],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_biases = tf.get_variable("bias", [conv_num], initializer=tf.constant_initializer(0.0))
        # SAME表示使用全0填充 VALID表示不填充
        conv = tf.nn.conv2d(input_tensor, conv_weights, strides=[1, stride, stride, 1], padding='SAME')
        if normalizer_fc:
            conv = tf.layers.batch_normalization(conv, training=is_train)
        if activation_fn:
            conv = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
        return conv


def bottle_net(input_tensor, output_depth, is_train, stride=1, scope='bottle_net'):
    with tf.variable_scope(scope):
        data = input_tensor
        depth = input_tensor.get_shape().as_list()[-1]
        if depth == output_depth:
            shortcut_tensor = input_tensor
        else:
            shortcut_tensor = conv2d('block_conv', input_tensor, 1, output_depth, stride, is_train, activation_fn=False)

        data = conv2d('block_conv1', data, 1, output_depth // 4, 1, is_train)
        data = conv2d('block_conv2', data, 3, output_depth // 4, stride, is_train)
        data = conv2d('block_conv3', data, 1, output_depth, 1, is_train, False, False)

        # 生成残差
        data = data + shortcut_tensor
        data = tf.nn.relu(data)
        return data


def create_block(input_tensor, output_depth, block_nums, init_stride=1, is_train=True, scope='scope'):
    with tf.variable_scope(scope):
        data = bottle_net(input_tensor, output_depth, is_train, stride=init_stride, scope=scope + '_0')
        for i in range(1, block_nums):
            data = bottle_net(data, output_depth, is_train, scope=scope + '_' + str(i))
        return data


def inference(input_tensor, is_train):
    # 将输入变为224x224x3
    input_tensor = tf.reshape(input_tensor, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    input_tensor = tf.pad(input_tensor, [[0, 0], [98, 98], [98, 98], [1, 1]])

    data = conv2d('conv1', input_tensor, CONV_SIZE, CONV_NUM, 2, is_train=is_train)
    data = slim.max_pool2d(data, 3, 2, padding='SAME', scope='pool_1')

    # 第一个残差块组
    data = create_block(data, NET_OUT_NUM1, 3, init_stride=1, is_train=is_train, scope='block1')

    # 第二个残差块组
    data = create_block(data, NET_OUT_NUM2, 4, init_stride=2, is_train=is_train, scope='block2')

    # 第三个残差块组
    data = create_block(data, NET_OUT_NUM3, 6, init_stride=2, is_train=is_train, scope='block3')

    # 第四个残差块组
    data = create_block(data, NET_OUT_NUM4, 3, init_stride=2, is_train=is_train, scope='block4')

    # 接下来就是池化层和全连接层
    data = slim.avg_pool2d(data, 7)
    data = conv2d('final_conv', data, 1, NUM_LABELS, 2, activation_fn=False)

    data_shape = data.get_shape().as_list()
    nodes = data_shape[1] * data_shape[2] * data_shape[3]
    logit = tf.reshape(data, [-1, nodes])

    return logit
