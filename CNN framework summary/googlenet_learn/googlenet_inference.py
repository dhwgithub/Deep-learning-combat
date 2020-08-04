# -*- coding: utf-8 -*-

import tensorflow as tf

####定义神经网络结构相关的参数########
INPUT_NODE = 784  # 输入层的节点数。对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 10  # 输出层的节点数。这个等于类别的数目。

####定义与样本数据相关的参数########
IMAGE_SIZE = 28  # 像素尺寸
NUM_CHANNELS = 1  # 通道数
NUM_LABELS = 10  # 手写数字类别数目

###########  第1阶段  #############
K_SIZE1 = 3  # 7
NET_DEEP1 = 48  # 64
###########  第2阶段  #############
K_SIZE2 = 3  # 3
NET_DEEP2 = 96  # 192
###########  第3--5阶段  #############
INCEPTION1_DEEP1 = [[32, 48],
                    [56, 64, 56, 54, 84],
                    [86, 86]]
# [[64, 128],
# [192, 160, 128, 112, 256],
# [256, 384]]
INCEPTION2_DEEP2_1 = [[16, 24],
                      [24, 32, 32, 48, 48],
                      [48, 48]]
# [[96, 128],
# [96, 112, 128, 144, 160],
# [160, 192]]
INCEPTION2_DEEP2_2 = [[32, 48],
                      [48, 54, 64, 84, 86],
                      [86, 96]]
# [[128, 192],
# [208, 224, 256, 288, 320],
# [320, 384]]
INCEPTION3_DEEP3_1 = [[16, 24],
                      [24, 32, 32, 32, 32],
                      [32, 32]]
# [[16, 32],
# [16, 24, 24, 32, 32],
# [32, 48]]
INCEPTION3_DEEP3_2 = [[32, 48],
                      [48, 56, 56, 56, 64],
                      [64, 64]]
# [[32, 96],
# [48, 64, 64, 64, 128],
# [128, 128]]
INCEPTION4_DEEP1 = [[24, 32],
                    [32, 32, 32, 32, 64],
                    [64, 64]]
# [[32, 64],
# [64, 64, 64, 64, 128],
# [128, 128]]


def __conv2d(scope_name, input_tensor, conv_size, conv_num, stride):
    # 由于每段卷积层包含多次卷积，因此这里额外加入变量域
    with tf.variable_scope(scope_name):
        conv_weights = tf.get_variable("weight",
                                       [conv_size, conv_size, input_tensor.shape[-1], conv_num],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_biases = tf.get_variable("bias", [conv_num], initializer=tf.constant_initializer(0.0))
        # SAME表示使用全0填充 VALID表示不填充
        conv = tf.nn.conv2d(input_tensor, conv_weights, strides=[1, stride, stride, 1], padding='SAME')
        # tf.nn.bias_add提供了一个方便的函数给每一个节点加上偏置项。注意这里不能直接使用加法，
        # 因为矩阵上不同位置上的节点都需要加上同样的偏置项。
        conv = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
        return conv


def __full_conn(scope_name, input_tensor, output_size, regularizer):
    with tf.variable_scope(scope_name):
        fc_weights = tf.get_variable("weight",
                                     [input_tensor.shape[-1], output_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc_weights))
        fc_biases = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.1))
        fc = tf.matmul(input_tensor, fc_weights) + fc_biases
        return fc


def __max_pool(scope_name, input_tensor, k_size, stride_size, padding='SAME'):
    with tf.variable_scope(scope_name):
        pool = tf.nn.max_pool(input_tensor, ksize=[1, k_size, k_size, 1],
                              strides=[1, stride_size, stride_size, 1],
                              padding=padding)
        return pool


def __avg_pool(scope_name, input_tensor, k_size, stride_size, padding='SAME'):
    with tf.variable_scope(scope_name):
        pool = tf.nn.avg_pool(input_tensor, ksize=[1, k_size, k_size, 1],
                              strides=[1, stride_size, stride_size, 1],
                              padding=padding)
        return pool


def __inception(scope_name, input_tensor,
                branch1_output_channels,
                branch2_reduce_output_channels, branch2_output_channels,
                branch3_reduce_output_channels, branch3_output_channels,
                branch4_output_channels):
    with tf.variable_scope(scope_name):
        with tf.variable_scope("branch_1"):
            # 1. 第一个分支
            net1 = __conv2d('conv1', input_tensor, 1, branch1_output_channels, 1)
        with tf.variable_scope("branch_2"):
            # 2. 第二个分支
            tmp_net = __conv2d('conv1', input_tensor, 1, branch2_reduce_output_channels, 1)
            net2 = __conv2d('conv2', tmp_net, 3, branch2_output_channels, 1)
        with tf.variable_scope("branch_3"):
            # 3. 第三个分支
            tmp_net = __conv2d('conv1', input_tensor, 1, branch3_reduce_output_channels, 1)
            net3 = __conv2d('conv2', tmp_net, 5, branch3_output_channels, 1)
        with tf.variable_scope("branch_4"):
            # 4. 第四个分支
            tmp_net = __max_pool('pool1', input_tensor, 3, 1)
            net4 = __conv2d('conv1', tmp_net, 1, branch4_output_channels, 1)
        with tf.variable_scope("Concat"):
            net = tf.concat([net1, net2, net3, net4], axis=-1)
    return net


def inference(input_tensor, is_train, regularizer):
    # 将输入变为224x224x3
    net = tf.reshape(input_tensor, shape=[-1, 28, 28, 1])
    net = tf.pad(net, [[0, 0], [98, 98], [98, 98], [1, 1]])

    # 第1阶段
    net = __conv2d('conv1', net, K_SIZE1, NET_DEEP1, 2)
    net = __max_pool('pool2', net, 3, 2)

    # 第2阶段
    net = __conv2d('conv3', net, K_SIZE2, NET_DEEP2, 1)
    net = __conv2d('conv4', net, K_SIZE2, NET_DEEP2, 1)
    net = __max_pool('pool5', net, 3, 2)

    # 第3阶段
    net = __inception('inception6', net,
                      INCEPTION1_DEEP1[0][0],
                      INCEPTION2_DEEP2_1[0][0], INCEPTION2_DEEP2_2[0][0],
                      INCEPTION3_DEEP3_1[0][0], INCEPTION3_DEEP3_2[0][0],
                      INCEPTION4_DEEP1[0][0])
    net = __inception('inception7', net,
                      INCEPTION1_DEEP1[0][0],
                      INCEPTION2_DEEP2_1[0][0], INCEPTION2_DEEP2_2[0][0],
                      INCEPTION3_DEEP3_1[0][0], INCEPTION3_DEEP3_2[0][0],
                      INCEPTION4_DEEP1[0][0])
    net = __inception('inception8', net,
                      INCEPTION1_DEEP1[0][1],
                      INCEPTION2_DEEP2_1[0][1], INCEPTION2_DEEP2_2[0][1],
                      INCEPTION3_DEEP3_1[0][1], INCEPTION3_DEEP3_2[0][1],
                      INCEPTION4_DEEP1[0][1])
    net = __inception('inception9', net,
                      INCEPTION1_DEEP1[0][1],
                      INCEPTION2_DEEP2_1[0][1], INCEPTION2_DEEP2_2[0][1],
                      INCEPTION3_DEEP3_1[0][1], INCEPTION3_DEEP3_2[0][1],
                      INCEPTION4_DEEP1[0][1])
    net = __max_pool('pool10', net, 3, 2)

    # 第4阶段
    net = __inception('inception11', net,
                      INCEPTION1_DEEP1[1][0],
                      INCEPTION2_DEEP2_1[1][0], INCEPTION2_DEEP2_2[1][0],
                      INCEPTION3_DEEP3_1[1][0], INCEPTION3_DEEP3_2[1][0],
                      INCEPTION4_DEEP1[1][0])
    net = __inception('inception12', net,
                      INCEPTION1_DEEP1[1][0],
                      INCEPTION2_DEEP2_1[1][0], INCEPTION2_DEEP2_2[1][0],
                      INCEPTION3_DEEP3_1[1][0], INCEPTION3_DEEP3_2[1][0],
                      INCEPTION4_DEEP1[1][0])
    net = __inception('inception13', net,
                      INCEPTION1_DEEP1[1][1],
                      INCEPTION2_DEEP2_1[1][1], INCEPTION2_DEEP2_2[1][1],
                      INCEPTION3_DEEP3_1[1][1], INCEPTION3_DEEP3_2[1][1],
                      INCEPTION4_DEEP1[1][1])
    net = __inception('inception14', net,
                      INCEPTION1_DEEP1[1][1],
                      INCEPTION2_DEEP2_1[1][1], INCEPTION2_DEEP2_2[1][1],
                      INCEPTION3_DEEP3_1[1][1], INCEPTION3_DEEP3_2[1][1],
                      INCEPTION4_DEEP1[1][1])
    net = __inception('inception15', net,
                      INCEPTION1_DEEP1[1][2],
                      INCEPTION2_DEEP2_1[1][2], INCEPTION2_DEEP2_2[1][2],
                      INCEPTION3_DEEP3_1[1][2], INCEPTION3_DEEP3_2[1][2],
                      INCEPTION4_DEEP1[1][2])
    net = __inception('inception16', net,
                      INCEPTION1_DEEP1[1][2],
                      INCEPTION2_DEEP2_1[1][2], INCEPTION2_DEEP2_2[1][2],
                      INCEPTION3_DEEP3_1[1][2], INCEPTION3_DEEP3_2[1][2],
                      INCEPTION4_DEEP1[1][2])
    net = __inception('inception17', net,
                      INCEPTION1_DEEP1[1][3],
                      INCEPTION2_DEEP2_1[1][3], INCEPTION2_DEEP2_2[1][3],
                      INCEPTION3_DEEP3_1[1][3], INCEPTION3_DEEP3_2[1][3],
                      INCEPTION4_DEEP1[1][3])
    net = __inception('inception18', net,
                      INCEPTION1_DEEP1[1][3],
                      INCEPTION2_DEEP2_1[1][3], INCEPTION2_DEEP2_2[1][3],
                      INCEPTION3_DEEP3_1[1][3], INCEPTION3_DEEP3_2[1][3],
                      INCEPTION4_DEEP1[1][3])
    net = __inception('inception19', net,
                      INCEPTION1_DEEP1[1][4],
                      INCEPTION2_DEEP2_1[1][4], INCEPTION2_DEEP2_2[1][4],
                      INCEPTION3_DEEP3_1[1][4], INCEPTION3_DEEP3_2[1][4],
                      INCEPTION4_DEEP1[1][4])
    net = __inception('inception20', net,
                      INCEPTION1_DEEP1[1][4],
                      INCEPTION2_DEEP2_1[1][4], INCEPTION2_DEEP2_2[1][4],
                      INCEPTION3_DEEP3_1[1][4], INCEPTION3_DEEP3_2[1][4],
                      INCEPTION4_DEEP1[1][4])
    net = __max_pool('pool21', net, 3, 2)

    # 第5阶段
    net = __inception('inception22', net,
                      INCEPTION1_DEEP1[2][0],
                      INCEPTION2_DEEP2_1[2][0], INCEPTION2_DEEP2_2[2][0],
                      INCEPTION3_DEEP3_1[2][0], INCEPTION3_DEEP3_2[2][0],
                      INCEPTION4_DEEP1[2][0])
    net = __inception('inception23', net,
                      INCEPTION1_DEEP1[2][0],
                      INCEPTION2_DEEP2_1[2][0], INCEPTION2_DEEP2_2[2][0],
                      INCEPTION3_DEEP3_1[2][0], INCEPTION3_DEEP3_2[2][0],
                      INCEPTION4_DEEP1[2][0])
    net = __inception('inception24', net,
                      INCEPTION1_DEEP1[2][1],
                      INCEPTION2_DEEP2_1[2][1], INCEPTION2_DEEP2_2[2][1],
                      INCEPTION3_DEEP3_1[2][1], INCEPTION3_DEEP3_2[2][1],
                      INCEPTION4_DEEP1[2][1])
    net = __inception('inception25', net,
                      INCEPTION1_DEEP1[2][1],
                      INCEPTION2_DEEP2_1[2][1], INCEPTION2_DEEP2_2[2][1],
                      INCEPTION3_DEEP3_1[2][1], INCEPTION3_DEEP3_2[2][1],
                      INCEPTION4_DEEP1[2][1])
    net = __avg_pool('pool26', net, 7, 1, padding='VALID')
    if is_train:
        net = tf.nn.dropout(net, keep_prob=0.4)

    shapes = net.shape
    net = tf.reshape(net, shape=[-1, shapes[1] * shapes[2] * shapes[3]])

    logit = __full_conn('fc27', net, NUM_LABELS, regularizer)

    return logit
