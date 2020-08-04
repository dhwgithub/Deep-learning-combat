# -*- coding: utf-8 -*-

import tensorflow as tf

####定义神经网络结构相关的参数########
INPUT_NODE = 49152  # 输入层的节点数。对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 2  # 输出层的节点数。这个等于类别的数目。

####定义与样本数据相关的参数########
IMAGE_SIZE = 128  # 像素尺寸
NUM_CHANNELS = 3  # 通道数
NUM_LABELS = 2  # 手写数字类别数目

#########第一层卷积层的尺寸和深度############
C0NV1_DEEP = 32
C0NV1_SIZE = 5
#########第二层卷积层的尺寸和深度############
CONV2_DEEP = 64
CONV2_SIZE = 5
#########第三层卷积层的尺寸和深度############
CONV3_DEEP = 98  # 348
CONV3_SIZE = 3
#########第四层卷积层的尺寸和深度############
CONV4_DEEP = 64  # 384
CONV4_SIZE = 3
#########第五层卷积层的尺寸和深度############
CONV5_DEEP = 32  # 256
CONV5_SIZE = 3
##########全连接层的节点个数####################
FC_SIZE1 = 125  # 4096
FC_SIZE2 = 10  # 4096


def conv2d(input_tensor, conv_size, num_channels, conv_num, stride):
    conv_weights = tf.get_variable("weight",
                                   [conv_size, conv_size, num_channels, conv_num],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv_biases = tf.get_variable("bias", [conv_num], initializer=tf.constant_initializer(0.0))
    # SAME表示使用全0填充 VALID表示不填充
    conv = tf.nn.conv2d(input_tensor, conv_weights, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))


def full_conn(input_tensor, input_size, output_size, regularizer, have_act_fun, is_train):
    fc_weights = tf.get_variable("weight",
                                 [input_size, output_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(fc_weights))
    fc_biases = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.1))
    fc = tf.matmul(input_tensor, fc_weights) + fc_biases
    if have_act_fun:
        fc = tf.nn.relu(fc)
    if is_train == 'True':
        fc = tf.nn.dropout(fc, 0.5)
    return fc


def LRN(input, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    """Local Response Normalization 局部响应归一化"""
    return tf.nn.local_response_normalization(input, depth_radius=depth_radius, alpha=alpha,
                                              beta=beta, bias=bias)


def inference(input_tensor, is_train, regularizer):
    if is_train is 'False':
        regularizer = None
        print(is_train, ' ', regularizer)

    # 调整输入向量大小
    input_tensor = tf.reshape(input_tensor, shape=[-1, 128, 128, 3])

    with tf.variable_scope('layer1-conv1'):
        relu1 = conv2d(input_tensor, C0NV1_SIZE, 3, C0NV1_DEEP, 4)

    with tf.name_scope("layer1-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.name_scope("layer1-LRN"):
        pool1 = LRN(pool1)

    with tf.variable_scope("layer2-conv2"):
        relu2 = conv2d(pool1, CONV2_SIZE, C0NV1_DEEP, CONV2_DEEP, 1)

    with tf.name_scope("layer2-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.name_scope("layer2-LRN"):
        pool2 = LRN(pool2)

    with tf.variable_scope("layer3-conv3"):
        relu3 = conv2d(pool2, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP, 1)

    with tf.variable_scope("layer4-conv4"):
        relu4 = conv2d(relu3, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP, 1)

    with tf.variable_scope("layer5-conv5"):
        relu5 = conv2d(relu4, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP, 1)

    with tf.name_scope("layer5-pool3"):
        pool3 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool3, [pool_shape[0], nodes])

    # print(nodes)
    with tf.variable_scope("layer6-fc1"):
        fc1 = full_conn(reshaped, nodes, FC_SIZE1, regularizer, True, is_train)

    with tf.variable_scope("layer7-fc2"):
        fc2 = full_conn(fc1, FC_SIZE1, FC_SIZE2, regularizer, True, is_train)

    with tf.variable_scope('layer8-fc3'):
        logit = full_conn(fc2, FC_SIZE2, NUM_LABELS, regularizer, False, False)

    return logit
