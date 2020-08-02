# -*- coding: utf-8 -*-

import tensorflow as tf

####定义神经网络结构相关的参数########
INPUT_NODE = 784  # 输入层的节点数。对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 10  # 输出层的节点数。这个等于类别的数目。

####定义与样本数据相关的参数########
IMAGE_SIZE = 28  # 像素尺寸
NUM_CHANNELS = 1  # 通道数
NUM_LABELS = 10  # 手写数字类别数目

#########第一层卷积层的尺寸和深度############
C0NV1_DEEP = 27  # 与下面卷积层成倍数关系
C0NV1_SIZE = 3
#########第二层卷积层的尺寸和深度############
CONV2_DEEP = 27
CONV2_SIZE = 3
#########第三层卷积层的尺寸和深度############
CONV3_DEEP = 54
CONV3_SIZE = 3
#########第四层卷积层的尺寸和深度############
CONV4_DEEP = 54
CONV4_SIZE = 3
#########第五层卷积层的尺寸和深度############
CONV5_DEEP = 54
CONV5_SIZE = 3
##########全连接层的节点个数####################
FC_SIZE1 = 256  # 512 与调整学习率结合可以达到一定的效果
FC_SIZE2 = 84


def conv2d(input_tensor, scope_name, conv_size, num_channels, conv_num, stride):
    # 由于每段卷积层包含多次卷积，因此这里额外加入变量域
    with tf.variable_scope(scope_name):
        conv_weights = tf.get_variable("weight",
                                       [conv_size, conv_size, num_channels, conv_num],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_biases = tf.get_variable("bias", [conv_num], initializer=tf.constant_initializer(0.0))
        # SAME表示使用全0填充 VALID表示不填充
        conv = tf.nn.conv2d(input_tensor, conv_weights, strides=[1, stride, stride, 1], padding='SAME')
        # tf.nn.bias_add提供了一个方便的函数给每一个节点加上偏置项。注意这里不能直接使用加法，
        # 因为矩阵上不同位置上的节点都需要加上同样的偏置项。
        conv = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    return conv


def full_conn(input_tensor, input_size, output_size, regularizer, have_act_fun, is_train, scope_name):
    with tf.variable_scope(scope_name):
        fc_weights = tf.get_variable("weight",
                                     [input_size, output_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc_weights))
        fc_biases = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.1))
        fc = tf.matmul(input_tensor, fc_weights) + fc_biases
        if have_act_fun:
            fc = tf.nn.relu(fc)
        if is_train:
            fc = tf.nn.dropout(fc, 0.5)
    return fc


def inference(input_tensor, is_train, regularizer):
    # 将输入变为224x224x3
    input_tensor = tf.reshape(input_tensor, shape=[-1, 28, 28, 1])
    input_tensor = tf.pad(input_tensor, [[0, 0], [98, 98], [98, 98], [1, 1]])

    with tf.variable_scope('layer1-conv1'):
        conv1 = conv2d(input_tensor, 'conv1_1', C0NV1_SIZE, 3, C0NV1_DEEP, 1)
        conv1 = conv2d(conv1, 'conv1_2', C0NV1_SIZE, 3, C0NV1_DEEP, 1)

    # 实现第二层池化层的前向传播过程。这里选用最大池化层，池化层过滤器的边长为2，
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 声明第三层卷积层的变量并实现前向传播过程
    with tf.variable_scope("layer3-conv2"):
        conv2 = conv2d(pool1, 'conv2_1', CONV2_SIZE, C0NV1_DEEP, CONV2_DEEP, 1)
        conv2 = conv2d(conv2, 'conv2_2', CONV2_SIZE, C0NV1_DEEP, CONV2_DEEP, 1)

    # 实现第四层池化层的前向传播过程。这一层和第二层的结构是一样的。
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer5-conv3"):
        conv3 = conv2d(pool2, 'conv3_1', CONV3_SIZE, CONV2_DEEP, CONV3_DEEP, 1)
        conv3 = conv2d(conv3, 'conv3_2', CONV3_SIZE, CONV2_DEEP, CONV3_DEEP, 1)
        conv3 = conv2d(conv3, 'conv3_3', CONV3_SIZE, CONV2_DEEP, CONV3_DEEP, 1)

    # 实现第四层池化层的前向传播过程。这一层和第二层的结构是一样的。
    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer7-conv4"):
        conv4 = conv2d(pool3, 'conv4_1', CONV4_SIZE, CONV3_DEEP, CONV4_DEEP, 1)
        conv4 = conv2d(conv4, 'conv4_2', CONV4_SIZE, CONV3_DEEP, CONV4_DEEP, 1)
        conv4 = conv2d(conv4, 'conv4_3', CONV4_SIZE, CONV3_DEEP, CONV4_DEEP, 1)

    # 实现第四层池化层的前向传播过程。这一层和第二层的结构是一样的。
    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer9-conv5"):
        conv5 = conv2d(pool4, 'conv5_1', CONV5_SIZE, CONV4_DEEP, CONV5_DEEP, 1)
        conv5 = conv2d(conv5, 'conv5_2', CONV5_SIZE, CONV4_DEEP, CONV5_DEEP, 1)
        conv5 = conv2d(conv5, 'conv5_3', CONV5_SIZE, CONV4_DEEP, CONV5_DEEP, 1)

    # 实现第四层池化层的前向传播过程。这一层和第二层的结构是一样的。
    with tf.name_scope("layer10-pool5"):
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool5.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool5, [pool_shape[0], nodes])

    fc1 = full_conn(reshaped, nodes, FC_SIZE1, regularizer, True, is_train, "layer11-fc1")

    fc2 = full_conn(fc1, FC_SIZE1, FC_SIZE2, regularizer, True, is_train, "layer12-fc2")

    # 这一层的输出通过Softmax之后就得到了最后的分类结果。
    logit = full_conn(fc2, FC_SIZE2, NUM_LABELS, regularizer, False, False, 'layer13-fc3')

    return logit
