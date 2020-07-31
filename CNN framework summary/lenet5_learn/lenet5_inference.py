# -*- coding: utf-8 -*-
# 参考https://blog.csdn.net/weixin_41695564/article/details/80240106?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param

import tensorflow as tf

####定义神经网络结构相关的参数########
INPUT_NODE = 784  # 输入层的节点数。对于MNIST数据集，这个就等于图片的像素
OUTPUT_NODE = 10  # 输出层的节点数。这个等于类别的数目。

####定义与样本数据相关的参数########
IMAGE_SIZE = 28  # 像素尺寸
NUM_CHANNELS = 1  # 通道数
NUM_LABELS = 10  # 手写数字类别数目

#########第一层卷积层的尺寸和深度(无0填充)############
C0NV1_DEEP = 6
C0NV1_SIZE = 5
#########第二层卷积层的尺寸和深度(无0填充)############
CONV2_DEEP = 16
CONV2_SIZE = 5
##########全连接层的节点个数####################
FC_SIZE1 = 120
FC_SIZE2 = 84


def conv2d(input_tensor, conv_size, num_channels, conv_num, stride):
    conv_weights = tf.get_variable("weight",
                                   [conv_size, conv_size, num_channels, conv_num],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv_biases = tf.get_variable("bias", [conv_num], initializer=tf.constant_initializer(0.0))
    # SAME表示使用全0填充 VALID表示不填充
    conv = tf.nn.conv2d(input_tensor, conv_weights, strides=[1, stride, stride, 1], padding='VALID')
    # tf.nn.bias_add提供了一个方便的函数给每一个节点加上偏置项。注意这里不能直接使用加法，
    # 因为矩阵上不同位置上的节点都需要加上同样的偏置项。
    return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))


def full_conn(input_tensor, input_size, output_size, regularizer):
    fc_weights = tf.get_variable("weight",
                                 [input_size, output_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 只有全连接层的权重需要加入正则
    # 当给出了正则化生成函数时，将当前变量的正则化损失加入名字为losses的集合。在这里
    # 使用了add_to_collection函数将一个张量加入一个集合，而这个集合的名称为losses。
    # 注意这是自定义的集合，不在TensorFlow自动管理的集合列表中。
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(fc_weights))
    fc_biases = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.1))
    return tf.matmul(input_tensor, fc_weights) + fc_biases


def inference(input_tensor, is_train, regularizer):
    """
    定义卷积神经网络的前向传播过程。这里添加了一个新的参数train，用于区分训练过程和测试过程。
    在这个程序中将用到dropout方法，dropout可以进一步提升模型可靠性并防止过拟合，dropout过程只在训练时使用。
    """
    # 调整输入向量大小
    input_tensor = tf.reshape(input_tensor, shape=[-1, 28, 28, 1])
    input_tensor = tf.pad(input_tensor, [[0, 0], [2, 2], [2, 2], [0, 0]])

    # 声明第一层卷积层的变量并实现前向传播过程。
    # 通过使用不同的命名空间来隔离不同层的变量，这可以让每一层中的变量命名只需要考虑在当前层的作用，而不需要担心重名的问题。
    # print('layer1-conv1 ' + str(input_tensor.shape))  # layer1-conv1 (100, 32, 32, 1)
    with tf.variable_scope('layer1-conv1'):
        relu1 = conv2d(input_tensor, C0NV1_SIZE, NUM_CHANNELS, C0NV1_DEEP, 1)
        # print('layer1-conv1 ' + str(relu1.shape))  # layer1-conv1 (100, 28, 28, 6)

    # 实现第二层池化层的前向传播过程。这里选用最大池化层，池化层过滤器的边长为2，
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # print('layer2-pool1 ' + str(pool1.shape))  # layer2-pool1 (100, 14, 14, 6)

    # 声明第三层卷积层的变量并实现前向传播过程
    with tf.variable_scope("layer3-conv2"):
        relu2 = conv2d(pool1, CONV2_SIZE, C0NV1_DEEP, CONV2_DEEP, 1)
        # print('layer3-conv2 ' + str(relu2.shape))  # layer3-conv2 (100, 10, 10, 16)

    # 实现第四层池化层的前向传播过程。这一层和第二层的结构是一样的。
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # print('layer4-pool2 ' + str(pool2.shape))  # layer4-pool2 (100, 5, 5, 16)

    # 将第四层池化层的输出转化为第五层全连接层的输入格式。
    # 然而第五层全连接层需要的输入格式为向量，所以在这里需要将这个7x7x64的矩阵拉直成一
    # 个向量。 pool2.get_Shape函数可以得到第四层输出矩阵的维度而不需要手工计算。
    # 注意,因为每一层神经网络碎输入输出都为一个batch的矩阵，所以这里得到的维度也包含了一个batch中数据的个数。
    pool_shape = pool2.get_shape().as_list()
    # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长宽及深度的乘积。
    # 注意这里pool_shape[0]为一个batch中数据的个数。
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 通过tf.reshape函数将第四层的输出变成一个batch的向量。
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 声明第五层全连接层的变量并实现前向传播过程。这一层的输入是拉直之后的一组向量
    # 此处引入了 dropout的概念。
    # dropout在训练时会随机将部分节点的输出改为0。dropout可以避免过拟合问题，从而使得模型在测试数据上的效果更好。
    # dropout一般只在全连接层而不是卷积层或者池化层使用。
    # print('reshaped ' + str(reshaped.shape))  # reshaped (100, 400)
    with tf.variable_scope("layer5-fc1"):
        fc1 = full_conn(reshaped, nodes, FC_SIZE1, regularizer)
        fc1 = tf.nn.relu(fc1)
        if is_train:
            fc1 = tf.nn.dropout(fc1, 0.5)
        # print('layer5-fc1 ' + str(fc1.shape))  # layer5-fc1 (100, 120)

    with tf.variable_scope("layer6-fc2"):
        fc2 = full_conn(fc1, FC_SIZE1, FC_SIZE2, regularizer)
        fc2 = tf.nn.relu(fc2)
        if is_train:
            fc2 = tf.nn.dropout(fc2, 0.5)
        # print('layer6-fc2 ' + str(fc2.shape))  # layer6-fc2 (100, 84)

    # 声明第六层全连接层的变量并实现前向传播过程。
    # 这一层的输出通过Softmax之后就得到了最后的分类结果。
    with tf.variable_scope('layer7-fc3'):
        logit = full_conn(fc2, FC_SIZE2, NUM_LABELS, regularizer)
        # print('layer7-fc3 ' + str(logit.shape))  # layer7-fc3 (100, 10)

    # 返回第六层的输出。
    return logit
