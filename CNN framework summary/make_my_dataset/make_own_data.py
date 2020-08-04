# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

'''
生成自己的数据集（tensorflow可以处理的tfrecords文件）
准备：
    训练集和测试集
    每个集合里包含已经分好类的文件夹
只需修改文件位置和保存名称即可
制作数据集是(128, 128, 3)格式的数据
'''
def _make_dataset():
    cwd = './my_dataset/train/'
    # cwd = './my_dataset/test/'
    classes = {'dog', 'cat'}  # 人为设定2类
    writer= tf.python_io.TFRecordWriter("./my_dataset/tf_files/dog_and_cat_train.tfrecords") #要生成的文件
    # writer = tf.python_io.TFRecordWriter("./my_dataset/tf_files/dog_and_cat_test.tfrecords")  # 要生成的文件

    for index, name in enumerate(classes):
        class_path = os.path.join(cwd, name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)  # 每一个图片的地址

            img = Image.open(img_path)
            img = img.resize((128, 128))
            img = img.convert('RGB')
            print(np.shape(img))

            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )
            )  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()


def read_and_decode(filename):  # 读入tfrecords
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量

    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量

    return img, label
