import tensorflow as tf
from PIL import Image
import numpy as np
import os
'''
基于mnist_cnn_learn.py进行扩展，实现自制数据集
'''
train_path = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\MNIST\train_img'
train_txt = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\MNIST\train_label.txt'
x_train_savepath = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\my_datasets\mnist_x_train.npy'
y_train_savepath = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\my_datasets\mnist_y_train.npy'

test_path = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\MNIST\test_img'
test_txt = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\MNIST\test_label.txt'
x_test_savepath = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\my_datasets\mnist_x_test.npy'
y_test_savepath = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\my_datasets\mnist_y_test.npy'


'''
将图片和标签进行整合并返回整理好的数据
'''
def generateds(path, txt):
    with open(txt, 'r') as f:
        labels = f.readlines()  # 按行读取，只有一行，因此无需循环

    x, y_ = [], []
    label_arr = labels[0].split(',')  # 以逗号分开，存入数组
    img_index = 0
    for name in label_arr:
        # 遍历每一张图片
        img_path = os.path.join(path, str(img_index)) + '.jpg'
        img_index += 1
        img = Image.open(img_path)
        # 转变为灰度图像
        img = np.array(img.convert('L'))
        img = img / 255.
        # 存储图像和标签
        x.append(img)  # 图片信息
        y_.append(name)  # 图片对应的标签

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_


if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and \
        os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
