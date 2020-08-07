import tensorflow as tf
import numpy as np

# 导包报错忽略
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

#################  导入数据  ####################
# mnist = tf.keras.datasets.mnist
def mnist_load_data(path='mnist.npz'):
    path = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\mnist.npz'  # 下载的文件路径
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = mnist_load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

################  构建模型（二选一，之后以class为主）  ####################
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y

model = MnistModel()

################  训练模型并显示结果  ####################
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()
