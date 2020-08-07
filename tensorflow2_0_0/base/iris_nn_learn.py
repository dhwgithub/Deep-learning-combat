import tensorflow as tf
from sklearn import datasets
import numpy as np

# 导包报错忽略
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

#########  导入数据集  #######################
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

#########  打乱数据集  #######################
np.random.seed(116)
np.random.shuffle(x_train)

np.random.seed(116)
np.random.shuffle(y_train)

tf.random.set_seed(116)

#########  构建网络结构（类与Sequential二选一）  #######################
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
# ])

class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.d1(x)
        return y

model = IrisModel()

#########  指定训练优化器  #######################
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

#########  训练模型  #######################
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

#########  显示训练结果  #######################
model.summary()
