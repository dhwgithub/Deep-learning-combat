import cv2
import os
import tensorflow as tf
import numpy as np

# path = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\tf_flowers'
path = r'E:\pycharm\tensorflow-learn\cnn_learn\make_my_dataset\open_datasets\Tobacco3482'

WIDTH = 224
HEIGHT = 224
CHANNEL = 3
CATEGORY = 10

def read_img(path):
    imgs = []
    labels = []
    cate = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]  # 获取每类目录

    for idx, i in enumerate(cate):
        for j in os.listdir(i):  # 遍历每类目录文件
            im = cv2.imread(os.path.join(i, j))
            img = cv2.resize(im, (WIDTH, HEIGHT)) / 255.
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def to_one_hot(label):
    return tf.one_hot(label, CATEGORY)


################  准备数据集  #####################
data, label = read_img(path)

num_example = data.shape[0]  # data.shape是(3029, 100, 100, 3)
arr = np.arange(num_example)  # 创建等差数组 0，1，...,3028
np.random.shuffle(arr)  # 打乱顺序
data = data[arr]
label = label[arr]

label_oh = to_one_hot(label)

ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label_oh.numpy()[:s]
x_val = data[s:]
y_val = label_oh.numpy()[s:]

################  准备预处理模型  #####################
img_shape = (WIDTH, HEIGHT, CHANNEL)
base_model = tf.keras.applications.VGG16(input_shape=img_shape, include_top=False, weights='imagenet')

# base_model.trainable = False
base_model.trainable = True

# 设置前fine_tune_at层不训练
# print(len(base_model.layers))  # AGG16共19层
fine_tune_at = 15
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

################  准备模型  #####################
model1 = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(CATEGORY, activation="softmax")
])

model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
               loss=tf.keras.losses.categorical_crossentropy,
               metrics=[tf.keras.metrics.categorical_accuracy])

################  开始训练  #####################
history = model1.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

model1.summary()

################  测试  #####################
model1.evaluate(x_val, y_val, verbose=2)

'''
冻结所有层
697/1 - 19s - loss: 0.8934 - categorical_accuracy: 0.7202

冻结前15层
697/1 - 19s - loss: 1.1876 - categorical_accuracy: 0.6758
'''