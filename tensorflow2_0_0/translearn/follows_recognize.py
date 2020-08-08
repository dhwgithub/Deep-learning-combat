from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# tf.enable_eager_execution()
IMAGE_RES = 224
NUM_CLASSES = 5

# 训练和验证数据集70：30 的比例分配数据
data_dirs = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\tf_flowers'
(training_set, validation_set), dataset_info = tfds.load(data_dirs, with_info=True, as_supervised=True, split=['train[:70%]', 'train[70%:]'])

# 由于图片的尺寸各不相同，我们需要统一尺寸，并且为数据制作 Batch
def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 225
    return image, label

BATCH_SIZE = 32
SEED = 66  # num_training_examples // 4
train_batches = training_set.shuffle(SEED).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)
# print(type(train_batches))  # <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>

# 下述代码解决：urllib.error.URLError: <urlopen error EOF occurred in violation of protocol (_ssl.c:749)>
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
import ssl
import requests

class MyAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       ssl_version=ssl.PROTOCOL_TLSv1)

requests.Session().mount('https://', MyAdapter())

# 导入已经被训练好的模型
# 需要注意的是我们需要的不是完整的模型，而是模型在分类之前(没有最后一层的神经网络)的结构和参数。这种模型是 Feature Vector
# 模型下载主页链接：https://www.tensorflow.org/hub
URL = r"https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES, 3))
feature_extractor.trainable = False  # 不改变已训练模型的参数

# 整合Keras模型
model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(
     train_batches,
     epochs=6,
     validation_data=validation_batches
)

model.summary()
