import tensorflow as tf
import os
import numpy as np

'''
基于mnist_cnn_learn.py进行扩展，实现模型的保存和读取
第一次运行时无模型，所以是直接训练
第二次运行有模型，可以从上次训练处开始训练
'''

def mnist_load_data(path='mnist.npz'):
    path = r'E:\pycharm\tensorflow-learn\tensorflow2_0_0\datasets\mnist.npz'  # 下载的文件路径
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = mnist_load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

############# 若有模型，则直接加载  ######################
checkpoint_save_path = "../save_model/mnist_model/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

############ 对此次训练模型进行保存  ######################
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()
