from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = '../save_model/mnist_model/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

model.load_weights(model_save_path)

preNum = int(input("input the number of test pictures:"))
for i in range(preNum):
    image_path = input("the path of test picture:")
    img = Image.open(image_path)

    ##########  图像预处理  #############
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]  # 将3维升至4维

    ############# 图像预测  ##############
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)

    tf.print(pred)
    print('\n')
