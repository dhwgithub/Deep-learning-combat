import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])) + 0.1  # biases一般不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # np.newaxis表示在当前位置增加一维
noise = np.random.normal(0, 0.05, x_data.shape)  # 均值、方差、尺寸
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(x_data, y_data)
# plt.ion()  # 使得显示图像后程序可继续运行
# plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data,
                                    ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data,
                                        ys: y_data}))

        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        # prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # lines = ax.plot(x_data, prediction_value, 'r', lw=5)
        # plt.pause(1)

'''
1.18525
0.183151
0.103059
0.0608771
0.0384419
0.0263286
0.0195183
0.0155344
0.0131399
0.0116792
0.0107763
0.0102006
0.00981162
0.00950294
0.00924729
0.00902806
0.0088188
0.00862792
0.00843844
0.0082569
'''