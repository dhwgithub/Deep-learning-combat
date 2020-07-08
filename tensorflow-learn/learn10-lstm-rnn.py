'''
分类例子
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./datasets/MNIST', False, one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28  # 图像一行一行读取入RNN
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,
                                             forget_bias=1.0,
                                             state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # pre step states
    outputs, states = tf.nn.dynamic_rnn(lstm_cell,
                                        X_in,
                                        initial_state=init_state,
                                        time_major=False)

    # hidden layer for output as the final result
    # # method 1
    # result = tf.matmul(states[1], weights['out']) + biases['out']
    # method 2
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    result = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return result


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
        step += 1


'''
0.2109375
0.703125
0.8203125
0.859375
0.8046875
0.8828125
0.8671875
0.859375
0.90625
0.9453125
0.9453125
0.9609375
0.921875
0.953125
0.9296875
0.921875
0.953125
0.9453125
0.9140625
0.9453125
0.9296875
0.9453125
0.9375
0.9453125
0.984375
0.984375
0.9296875
0.9765625
0.9765625
0.984375
0.9296875
0.9609375
0.96875
0.9453125
0.96875
0.9609375
0.9609375
0.953125
0.9765625
0.984375
'''