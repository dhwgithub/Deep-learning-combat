import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate

# data
mnist = input_data.read_data_sets('../datasets/MNIST', False, one_hot=True)              # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])       # shape(batch, 784)
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])                   # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])                             # input y

# RNN
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
output = tf.layers.dense(outputs[:, -1, :], 10)              # output based on the last output step

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

for step in range(1200):    # training
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:      # testing
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
'''
train loss: 2.3271 | test accuracy: 0.14
train loss: 0.8738 | test accuracy: 0.42
train loss: 0.6539 | test accuracy: 0.53
train loss: 0.6075 | test accuracy: 0.62
train loss: 0.3204 | test accuracy: 0.67
train loss: 0.1272 | test accuracy: 0.71
train loss: 0.2556 | test accuracy: 0.74
train loss: 0.1149 | test accuracy: 0.77
train loss: 0.2763 | test accuracy: 0.79
train loss: 0.3610 | test accuracy: 0.80
train loss: 0.1498 | test accuracy: 0.81
train loss: 0.3310 | test accuracy: 0.82
train loss: 0.1810 | test accuracy: 0.83
train loss: 0.1142 | test accuracy: 0.84
train loss: 0.1013 | test accuracy: 0.85
train loss: 0.1615 | test accuracy: 0.86
train loss: 0.1302 | test accuracy: 0.86
train loss: 0.1756 | test accuracy: 0.87
train loss: 0.2822 | test accuracy: 0.87
train loss: 0.0824 | test accuracy: 0.88
train loss: 0.2707 | test accuracy: 0.88
train loss: 0.1206 | test accuracy: 0.88
train loss: 0.1329 | test accuracy: 0.89
train loss: 0.0791 | test accuracy: 0.89
'''

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')
'''
[7 2 1 0 4 1 4 9 5 9] prediction number
[7 2 1 0 4 1 4 9 5 9] real number
'''