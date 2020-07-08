import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# start train
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

'''
0 [-0.21398616] [ 0.67932057]
20 [-0.00525665] [ 0.35859704]
40 [ 0.07236522] [ 0.31538448]
60 [ 0.09274458] [ 0.30403915]
80 [ 0.09809512] [ 0.30106047]
100 [ 0.09949987] [ 0.30027843]
120 [ 0.09986869] [ 0.30007312]
140 [ 0.09996552] [ 0.3000192]
160 [ 0.09999095] [ 0.30000505]
180 [ 0.09999764] [ 0.30000132]
200 [ 0.09999939] [ 0.30000037]
'''