import tensorflow as tf
import numpy as np

# # Save model
# # remember to define the same dtype and shape when restore
# W = tf.Variable([[1, 2, 3],
#                  [3, 4, 5]], dtype=tf.float32, name='weight')
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, './save_model/save_net.ckpt')
#     print('Save to path', save_path)


# Restore variable
# redefine the same shape and same type for your [variables], not save model
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weight')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

# not need init step
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './save_model/save_net.ckpt')
    print('weight:', sess.run(W))
    print('biases:', sess.run(b))
'''
weight: [[1. 2. 3.]
 [3. 4. 5.]]
biases: [[1. 2. 3.]]
'''