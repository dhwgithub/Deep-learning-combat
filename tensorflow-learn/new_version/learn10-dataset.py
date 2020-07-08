import tensorflow as tf
import numpy as np


# load your data or create your data in here
npx = np.random.uniform(-1, 1, (1000, 1))                           # x data
npy = np.power(npx, 2) + np.random.normal(0, 0.1, size=npx.shape)   # y data
npx_train, npx_test = np.split(npx, [800])                          # training and test data
npy_train, npy_test = np.split(npy, [800])

# use placeholder, later you may need different data, pass the different data into placeholder
tfx = tf.placeholder(npx_train.dtype, npx_train.shape)
tfy = tf.placeholder(npy_train.dtype, npy_train.shape)

# create dataloader
dataset = tf.data.Dataset.from_tensor_slices((tfx, tfy))
dataset = dataset.shuffle(buffer_size=1000)   # choose data randomly from this buffer
dataset = dataset.batch(32)                   # batch size you will use
dataset = dataset.repeat(3)                   # repeat for 3 epochs
iterator = dataset.make_initializable_iterator()  # later we have to initialize this one

# your network
bx, by = iterator.get_next()                  # use batch to update
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
out = tf.layers.dense(l1, npy.shape[1])
loss = tf.losses.mean_squared_error(by, out)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# need to initialize the iterator in this case
sess.run([iterator.initializer, tf.global_variables_initializer()], feed_dict={tfx: npx_train, tfy: npy_train})

for step in range(201):
    try:
        _, trainl = sess.run([train, loss])
        if step % 10 == 0:
            testl = sess.run(loss, {bx: npx_test, by: npy_test})    # test
            print('step: %i/200' % step, '|train loss:', trainl, '|test loss:', testl)
    except tf.errors.OutOfRangeError:     # if training takes more than 3 epochs, training will be stopped
        print('Finish the last epoch.')
        break
'''
step: 0/200 |train loss: 0.021361219 |test loss: 0.034942627
step: 10/200 |train loss: 0.0305059 |test loss: 0.03167681
step: 20/200 |train loss: 0.019857055 |test loss: 0.02710227
step: 30/200 |train loss: 0.024798539 |test loss: 0.025249688
step: 40/200 |train loss: 0.023114882 |test loss: 0.022836378
step: 50/200 |train loss: 0.011948485 |test loss: 0.023002312
step: 60/200 |train loss: 0.012579769 |test loss: 0.022071866
step: 70/200 |train loss: 0.01895435 |test loss: 0.019848838
Finish the last epoch.
'''