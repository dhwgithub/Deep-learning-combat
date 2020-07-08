import tensorflow as tf

state = tf.Variable(0, name='counter')  # 变量
print(state.name)  # counter:0

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()  # must have if have variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print((sess.run(state)))

'''
1
2
3
'''