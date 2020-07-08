'''
name_scope：
    对于get_variable方法，其输出名称时不会携带scope的前缀
    对于Variable方法，其输出名称时可以携带scope的前缀

variable_scope：
    对于get_variable方法和Variable方法，其输出名称时都可以携带scope的前缀
    当前面调用reuse_variables方法时，可以重复利用已存在的变量
    否则对于相同的变量会重新创建，如var1、var_1、var_2等
'''
import tensorflow as tf

with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)

    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(var1.name)        # var1:0
    print(sess.run(var1))   # [ 1.]

    print(var2.name)        # a_name_scope/var2:0
    print(sess.run(var2))   # [ 2.]

    print(var21.name)       # a_name_scope/var2_1:0
    print(sess.run(var21))  # [2.1]

    print(var22.name)       # a_name_scope/var2_2:0
    print(sess.run(var22))  # [2.2]


with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)

    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)

    scope.reuse_variables()
    var3_reuse = tf.get_variable(name='var3',)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    print(var3.name)            # a_variable_scope/var3:0
    print(sess.run(var3))       # [ 3.]

    print(var4.name)            # a_variable_scope/var4:0
    print(sess.run(var4))       # [ 4.]

    print(var4_reuse.name)      # a_variable_scope/var4_1:0
    print(sess.run(var4_reuse)) # [ 4.]

    print(var3_reuse.name)      # a_variable_scope/var3:0
    print(sess.run(var3_reuse)) # [ 3.]
