# 测试可视化

import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0], name='input1')
b = tf.Variable(tf.random_uniform([3]), name='input2')
add = tf.add_n([a, b], name='addOP')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with tf.summary.FileWriter("TensorBoard", sess.graph) as writer:
        print(sess.run(add))
    # writer = tf.summary.FileWriter("TensorBoard", sess.graph)

# writer.close()
