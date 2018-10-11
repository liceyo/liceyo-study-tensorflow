import tensorflow as tf

#  标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
input1 = tf.placeholder(tf.float32, shape=[3, 2])
input2 = tf.placeholder(tf.float32, shape=[2, 3])
output = tf.matmul(input1, input2)
with tf.Session() as sess:
    print(sess.run([output],
                   feed_dict={
                       input1: [[1., 4.], [2., 5.], [3., 6.]],
                       input2: [[1., 2., 3.], [4., 5., 6.]]
                   }))
