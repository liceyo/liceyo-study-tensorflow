# 恢复变量
import tensorflow as tf

# Create some variables.
v1 = tf.Variable(2, name="v1")
v2 = tf.Variable(4, name="v2")
mul = tf.multiply(v1, v2)
# Add ops to save and restore all the variables.
saver = tf.train.Saver({"my_v2": v2})

init_op = tf.global_variables_initializer()
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    sess.run(init_op)
    # Restore variables from disk.
    saver.restore(sess, "tf-3-2-2/model.ckpt")
    print("Model restored.")
    print(sess.run(mul))
    # Do some work with the model
