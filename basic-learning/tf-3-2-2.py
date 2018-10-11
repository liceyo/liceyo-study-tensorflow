import tensorflow as tf

# 保存变量
# Create some variables.
v1 = tf.Variable(10, name="v1")
v2 = tf.Variable(20, name="v2")
# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
# 如果要保存全部变量，不加参数就行了
saver = tf.train.Saver({"my_v2": v2})

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)
    # Do some work with the model.
    # Save the variables to disk.
    save_path = saver.save(sess, "tf-3-2-2/model.ckpt")
    print("Model saved in file: ", save_path)
