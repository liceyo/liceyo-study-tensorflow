import tensorflow as tf

# Create a variable with a random value.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")
# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 0.2, name="w_twice")

init_op = tf.global_variables_initializer()

with tf.Session as sess:
    sess.run(init_op)