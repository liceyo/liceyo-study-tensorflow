import tensorflow as tf

filename_queue = tf.train.string_input_producer(["files/file0.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
# ['']解析为string类型 ，[1]为整型，[1.0]解析为浮点
record_defaults = [[1], [1], [1], [''], [1.0]]
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
# 顺序
# example_batch, label_batch = tf.train.batch([example, label], batch_size=1, capacity=200, num_threads=2)#保证样本和标签一一对应
# 无序
example_batch, label_batch = tf.train.shuffle_batch([col4, col5], batch_size=1, capacity=200, min_after_dequeue=100,
                                                    num_threads=2)  # 保证样本和标签一一对应
with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        for i in range(10):
            if coord.should_stop():
                break
            example, label = sess.run([example_batch, label_batch])
            print(example, label)
    except Exception as e:
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads)
