import tensorflow as tf

graph1 = tf.Graph()
with graph1.as_default():
    weight_init = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    # If constant init, then don't specify dtype or shape
    weight = tf.get_variable('weight', initializer=weight_init)
    bias_init = tf.constant([7, 8, 9], dtype=tf.float32)
    bias = tf.get_variable('bias', initializer=bias_init)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        save_path = saver.save(sess, 'saved_graph/saved_net.ckpt')

graph2 = tf.Graph()
with graph2.as_default():
    weight_restored = tf.get_variable('weight', [3, 2], tf.float32)
    bias_restored = tf.get_variable('bias', [3], tf.float32)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
        print('weight', sess.run(weight_restored))
        print('bias', sess.run(bias_restored))
