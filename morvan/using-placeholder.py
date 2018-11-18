import tensorflow as tf

x = tf.placeholder(tf.float32, [1])
y = tf.placeholder(tf.float32, [1])
product = tf.multiply(x, y)

with tf.Session() as sess:
    print(sess.run(product, feed_dict={x: [4.], y: [5.]}))
