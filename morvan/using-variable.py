import tensorflow as tf

counter = tf.Variable(0, name='counter')
one = tf.constant(1)
added = counter + one
op = tf.assign(counter, added)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(op)
        print(sess.run(counter))
