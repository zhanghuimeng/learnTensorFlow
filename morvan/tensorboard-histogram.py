import tensorflow as tf
import numpy as np


def add_layer(scope, inputs, in_size, out_size, activation_function=None):
    '''Add a linear activation layer.
    :param inputs: The input data.
    :param in_size: The size of the input data.
    :param out_size: The size of the output data.
    :param activation_function: Activation function applied to the linear output.
    :return: Output data.
    '''
    # 注意这是个矩阵
    with tf.variable_scope(scope, reuse=False):
        weights = tf.get_variable('weights', [in_size, out_size], initializer=tf.random_normal_initializer(0.1))
        tf.summary.histogram('weights', weights)  # write weights summary to HISTOGRAMS
        bias = tf.get_variable('bias', [1, out_size], initializer=tf.constant_initializer(0.05))
        tf.summary.histogram('bias', bias)  # write bias summary
        outputs = tf.matmul(inputs, weights) + bias
        if not activation_function is None:
            outputs = activation_function(outputs)
        tf.summary.histogram('outputs', outputs)  # write outputs summary
        return outputs


x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]  # uniform points
noise = np.random.normal(0, 0.05, (300, 1))
y_data = np.square(x_data) - 0.5 + noise

# Create graph
with tf.name_scope('inputs'):
    x_placeholder = tf.placeholder(tf.float32, [None, 1], 'x')
    y_placeholder = tf.placeholder(tf.float32, [None, 1], 'y')

layer1_output = add_layer('layer_1', x_placeholder, 1, 10, tf.nn.relu)
y_output = add_layer('layer_2', layer1_output, 10, 1, None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_output - y_placeholder), name='loss')
    tf.summary.scalar('loss', loss) # write loss summary to EVENTS

with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()

# Training
with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/', sess.graph)
    merged = tf.summary.merge_all()  # merge all summary
    sess.run(init)
    for i in range(0, 500):
        _, loss_val = sess.run([train_op, loss], feed_dict={x_placeholder: x_data, y_placeholder: y_data})
        if i % 20 == 0:
            print("Step %d, loss = %f" % (i, loss_val))
            rs = sess.run(merged, feed_dict={x_placeholder: x_data, y_placeholder: y_data})  # run to get summary
            writer.add_summary(rs, i)  # write summary

    y_prediction = sess.run(y_output, feed_dict={x_placeholder: x_data})
