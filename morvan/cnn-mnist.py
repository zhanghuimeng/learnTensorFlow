import tensorflow as tf
import numpy as np


class MnistDataLoader:
    def __init__(self):
        mnist = np.load('../data/mnist.npz')
        # size * width * height * channels
        self.train_data = np.reshape(mnist['x_train'].astype(np.float32), [-1, 28, 28, 1])
        self.train_label = mnist['y_train'].astype(np.int32)
        self.test_data = np.reshape(mnist['x_test'].astype(np.float32), [-1, 28, 28, 1])
        self.test_label = mnist['y_test'].astype(np.int32)

    def get_batch(self, size):
        indices = np.random.randint(0, np.shape(self.train_data)[0], size)
        return self.train_data[indices], self.train_label[indices]


histogram_summaries = []


def get_weight_variable(size):
    weight = tf.get_variable('weight', size, dtype=tf.float32, initializer=tf.truncated_normal_initializer())
    histogram_summaries.append(tf.summary.histogram('weight', weight))
    return weight


def get_bias_variable(size):
    bias = tf.get_variable('bias', size, dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    histogram_summaries.append(tf.summary.histogram('bias', bias))
    return bias


def conv2d(x, weight):
    return tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME')


def max_pooling(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')


def add_linear_layer(x, in_size, out_size, activation_function=None):
    weight = get_weight_variable([in_size, out_size])
    bias = get_bias_variable([out_size])
    output = tf.matmul(x, weight) + bias  # * means element-wise mul. Should use matmul.
    if activation_function is not None:
        output = activation_function(output)
    return output


x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='data_input')
y = tf.placeholder(tf.int32, shape=[None], name='data_label')

with tf.variable_scope('conv2d-layer-1'):
    weight1 = get_weight_variable([5, 5, 1, 32])
    bias1 = get_bias_variable([32])
    conv1 = conv2d(x, weight1) + bias1  # bs*28*28*32
    conv1 = tf.nn.relu(conv1)

with tf.variable_scope('max-pooling-layer-1'):
    maxpool1 = max_pooling(conv1)  # bs*14*14*32

with tf.variable_scope('conv2d-layer-2'):
    weight2 = get_weight_variable([5, 5, 32, 64])
    bias2 = get_bias_variable([64])
    conv2 = conv2d(maxpool1, weight2) + bias2  # bs*14*14*64
    conv2 = tf.nn.relu(conv2)

with tf.variable_scope('max-pooling-layer-2'):
    maxpool2 = max_pooling(conv2)  # bs*7*7*64

with tf.variable_scope('feedforward-layer-1'):
    linear1 = add_linear_layer(tf.reshape(maxpool2, [-1, 7*7*64]), 7*7*64, 1024, tf.nn.relu)

with tf.variable_scope('feedforward-layer-2'):
    linear2 = add_linear_layer(linear1, 1024, 10, None)

# 我发现我之前没有想column和分类表不同的问题
with tf.name_scope('training'):
    loss = tf.losses.sparse_softmax_cross_entropy(y, linear2)
    loss_summary = tf.summary.scalar('loss', loss)
    train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

with tf.name_scope('prediction'):
    prediction = tf.argmax(tf.nn.softmax(linear2), axis=-1, output_type=tf.int32)
    correct_prediction = tf.equal(prediction, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_summary = tf.summary.scalar('accuracy', accuracy)

dataLoader = MnistDataLoader()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)  # So as to see the graph
    test_writer = tf.summary.FileWriter('logs/test')
    merged_train_summary = tf.summary.merge([loss_summary] + histogram_summaries)
    merged_test_summary = tf.summary.merge([acc_summary])
    for i in range(0, 500):
        tx, ty = dataLoader.get_batch(500)
        loss_val, _ = sess.run([loss, train_op], feed_dict={x: tx, y: ty})
        print('Step %d: loss=%f' % (i, loss_val))
        train_summary_val = sess.run(merged_train_summary,
                                     feed_dict={x: tx, y: ty})
        test_summary_val = sess.run(merged_test_summary,
                                    feed_dict={x: dataLoader.test_data, y: dataLoader.test_label})
        train_writer.add_summary(train_summary_val, i)
        test_writer.add_summary(test_summary_val, i)
        train_writer.flush()
        test_writer.flush()

    train_writer.close()
    test_writer.close()