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
        weights = tf.get_variable('weights', [in_size, out_size], initializer=tf.random_normal_initializer(0.0))
        weights_summary = tf.summary.histogram('weights', weights)
        bias = tf.get_variable('bias', [1, out_size], initializer=tf.constant_initializer(0.1))
        bias_summary = tf.summary.histogram('bias', bias)
        outputs = tf.matmul(inputs, weights) + bias
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)  # dropout!
        if activation_function is not None:
            outputs = activation_function(outputs)
        return outputs


class MnistDataloader:
    def __init__(self):
        mnist = np.load('../data/mnist.npz')
        self.train_data = np.reshape(mnist['x_train'].astype(np.float32), [-1, 28 * 28])
        self.train_label = mnist['y_train'].astype(np.int32)
        self.test_data = np.reshape(mnist['x_test'].astype(np.float32), [-1, 28 * 28])
        self.test_label = mnist['y_test'].astype(np.int32)

    def get_batch(self, size):
        indices = np.random.randint(0, np.shape(self.train_data)[0], size)
        return self.train_data[indices], self.train_label[indices]


# 我注意到，一张计算流图的方法没法做到跑一次就算出所有的summary……
# 但是也可以一次干脆进行训练和compute acc
# 因为计算流图看起来很聪明，它们不会多计算东西的。
# 问题是要分两个不同的writer, 写到两个不同的目录下。

dataLoader = MnistDataloader()
with tf.name_scope('inputs'):
    data_placeholder = tf.placeholder(tf.float32, [None, 28 * 28], 'data')
    label_placeholder = tf.placeholder(tf.int32, [None], 'label')  # The data is not one-hot
    keep_prob = tf.placeholder(tf.float32, None, 'keep_prob')
output = add_layer('layer1', data_placeholder, 28 * 28, 10, None)  # Don't apply softmax twice!
with tf.name_scope('loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(label_placeholder, output)  # Why this API?
    loss_summary = tf.summary.scalar('loss', loss)
# train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
# Testing
with tf.name_scope('prediction'):
    prediction = tf.argmax(tf.nn.softmax(output), axis=-1, output_type=tf.int32)
    correct_prediction = tf.equal(prediction, label_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    prediction_summary = tf.summary.scalar('prediction_accuracy', accuracy)

initializer = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializer)
    training_writer = tf.summary.FileWriter('logs/train', sess.graph)
    testing_writer = tf.summary.FileWriter('logs/test')  # graph not necessary
    merged_training = tf.summary.merge_all()
    merged_testing = tf.summary.merge([prediction_summary])
    print('Training...')
    for i in range(1000):
        data, label = dataLoader.train_data, dataLoader.train_label
        lossVal, _ = sess.run([loss, train_op],
                              feed_dict={data_placeholder: data, label_placeholder: label, keep_prob: 0.5})
        if i % 1 == 0:
            print("step %d: loss=%f" % (i, lossVal))
            accVal = sess.run(accuracy,
                              feed_dict={data_placeholder: dataLoader.test_data,
                                         label_placeholder: dataLoader.test_label,
                                         keep_prob: 1})
            print("step %d: acc on eval set=%f" % (i, accVal))
            training_summary_val = sess.run(merged_training,
                                            feed_dict={data_placeholder: dataLoader.train_data,
                                                       label_placeholder: dataLoader.train_label,
                                                       keep_prob: 1})
            training_writer.add_summary(training_summary_val, i)
            test_summary_val = sess.run(merged_testing,
                                        feed_dict={data_placeholder: dataLoader.test_data,
                                                   label_placeholder: dataLoader.test_label,
                                                   keep_prob: 1})
            testing_writer.add_summary(test_summary_val, i)
            training_writer.flush()
            testing_writer.flush()

training_writer.close()
testing_writer.close()
