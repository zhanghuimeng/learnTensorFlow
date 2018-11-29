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
        bias = tf.get_variable('bias', [1, out_size], initializer=tf.constant_initializer(0.1))
        outputs = tf.matmul(inputs, weights) + bias
        if not activation_function is None:
            outputs = activation_function(outputs)
        return outputs


class MnistDataloader:
    def __init__(self):
        mnist = np.load('../data/mnist.npz')
        self.train_data = np.reshape(mnist['x_train'].astype(np.float32), [-1, 28*28])
        self.train_label = mnist['y_train'].astype(np.int32)
        self.test_data = np.reshape(mnist['x_test'].astype(np.float32), [-1, 28 * 28])
        self.test_label = mnist['y_test'].astype(np.int32)

    def get_batch(self, size):
        indices = np.random.randint(0, np.shape(self.train_data)[0], size)
        return self.train_data[indices], self.train_label[indices]


dataLoader = MnistDataloader()
data_placeholder = tf.placeholder(tf.float32, [None, 28*28], 'data')
label_placeholder = tf.placeholder(tf.int32, [None], 'label')  # The data is not one-hot
output = add_layer('layer1', data_placeholder, 28*28, 10, None)  # Don't apply softmax twice!
# 不要用两次softmax！tf的这个函数会自己加softmax
# 删除多余的一层softmax之后结果就正常了
loss = tf.losses.sparse_softmax_cross_entropy(label_placeholder, output)  # Why this API?
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
# Testing
prediction = tf.argmax(output, axis=-1, output_type=tf.int32)
correct_prediction = tf.equal(prediction, label_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 虽然我觉得有点奇怪，这样混用一张计算流图真的对吗？

initializer = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializer)
    print('Training...')
    for i in range(1, 1000):
        data, label = dataLoader.get_batch(100)
        lossVal, _ = sess.run([loss, train_op],
                              feed_dict={data_placeholder: data, label_placeholder: label})
        if i % 50 == 0:
            print("step %d: loss=%f" % (i, lossVal))
            accVal = sess.run(accuracy,
                              feed_dict={data_placeholder: dataLoader.test_data,
                                         label_placeholder: dataLoader.test_label})
            print("step %d: acc on eval set=%f" % (i, accVal))

