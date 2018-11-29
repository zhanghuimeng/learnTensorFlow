import tensorflow as tf
import numpy as np


def np_get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


class MnistDataloader:
    def __init__(self):
        mnist = np.load('../data/mnist.npz')
        self.train_data = np.reshape(mnist['x_train'].astype(np.float32), [-1, 28*28])
        self.train_label = mnist['y_train'].astype(np.int32)
        self.train_label = np_get_one_hot(self.train_label, 10)
        self.test_data = np.reshape(mnist['x_test'].astype(np.float32), [-1, 28*28])
        self.test_label = mnist['y_test'].astype(np.int32)
        self.test_label = np_get_one_hot(self.test_label, 10)

    def get_batch(self, size):
        indices = np.random.randint(0, np.shape(self.train_data)[0], size)
        return self.train_data[indices], self.train_label[indices]


def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)
# prediction = add_layer(xs, 784, 10)

# the error between prediction and real data
# 这个计算从理论上是正确的，但实践中会导致nan的问题
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss

# 这么算的前提是，输出的是没做过softmax的logits
# 输出结果是比较正确的
# cross_entropy = tf.losses.softmax_cross_entropy(ys, prediction)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

dataloader = MnistDataloader()
for i in range(1000):
    batch_xs, batch_ys = dataloader.get_batch(100)
    # if i == 0:
    #     print(batch_xs)
    #     print(batch_ys)
    loss, _ = sess.run([cross_entropy, train_step], feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print('step %d loss=%f' % (i, loss))
        print(compute_accuracy(
            dataloader.test_data, dataloader.test_label))
