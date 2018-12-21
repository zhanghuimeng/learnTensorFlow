import tensorflow as tf
import numpy as np


# Load Mnist Data from file, and batching
class MnistDataLoader:
    def __init__(self):
        mnist = np.load('../data/mnist.npz')
        # size * width * height * channels
        self.train_data = np.reshape(mnist['x_train'].astype(np.float32), [-1, 28, 28])
        self.train_label = mnist['y_train'].astype(np.int32)
        self.test_data = np.reshape(mnist['x_test'].astype(np.float32), [-1, 28, 28])
        self.test_label = mnist['y_test'].astype(np.int32)

    def get_batch(self, size):
        indices = np.random.randint(0, np.shape(self.train_data)[0], size)
        return self.train_data[indices], self.train_label[indices]


class RNNClassifier:
    def __init__(self, batch_size, time_steps, input_size, hidden_size, state_size, output_size, lr):
        """
        Init function, building up the whole graph.
        :param batch_size:
        :param time_steps:
        :param input_size:
        :param hidden_size:
        :param state_size:
        :param output_size:
        """
        # Construct input placeholder
        with tf.name_scope('inputs'):
            self.x_placeholder = tf.placeholder(tf.float32, [None, time_steps, input_size], 'x')
            self.y_placeholder = tf.placeholder(tf.int32, [None], 'y')
            self.x = self.x_placeholder
            self.y = self.y_placeholder
        # Add input layer
        # x = [batch_size, time_steps, input_size] ====> [batch_size, time_steps, hidden_size]
        with tf.variable_scope('input_linear_layer'):
            self.x = self.add_linear_layer(self.x, [batch_size, time_steps, input_size], input_size, hidden_size)
        # Add RNN layer
        # outputs = [batch_size, time_steps, state_size]
        # state = [batch_size, state_size]
        with tf.variable_scope('LSTM'):
            outputs, state = self.add_lstm_layer(self.x, batch_size, state_size)
            self.x = state[1]
        # Add output layer
        # x = [batch_size, state_size] ====> [batch_size, output_size]
        with tf.variable_scope('output_linear_layer'):
            self.x = self.add_linear_layer(self.x, [batch_size, state_size], state_size, output_size)
        with tf.name_scope('loss'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(logits=self.x, labels=self.y)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def add_linear_layer(self, x, x_shape, in_size, out_size, activation_function=None):
        weight = self.get_weight_variable([in_size, out_size])
        bias = self.get_bias_variable([out_size])
        # Patching up tf.matmul
        if len(x_shape) > 2:
            x = tf.reshape(x, [-1, in_size])
        output = tf.matmul(x, weight) + bias
        if len(x_shape) > 2:
            output = tf.reshape(output, x_shape[:-1] + [out_size])
        if activation_function is not None:
            output = activation_function(output)
        return output

    @staticmethod
    def add_lstm_layer(x, batch_size, state_size):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(state_size)
        initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        return tf.nn.dynamic_rnn(lstm_cell, x, initial_state=initial_state, dtype=tf.float32)

    @staticmethod
    def get_weight_variable(size):
        weight = tf.get_variable('weight', shape=size, initializer=tf.random_normal_initializer())
        return weight

    @staticmethod
    def get_bias_variable(size):
        bias = tf.get_variable('bias', shape=size, initializer=tf.constant_initializer(0.1))
        return bias


data_loader = MnistDataLoader()
model = RNNClassifier(500, 28, 28, 128, 128, 10, 1e-3)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        xd, yd = data_loader.get_batch(500)
        # Don't use same names for placeholder & input
        loss, _ = sess.run([model.loss, model.train_op],
                           feed_dict={model.x_placeholder: xd, model.y_placeholder: yd})
        print("Step %d: loss=%f" % (i, loss))
