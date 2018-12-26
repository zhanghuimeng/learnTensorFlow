import tensorflow as tf
import numpy as np


# Load Mnist Data from file, and batching
class MnistDataLoader:
    def __init__(self):
        mnist = np.load('../data/mnist.npz')
        self.train_data = np.reshape(mnist['x_train'].astype(np.float32), [-1, 28, 28])
        self.train_label = mnist['y_train'].astype(np.int32)
        self.test_data = np.reshape(mnist['x_test'].astype(np.float32), [-1, 28, 28])
        self.test_label = mnist['y_test'].astype(np.int32)

    def get_batch(self, size):
        indices = np.random.randint(0, np.shape(self.train_data)[0], size)
        return self.train_data[indices], self.train_label[indices]


# The Model
class RNNClassifier:
    def __init__(self, time_steps, input_size, hidden_size, state_size, output_size, lr):
        """
        Init function, building up the whole graph.
        :param time_steps: The length (of time) of the input. 28 for MNIST (28 lines).
        :param input_size: The size of input at each time point. 28 for MNIST (28 columns).
        :param hidden_size: The size of LSTMCell input.
        :param state_size: The size of LSTMCell state.
        :param output_size: The size of output hidden layer. 10 for MNIST (10 classes).
        """
        self.train_summaries = []
        # Construct input placeholder
        with tf.name_scope('inputs'):
            # Use special names to avoid feed_dict from shadowing other Tensors
            self.x_placeholder = tf.placeholder(tf.float32, [None, time_steps, input_size], 'x')
            self.y_placeholder = tf.placeholder(tf.int32, [None], 'y')
            self.x = self.x_placeholder
            self.y = self.y_placeholder
        batch_size = tf.shape(self.x)[0]  # Patch for different batch sizes (using placeholder is also ok)
        # Add input layer
        # x = [batch_size, time_steps, input_size] ====> [batch_size, time_steps, hidden_size]
        with tf.variable_scope('input_linear_layer'):
            self.x = self.add_linear_layer(self.x, [batch_size, time_steps, input_size], input_size, hidden_size)
        # Add RNN layer
        # outputs = [batch_size, time_steps, state_size]
        # state = (c_state, m_state)
        # state[1] = [batch_size, state_size]
        with tf.variable_scope('LSTM'):
            outputs, state = self.add_lstm_layer(self.x, batch_size, state_size)
            print(tf.shape(state))
            self.x = state[1]
        # Add output layer
        # x = [batch_size, state_size] ====> [batch_size, output_size]
        with tf.variable_scope('output_linear_layer'):
            self.x = self.add_linear_layer(self.x, [batch_size, state_size], state_size, output_size)
        with tf.name_scope('test'):
            # I forgot to add a layer of softmax (but should get the same result anyway)
            pred = tf.argmax(self.x, axis=-1, output_type=tf.int32)
            correct = tf.equal(pred, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            self.test_summary = tf.summary.merge([tf.summary.scalar('accuracy', self.accuracy)])
        with tf.name_scope('loss'):  # Still single scalar loss
            self.loss = tf.losses.sparse_softmax_cross_entropy(logits=self.x, labels=self.y)
            self.train_summaries.append(tf.summary.scalar('loss', self.loss))
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
            self.train_summary = tf.summary.merge(self.train_summaries)

    def add_linear_layer(self, x, x_shape, in_size, out_size, activation_function=None):
        """
        Add a linear layer to the graph.
        :param x: input tensor.
        :param x_shape: the shape of x (to do rank>=3 multiply).
        :param in_size: the last dimension of x.
        :param out_size: the last dimension of output.
        :param activation_function: applied to output (if not None).
        :return: output tensor.
        """
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

    def add_lstm_layer(self, x, batch_size, state_size):
        """
        Add a LSTM layer to the graph.
        :param x: input tensor.
        :param batch_size: concrete value of batch size.
        :param state_size: size of cell state for LSTM cell.
        :return: (outputs, state) tuple.
        """
        lstm_cell = tf.nn.rnn_cell.LSTMCell(state_size)  # The cell state size
        # [batch_size, state_size] of zeros
        # NOTE: to run RNN, you need the concrete value of batch_size
        initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        tup = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=initial_state, dtype=tf.float32)
        for variable in lstm_cell.variables:
            self.train_summaries.append(tf.summary.histogram(variable.name, variable))
        return tup

    def get_weight_variable(self, shape):
        """
        Define a weight variable and its initializer.
        :param shape: the shape of the variable.
        :return: the weight variable.
        """
        weight = tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer())
        self.train_summaries.append(tf.summary.histogram('weight', weight))
        return weight

    def get_bias_variable(self, shape):
        """
        Define a bias variable and its initializer.
        :param shape: the shape of the variable.
        :return: the weight variable.
        """
        bias = tf.get_variable('bias', shape=shape, initializer=tf.constant_initializer(0.1))
        self.train_summaries.append(tf.summary.histogram('bias', bias))
        return bias


data_loader = MnistDataLoader()
model = RNNClassifier(28, 28, 128, 128, 10, 1e-3)
with tf.Session() as sess:
    trainWriter = tf.summary.FileWriter('logs/train', sess.graph)
    testWriter = tf.summary.FileWriter('logs/test')
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        xd, yd = data_loader.get_batch(500)
        # Don't use same names for placeholder & input
        loss, _, train_summary = sess.run([model.loss, model.train_op, model.train_summary],
                                          feed_dict={model.x_placeholder: xd, model.y_placeholder: yd})
        acc, test_summary = sess.run([model.accuracy, model.test_summary],
                                     feed_dict={model.x_placeholder: data_loader.test_data,
                                                model.y_placeholder: data_loader.test_label})
        print("Step %d: loss=%f, acc=%f" % (i, loss, acc))
        trainWriter.add_summary(train_summary, i)
        testWriter.add_summary(test_summary, i)
        trainWriter.flush()
        testWriter.flush()
    trainWriter.close()
    testWriter.close()
