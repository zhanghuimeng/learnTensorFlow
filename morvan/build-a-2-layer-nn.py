import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


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
        bias = tf.get_variable('bias', [1, out_size], initializer=tf.constant_initializer(0.05))
        outputs = tf.matmul(inputs, weights) + bias
        if not activation_function is None:
            outputs = activation_function(outputs)
        return outputs


x_data = np.random.uniform(-3, 3, [300, 1])
noise = np.random.normal(0, 0.1, [300, 1])
y_data = np.square(x_data) + 0.5 + noise

# Create graph
# x和y的每个数据都只有1维，所以第二维是1
x_placeholder = tf.placeholder(tf.float32, [None, 1], 'x')
y_placeholder = tf.placeholder(tf.float32, [None, 1], 'y')

layer1_output = add_layer('layer_1', x_placeholder, 1, 10, tf.nn.relu)
y_output = add_layer('layer_2', layer1_output, 10, 1, None)
loss = tf.reduce_mean(tf.square(y_output - y_placeholder))  # (R)MSE
# learning rate is usually < 1
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()

# Training
with tf.Session() as sess:
    sess.run(init)
    for i in range(0, 500):
        _, loss_val = sess.run([train_op, loss], feed_dict={x_placeholder: x_data, y_placeholder: y_data})
        if i % 20 == 0:
            print("Step %d, loss = %f" % (i, loss_val))

    y_prediction = sess.run(y_output, feed_dict={x_placeholder: x_data})

plt.figure()
plt.title("x and y")
plt.ylabel('y')
plt.xlabel('x')
plt.scatter(x_data, y_data, color='blue')
plt.scatter(x_data, y_prediction, color='red')
plt.show()