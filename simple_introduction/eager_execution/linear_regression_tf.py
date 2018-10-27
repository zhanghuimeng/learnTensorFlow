import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

# y = ax + b

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X = tf.constant(X)
y = tf.constant(y)

# Shape is simply []; initializer is zero
a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
variables = [a, b]

num_epoch = 10000  # 训练10000轮
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)  # 优化器，可根据梯度和学习率自动更新参数
for e in range(num_epoch):
    # Use tf.GradientTape() to record the gradient value of L(a, b)
    with tf.GradientTape() as tape:
        y_pred = X * a + b  # This is model
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # 自动计算L关于a和b的梯度
    grads = tape.gradient(loss, variables)
    # Auto update a, b according to L
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)