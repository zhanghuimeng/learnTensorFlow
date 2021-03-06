import tensorflow as tf
import numpy as np

x = np.random.rand(100).astype(np.float32)  # 否则它默认会用float64；然后默认用float32的tf就会报错
y_expected = x * 0.1 + 0.3

# 一种初始化方式
# weight = tf.Variable(tf.random_uniform([1], -1, 1))
# bias = tf.Variable(tf.zeros([1]))
# 另一种初始化方式
weight = tf.get_variable('w', [1], initializer=tf.random_uniform_initializer(-1, 1))
bias = tf.get_variable('b', [1], initializer=tf.zeros_initializer())
y_calculated = x * weight + bias
loss = tf.reduce_mean(tf.square(y_calculated - y_expected))  # reduce_mean和square结合使用==
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)
initializer = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializer)
    for i in range(201):
        sess.run(train_op)
        if i % 20 == 0:
            print("The %dth training: " % i)
            print("weight = ", sess.run(weight))  # 需要执行一步sess.run才能打印
            print("bias = ", sess.run(bias))