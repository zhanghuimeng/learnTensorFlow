import tensorflow as tf

# feed_dict不接受tensor作为输入
X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
y = [[10.0], [20.0]]

# 将模型抽象为类的形式
class Linear(tf.keras.Model):
    # init：构造函数
    def __init__(self):
        super().__init__()
        # Add layers
        self.dense = tf.keras.layers.Dense(units=1, kernel_initializer=tf.zeros_initializer,
                                           bias_initializer=tf.zeros_initializer)
        # Dense Layer: output = activation(tf.matmul(input, kernel) + bias)
        # 单个unit的全连接层等价于单变量的线性变换

    # 模型调用方法
    def call(self, input):
        # 处理输入，返回输出
        output = self.dense(input)
        return output


model = Linear()
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01)
X_placeholder = tf.placeholder(name='X', shape=[None, 3], dtype=tf.float32)
y_placeholder = tf.placeholder(name='y', shape=[None, 1], dtype=tf.float32)
y_pred = model(X_placeholder)
loss = tf.reduce_mean(tf.square(y_pred - y_placeholder))
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train_op, feed_dict={X_placeholder: X, y_placeholder: y})
    print(sess.run(model.variables)) # ?