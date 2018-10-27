import tensorflow as tf
tf.enable_eager_execution()

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

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
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)  # call the model
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables)