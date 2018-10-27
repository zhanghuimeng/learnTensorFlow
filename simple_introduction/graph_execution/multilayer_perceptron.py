import tensorflow as tf
import numpy as np


# Load MNIST data
class Dataloader():
    def __init__(self):
        mnist = np.load("../../data/mnist.npz")
        # 必须把uint8图片数据转换为float32类型，否则Dense层会报错
        self.train_data = np.ndarray.astype(mnist["x_train"], dtype=np.float32)  # [60000, 28, 28]
        self.train_labels = np.asarray(mnist["y_train"], dtype=np.int32) # 60000 unit8
        self.eval_data = np.ndarray.astype(mnist["x_test"], dtype=np.float32)  # [10000, 28, 28]
        self.eval_labels = np.asarray(mnist["y_test"], dtype=np.int32)  # 10000 unit8

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_labels[index]


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)
        # units指的是输出维度（输入维度是动态确定的==）
        # input shape: (batch_size, ..., input_dim)
        # output shape: (batch_size, ..., units)

    def call(self, inputs):
        # 把输入图片拉直成一维向量（多维似乎是会bug的）
        x = tf.reshape(inputs, [-1, 28*28])
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    # 选择概率最大的数字进行预测输出
    def predict(self, inputs):
        logits = self(inputs)
        return tf.argmax(logits, axis=-1)


# hyper-parameters
num_batches = 1200
batch_size = 50
learning_rate = 0.001

# Model and optimizer
model = MLP()
dataloader = Dataloader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
X_placeholder = tf.placeholder(name='X', shape=[None, 28, 28], dtype=tf.float32)
y_placeholder = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
y_logit_pred = model(X_placeholder)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_placeholder, logits=y_logit_pred)
train_op = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch_index in range(num_batches):
        X, y = dataloader.get_batch(batch_size)
        sess.run(train_op, feed_dict={X_placeholder: X, y_placeholder: y})

    num_eval_examples = np.shape(dataloader.eval_data)[0]
    y_pred = sess.run(model.predict(dataloader.eval_data))  # 好像就直接换成sess.run()就可以了……？
    # 大概sess.run可以获知运算结果吧
    print("Test Accuracy: %f" % (sum(y_pred == dataloader.eval_labels) / num_eval_examples))