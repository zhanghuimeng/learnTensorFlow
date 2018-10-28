import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


# Load MNIST data
class Dataloader():
    def __init__(self):
        mnist = np.load("../../data/mnist.npz")
        # Must use asarray to convert uint8 to float, or Dense complains
        self.train_data = np.ndarray.astype(mnist["x_train"], dtype=np.float32)  # [60000, 28, 28]
        self.train_labels = np.asarray(mnist["y_train"], dtype=np.int32) # 60000 unit8
        self.eval_data = np.ndarray.astype(mnist["x_test"], dtype=np.float32)  # [10000, 28, 28]
        self.eval_labels = np.asarray(mnist["y_test"], dtype=np.int32)  # 10000 unit8

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_labels[index]


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/
        # Tensor: 28 * 28
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积核的数目（即输出的维度）
            # 单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度
            kernel_size=[5, 5],
            # 补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，
            # 通常会导致输出shape与输入shape相同
            padding="same",
            activation=tf.nn.relu
        )
        # Tensor: 32 * 28 * 28
        # pool_size: 整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，
        # 如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        # Tensor: 32 * 14 * 14
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )
        # Tensor: 64 * 14 * 14
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        # Tensor: 64 * 7 * 7
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        # still has to reshape, because "1" means 1 channel
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])  # [batch_size, 28, 28, 1]
        x = self.conv1(inputs)  # [batch_size, 28, 28, 32]
        x = self.pool1(x)       # [batch_size, 14, 14, 32]
        x = self.conv2(x)       # [batch_size, 14, 14, 64]
        x = self.pool2(x)       # [batch_size, 7, 7, 64]
        x = self.flatten(x)     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)      # [batch_size, 1024]
        x = self.dense2(x)      # [batch_size, 10]
        return x

    def predict(self, inputs):
        logits = self(inputs)
        #  Describes which axis of the input Tensor to reduce across. For vectors, use axis = 0.
        return tf.argmax(logits, axis=-1)


# hyper-parameters
num_batches = 1200
batch_size = 50
learning_rate = 0.001

# Model and optimizer
model = CNN()
dataloader = Dataloader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
X_placeholder = tf.placeholder(name='X', shape=[None, 28, 28], dtype=tf.float32)
y_placeholder = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
y_logit_pred = model(X_placeholder)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_placeholder, logits=y_logit_pred)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    print("Training")
    sess.run(tf.global_variables_initializer())
    for batch_index in range(num_batches):
        X, y = dataloader.get_batch(batch_size)
        # sess.run可以传多个参数，这使得我们可以打印loss……
        _, lossVal = sess.run([train_op, loss], feed_dict={X_placeholder: X, y_placeholder: y})
        print("batch %d: loss %f" % (batch_index, lossVal))
    print("Training Finished\n")

    print("Evaluating")
    num_eval_examples = np.shape(dataloader.eval_data)[0]
    # y_pred = sess.run(model.predict(dataloader.eval_data)) # 内存爆炸了……
    # sess.run返回的是np array，因此在此处也踩了一些坑。
    y_pred = np.array([])
    num_eval_batches = int(num_eval_examples) // batch_size
    for batch_index in range(num_eval_batches):
        y_pred = np.append(y_pred, sess.run(model.predict(
            dataloader.eval_data[batch_index * batch_size: (batch_index + 1) * batch_size])))
        print("batch %d" % batch_index)
    print("Test Accuracy: %f" % (sum(y_pred == dataloader.eval_labels) / num_eval_examples))