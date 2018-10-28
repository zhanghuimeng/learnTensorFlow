import tensorflow as tf
import numpy as np
tf.enable_eager_execution()


# Load MNIST data
class Dataloader():
    def __init__(self):
        mnist = np.load("../../data/mnist.npz")
        # Must use asarray to convert uint8 to float, or Dense complains
        self.train_data = np.ndarray.astype(mnist["x_train"], dtype=np.float32)  # [60000, 28, 28]
        self.train_labels = np.asarray(mnist["y_train"], dtype=np.int32) # 60000 unit8
        self.eval_data = np.ndarray.astype(mnist["x_test"], dtype=np.float32)  # [10000, 28, 28]
        self.eval_labels = np.asarray(mnist["y_test"], dtype=np.int32)  # 10000 unit8
        # print(np.shape(self.train_data))

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

# Feed batches of data into the Model, calc loss, and update Model
for batch_index in range(num_batches):
    X, y = dataloader.get_batch(batch_size)
    # print(np.shape(X))
    # print(np.shape(y))
    with tf.GradientTape() as tape:
        X = tf.convert_to_tensor(X)
        y_logit_pred = model(X)
        # labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result) and dtype int32 or
        # int64. Each entry in labels must be an index in [0, num_classes).
        # logits: Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1}, num_classes] and dtype float16,
        # float32 or float64.
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

num_eval_examples = np.shape(dataloader.eval_data)[0]
y_pred = model.predict(dataloader.eval_data).numpy()
print("Test Accuracy: %f" % (sum(y_pred == dataloader.eval_labels) / num_eval_examples))