import tensorflow as tf
import numpy as np
tf.enable_eager_execution()


# Lesson
# 1. Don't blame eager execution and Keras.
# 2. Carefully consider input type and shpe
# 3. Understand what the funcions are saying


# Load MNIST data
class Dataloader():
    def __init__(self):
        mnist = np.load("../data/mnist.npz")
        # 必须把uint8图片数据转换为float32类型，否则Dense层会报错
        self.train_data = np.ndarray.astype(mnist["x_train"], dtype=np.float32)  # [60000, 28, 28]
        self.train_labels = np.asarray(mnist["y_train"], dtype=np.int32) # 60000 unit8
        self.eval_data = np.ndarray.astype(mnist["x_test"], dtype=np.float32)  # [10000, 28, 28]
        self.eval_labels = np.asarray(mnist["y_test"], dtype=np.int32)  # 10000 unit8
        print(np.shape(self.train_data))

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_labels[index]


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)
        # units指的是输出维度（输入维度是动态确定的？）
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

# Feed batches of data into the Model, calc loss, and update Model
for batch_index in range(num_batches):
    X, y = dataloader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        X = tf.convert_to_tensor(X)
        y_logit_pred = model(X)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

num_eval_examples = np.shape(dataloader.eval_data)[0]
y_pred = model.predict(dataloader.eval_data).numpy()
print("Test Accuracy: %f" % (sum(y_pred == dataloader.eval_labels) / num_eval_examples))