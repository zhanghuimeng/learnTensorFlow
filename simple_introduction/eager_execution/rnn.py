import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


class Dataloader():
    def __init__(self):
        with open("../../data/nietzsche.txt", encoding="utf-8") as f:
            self.raw_text = f.read().lower()
        # 将字符转化为数字
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        # 每组测试数据：用seq_length个字符预测下一个字符
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index: index + seq_length])
            next_char.append(self.text[index + seq_length])
        return np.array(seq), np.array(next_char)  # [num_batch, seq_length], [num_batch]


# 首先对序列进行one hot操作：把编码i变换为一个n维向量，第i位为1，其他位均为0
# [num_batch, seq_length] -> [num_batch, seq_length, num_chars]
# 然后把t时刻的序列送入RNN单元
class RNN(tf.keras.Model):
    def __init__(self, num_chars):
        super().__init__()
        self.num_chars = num_chars
        # BasicLSTMCell is deprecated
        # num_units: int, The number of units in the LSTM cell.
        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs):
        batch_size, seq_length = tf.shape(inputs)  # it's assigned dynamically
        # depth: A scalar defining the depth of the one hot dimension.
        inputs = tf.one_hot(inputs, depth=self.num_chars)  # [batch_size, seq_length, num_chars]
        # Return zero-filled state tensor(s).
        state = self.cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        for t in range(seq_length.numpy()):
            output, state = self.cell(inputs[:, t, :], state)  # 也就是说，一个batch是混在一起训练的……
        output = self.dense(output)
        return output

    # temperature：控制分布的形状，参数值越大则分布越平缓，生成文本的丰富度越高；参数值越小则分布越陡峭，生成文本的丰富度越低
    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs)
        prob = tf.nn.softmax(logits / temperature).numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[i, :])
                         for i in range(batch_size.numpy())])

# hyper-parameters
batch_size = 128
learning_rate = 0.01
seq_length = 40
num_batches = 7200

dataLoader = Dataloader()
model = RNN(len(dataLoader.chars))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
for batch_index in range(num_batches):
    X, y = dataLoader.get_batch(seq_length, batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred = model(X)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# Generate Text
X_, _ = dataLoader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
    X = X_
    print("diversity %f:" % diversity)
    for t in range(400):
        y_pred = model.predict(X, diversity)
        print(dataLoader.indices_char[y_pred[0]], end='', flush=True)
        X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
    print()