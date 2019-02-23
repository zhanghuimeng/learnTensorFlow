import tensorflow as tf
import numpy as np

# 由于之前的效果太差，决定直接参考
# https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

maxlen = 40  # 最大序列长度

class Dataloader():
    def __init__(self):
        with open("../../data/nietzsche.txt", encoding="utf-8") as f:
            self.raw_text = f.read().lower()
        print("corpus length:", len(self.raw_text))
        # 将字符转化为数字
        self.chars = sorted(list(set(self.raw_text)))
        print("total chars:", len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        self.sentences = []
        self.next_chars = []
        for i in range(0, len(self.raw_text) - maxlen, 3):
            self.sentences.append(self.raw_text[i: i + maxlen])
            self.next_chars.append(self.raw_text[i + maxlen])
        print("sequences:", len(self.sentences))

        print("one-hot...")
        self.x = np.zeros((len(self.sentences), maxlen, len(self.chars)), dtype=np.bool)
        self.y = np.zeros((len(self.sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                self.x[i, t, self.char_indices[char]] = 1
            self.y[i, self.char_indices[self.next_chars[i]]] = 1

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.x)[0], batch_size)
        return self.x[index, :], self.y[index]


class RNN(tf.keras.Model):
    def __init__(self, num_chars, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=128)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]  # 不能动态分配了
        state = self.cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        for t in range(self.seq_length):
            output, state = self.cell(inputs[:, t, :], state)  # 也就是说，一个batch是混在一起训练的……
        output = self.dense(output)
        return output

    # temperature：控制分布的形状，参数值越大则分布越平缓，生成文本的丰富度越高；参数值越小则分布越陡峭，生成文本的丰富度越低
    def predict(self, inputs, temperature=1., batch_size=1):
        logits = self(inputs)
        prob = tf.nn.softmax(logits / temperature)
        return np.array([np.random.choice(self.num_chars, p=prob[i, :])
                         for i in range(batch_size)])

# hyper-parameters
batch_size = 128
learning_rate = 0.01
seq_length = 40
num_batches = 7200

dataLoader = Dataloader()
model = RNN(len(dataLoader.chars), seq_length)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
X_placeholder = tf.placeholder(name='X', shape=[None, seq_length], dtype=tf.int32)
y_placeholder = tf.placeholder(name='y', shape=[None], dtype=tf.int32)
y_logit_pred = model(X_placeholder)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_placeholder, logits=y_logit_pred)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for batch_index in range(num_batches):
        X, y = dataLoader.get_batch(seq_length, batch_size)
        _, lossVal = sess.run([train_op, loss], feed_dict={X_placeholder: X, y_placeholder: y})
        print("batch %d: loss=%f" % (batch_index, lossVal))

    # Generate Text
    X_, _ = dataLoader.get_batch(seq_length, 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        X = X_
        print("diversity %f:" % diversity)
        for t in range(400):
            y_pred = sess.run(model.predict(X_, diversity, batch_size))
            print(dataLoader.indices_char[y_pred[0]], end='', flush=True)
            X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)