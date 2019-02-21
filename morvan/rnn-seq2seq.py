import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


class SeqDataLoader:
    def __init__(self):
        with open('../data/letters_source.txt') as f:
            source_str = f.read().splitlines()
        with open('../data/letters_target.txt') as f:
            target_str = f.read().splitlines()
        # 创建词（字母）表（源端和目标端的词表相同）
        char_set = set(''.join(source_str))
        self.vocab_list = sorted(list(char_set))
        # 添加控制字符
        self.vocab_list = ['<pad>', '<eos>', '<go>'] + self.vocab_list
        self.vocab_len = len(self.vocab_list)
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        # 创建词和id的相互映射
        for i, ch in enumerate(self.vocab_list):
            self.vocab_to_id[ch] = i
            self.id_to_vocab[i] = ch
        self.eos_id = self.vocab_to_id['<eos>']
        self.go_id = self.vocab_to_id['<go>']
        self.pad_id = self.vocab_to_id['<pad>']
        self.source = []
        self.source_length = []  # 源端长度（不含控制字符）
        self.target = []
        self.target_length = []  # 目标端长度（不含控制字符）
        # 原来添加控制字符这一步是在这里做的，但后来由于分batch的原因移到了get_batch中
        for i in range(len(source_str)):
            self.source.append([self.vocab_to_id[ch] for ch in source_str[i]])
            self.source_length.append(len(source_str[i]))
            self.target.append([self.vocab_to_id[ch] for ch in target_str[i]])
            self.target_length.append(len(target_str[i]))
        # 将数据分成训练数据和测试数据
        # 因为每一行长度不同，所以不适合直接用numpy来做，所以才写了divide辅助函数
        # 用np.random.choice来选择不重复的index
        indices = np.random.choice(np.shape(self.source)[0], int(len(self.source) * 0.8), replace=False)
        indices = set(indices)
        self.train_source_str, self.test_source_str = self.divide(source_str, indices)
        self.train_source, self.test_source = self.divide(self.source, indices)
        self.train_target_str, self.test_target_str = self.divide(target_str, indices)
        self.train_target, self.test_target = self.divide(self.target, indices)
        self.train_source_length, self.test_source_length = self.divide(self.source_length, indices)
        self.train_target_length, self.test_target_length = self.divide(self.target_length, indices)
        # 对测试数据进行padding
        # 之所以没有直接对全部训练数据进行padding，是因为对于训练数据的每个batch，
        # 为了节省内存，padding size可能会更小，不过在这个例子中估计没什么区别
        # 当然，如果测试数据很多，不能一batch测完，那也需要进行同样的操作
        self.test_max_len = max(self.test_source_length) + 1
        self.test_source_padded = []
        for i in range(len(self.test_source)):
            line = self.test_source[i]
            while len(line) < self.test_max_len:
                line.append(self.pad_id)
            self.test_source_padded.append(line)

    @staticmethod
    def divide(data, indices):
        """
        用于分割数据的辅助函数
        """
        train = []
        test = []
        for i in range(len(data)):
            if i in indices:
                train.append(data[i])
            else:
                test.append(data[i])
        return train, test

    def get_batch(self, size):
        """
        返回大小为size的一个batch，加入了控制字符和padding
        :param size: batch大小
        :return: 源端数据, 源端长度, 目标端输入, 目标端输入长度, 目标端输出, 目标端输出长度
        """
        indices = np.random.randint(0, np.shape(self.train_source)[0], size)
        # 源端不需要加控制字符，只需要加padding
        # 1, 2, ..., len (padding)
        batch_source_length = [self.train_source_length[i] for i in indices]
        max_len = max(batch_source_length)
        batch_source = []
        for i in indices:
            line = self.train_source[i]
            while len(line) < max_len:
                line.append(self.pad_id)
            batch_source.append(line)
        # 目标端输入开头需要加控制字符<go>
        # <go>, 1, 2, ..., len (padding)
        batch_target_input_length = [self.train_target_length[i] + 1 for i in indices]
        max_len = max(batch_target_input_length)
        batch_target_input = []
        for i in indices:
            line = [self.go_id] + self.train_target[i]
            while len(line) < max_len:
                line.append(self.pad_id)
            batch_target_input.append(line)
        # 目标端输出结尾需要加控制字符<eos>
        # 1, 2, ..., len, <eos> (padding)
        batch_target_output_length = [self.train_target_length[i] + 1 for i in indices]
        max_len = max(batch_target_output_length)
        batch_target_output = []
        for i in indices:
            line = self.train_target[i] + [self.eos_id]
            while len(line) < max_len:
                line.append(self.pad_id)
            batch_target_output.append(line)

        return batch_source, batch_source_length, batch_target_input, batch_target_input_length, \
            batch_target_output, batch_target_output_length

    def lookup_to_str(self, ids):
        """
        将一维或二维id np.array转换为字符串
        """
        if len(ids.shape) == 1:
            one_str = ''
            for i in range(ids.shape[0]):
                if ids[i] == self.eos_id:
                    break
                one_str += self.id_to_vocab[ids[i]]
            return one_str

        str_list = []
        for i in range(ids.shape[0]):
            one_str = ''
            for j in range(ids.shape[1]):
                if ids[i, j] == self.eos_id:
                    break
                one_str += self.id_to_vocab[ids[i, j]]
            str_list.append(one_str)
        return str_list


class Seq2Seq:
    def __init__(self, vocab_len, input_embedding_size, state_size, learning_rate, max_len):
        """
        定义模型
        :param vocab_len: 词表大小
        :param input_embedding_size: 源端（和目标端）embedding大小
        :param state_size: LSTM state大小
        :param learning_rate: 学习率
        :param max_len: 源端和目标端最长数据
        """
        self.learning_rate = learning_rate
        self.max_len = max_len
        # 创建输入placeholder
        # 对于第二维是None的placeholder，它在输入时第二维最长为max_len，实际上与batch有关
        with tf.variable_scope('input'):
            self.source_id = tf.placeholder(tf.int32, [None, None], 'source')                    # [batch_size, max_len]
            self.source_length = tf.placeholder(tf.int32, [None], 'source_length')               # [batch_size]
            self.target_input_id = tf.placeholder(tf.int32, [None, None], 'target_input')        # [batch_size, max_len]
            self.target_input_length = tf.placeholder(tf.int32, [None], 'target_input_length')   # [batch_size]
            self.target_output_id = tf.placeholder(tf.int32, [None, None], 'target_output')      # [batch_size, max_len]
            self.target_output_length = tf.placeholder(tf.int32, [None], 'target_output_length') # [batch_size]
            batch_size = tf.shape(self.source_id)[0]  # 然而在RNN里总是需要batch_size这个参数
            self.vocab_len = vocab_len
        # 创建源端和目标端的embedding
        with tf.variable_scope('input-embedding'):
            # 这里把源端和目标端输入的embedding分开了，不过既然词表一样，我觉得不分开也差不多
            self.source_input_embedding = tf.layers.Dense(input_embedding_size)
            self.target_input_embedding = tf.layers.Dense(input_embedding_size)
            self.target_output_embedding = tf.layers.Dense(vocab_len)
            # [batch_size, max_len] => [batch_size, max_len, emb_size]
            source_emb = self.embed_sparse_source_input(self.source_id)
            # [batch_size, max_len] => [batch_size, max_len, emb_size]
            target_input_emb = self.embed_sparse_target_input(self.target_input_id)
            # 这是专门用于目标端infer时的第一个输入的
            # [batch_size, emb_size] of <go>s
            self.go_input = self.embed_sparse_target_input(tf.tile([data_loader.go_id], [batch_size]))
        # 创建encoder
        with tf.variable_scope('encoder'):
            self.enc_lstm_cell = tf.nn.rnn_cell.LSTMCell(state_size)
            # state = [batch_size, state_size (c, h)]
            state = self.encoder(source_emb, self.source_length, batch_size)
            kernel, bias = self.enc_lstm_cell.variables
            self.k_sum_1 = tf.summary.histogram('Encoder Kernel', kernel)
            self.b_sum_1 = tf.summary.histogram('Encoder Bias', bias)
        # 创建decoder
        with tf.variable_scope('decoder'):
            self.dec_lstm_cell = tf.nn.rnn_cell.LSTMCell(state_size)
            # 训练时的decoder
            with tf.variable_scope('training'):
                # outputs = [batch_size, max_len, state_size]
                outputs = self.decoder_training(target_input_emb, state, self.target_input_length)
                # outputs => [batch_size, max_len, vocab_size]
                outputs = self.target_output_embedding(outputs)
                self.train_outputs = tf.argmax(outputs, -1)
                self.mask = tf.sequence_mask(self.target_output_length, dtype=tf.float32)
                # 计算输出和期望输出之间的cross-entropy loss
                # 不考虑pad时：
                # self.loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=self.target_output_id,
                #                                              weights=mask)
                # 考虑pad时：
                weights = tf.fill([batch_size, tf.reduce_max(self.target_output_length)], 1.0)
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=self.target_output_id,
                                                             weights=weights)
                self.loss_sum = tf.summary.scalar('loss', self.loss)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                # 不进行clipping：
                # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                # 进行clipping：
                gradients = optimizer.compute_gradients(self.loss)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gradients)
            # 预测（infer）时的decoder
            with tf.variable_scope('predicting'):
                self.predict_outputs = self.decoder_predicting(state, batch_size)

            kernel, bias = self.dec_lstm_cell.variables
            self.k_sum_2 =tf.summary.histogram('Decoder Kernel', kernel)
            self.b_sum_2 = tf.summary.histogram('Decoder Bias', bias)
            self.train_summary = tf.summary.merge([self.k_sum_1, self.k_sum_2, self.b_sum_1,
                                                   self.b_sum_2, self.loss_sum])

    def encoder(self, inputs, length, batch_size):
        """
        encoder的具体实现
        :param inputs: 输入数据
        :param length: 输入数据长度
        :param batch_size: batch_size
        :return: 以zero_state为初始状态运行后得到的状态
        """
        initial_state = self.enc_lstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(self.enc_lstm_cell, inputs, sequence_length=length,
                                                 initial_state=initial_state)
        return final_state

    def decoder_training(self, inputs, encoder_state, length):
        """
        training时decoder的具体实现
        因为有teacher forcing，比较简单
        :param inputs: 输入数据
        :param encoder_state: 初始状态
        :param length: 输入长度
        :return:
        """
        initial_state = encoder_state
        # 输入：    <go>, 1, 2, ..., len
        # 期望输出： 1, 2, ..., len, <eos>
        # outputs = [batch_size, max_len, state_size]
        outputs, final_state = tf.nn.dynamic_rnn(self.dec_lstm_cell, inputs, sequence_length=length,
                                                 initial_state=initial_state)
        return outputs

    def decoder_predicting(self, encoder_state, batch_size):
        """
        infer时decoder的具体实现
        因为需要手动将上一时刻的输出变换成下一时刻的输入，所以逻辑很复杂
        :param encoder_state: 初始状态
        :param batch_size: batch_size
        :return: infer结果
        """
        finished = tf.fill([batch_size], False)  # use tf.fill for scalar values
        # 0时刻输入为<go>
        inputs = self.go_input
        state = encoder_state
        # 创建用于存储infer输出的变量（必须用变量）
        decoded_outputs = tf.fill([batch_size, 0], 0)
        # 当前时刻
        initial_i = tf.Variable(0)
        print(decoded_outputs)

        # 因为不能用tensor进行条件判断，所以必须用while_loop operation...
        def body(i, inputs, state, decoded_outputs, finished):
            output, state = self.dec_lstm_cell(inputs, state)  # 只算一步，所以不用dynamic_rnn
            token = self.get_target_output_id(output)
            decoded_outputs = tf.concat([decoded_outputs, tf.transpose([token])], -1)
            # 整个batch是否都已经结束了
            finished = tf.logical_or(finished, tf.equal(token, data_loader.eos_id))
            inputs = self.embed_sparse_target_input(token)
            return i + 1, inputs, state, decoded_outputs, finished

        def condition(i, inputs, state, decoded_outputs, finished):
            # 如果还没有全部结束且长度没有超过max_len
            return tf.logical_and(tf.logical_not(tf.reduce_all(finished)), i < self.max_len)

        i, inputs, state, decoded_outputs, finished = \
            tf.while_loop(condition, body, [initial_i, inputs, state, decoded_outputs, finished],
                          shape_invariants=[initial_i.get_shape(), inputs.get_shape(),
                                            tf.nn.rnn_cell.LSTMStateTuple(
                                                *tuple(tf.TensorShape((None, size))
                                                       for size in self.dec_lstm_cell.state_size)),
                                            tf.TensorShape([None, None]), finished.get_shape()])

        return tf.stack(decoded_outputs)

    def embed_sparse_source_input(self, inputs):
        """
        将id转换成稀疏的one-hot向量
        """
        inputs = tf.one_hot(inputs, self.vocab_len)
        return self.source_input_embedding(inputs)

    def embed_sparse_target_input(self, inputs):
        inputs = tf.one_hot(inputs, self.vocab_len)
        return self.target_input_embedding(inputs)

    def get_target_output_id(self, inputs):
        """
        将目标端输出的logits转成id
        """
        with tf.variable_scope('output-embedding'):
            outputs = self.target_output_embedding(inputs)
            outputs = tf.argmax(outputs, -1, output_type=tf.int32)  # this is annoying
            return outputs


EPOCH = 60
EPOCH_STEP = 40
BATCH_SIZE = 200
EMBEDDING = 15
RNN_STATE = 100
LR = 1e-3
MAX_LEN = 128

data_loader = SeqDataLoader()
model = Seq2Seq(vocab_len=data_loader.vocab_len, input_embedding_size=EMBEDDING, state_size=RNN_STATE,
                learning_rate=LR, max_len=MAX_LEN)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs', sess.graph)
    # 在tensorboard中加入test bleu值
    bleu_summary = tf.Summary()
    bleu_summary.value.add(tag='test bleu', simple_value=None)
    for epoch in range(EPOCH):
        print("Epoch: %d" % epoch)
        for i in range(EPOCH_STEP):
            s, sl, ti, til, to, tol = data_loader.get_batch(BATCH_SIZE)
            loss, _, train_outputs, mask, train_summary = \
                sess.run([model.loss, model.train_op, model.train_outputs, model.mask, model.train_summary],
                         feed_dict={model.source_id: s, model.source_length: sl, model.target_input_id: ti,
                                    model.target_input_length: til, model.target_output_id: to,
                                    model.target_output_length: tol})
            print("Epoch %d step %d: loss=%f" % (epoch, i, loss))

            prediction = sess.run(
                model.predict_outputs,
                feed_dict={model.source_id: data_loader.test_source_padded,
                           model.source_length: data_loader.test_source_length})
            # 计算BLEU平均值（考虑n<4时的平滑）
            prediction = data_loader.lookup_to_str(prediction)
            target = data_loader.test_target_str
            bleu = []
            for j in range(len(target)):
                bleu.append(
                    sentence_bleu(references=[list(target[j])], hypothesis=list(prediction[j]),
                                  smoothing_function=SmoothingFunction().method3))
            for j in range(20):
                print(target[j], prediction[j], bleu[j])
            bleu_average = np.average(bleu)
            print("bleu =", np.average(bleu_average))
            bleu_summary.value[0].simple_value = bleu_average
            writer.add_summary(train_summary, epoch * EPOCH_STEP + i)
            writer.add_summary(bleu_summary, epoch * EPOCH_STEP + i)
            writer.flush()
