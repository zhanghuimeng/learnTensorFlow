import tensorflow as tf

# https://github.com/tensorflow/tensorflow/issues/4814
def create_reset_metric(metric, scope='reset_metrics', **metric_args):
  with tf.variable_scope(scope) as scope:
    metric_op, update_op = metric(**metric_args)
    vars = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
    reset_op = tf.variables_initializer(vars)
  return metric_op, update_op, reset_op


class Model:
    def __init__(self, enc_hidden_size, dec_hidden_size,
                 src_vocab_size, tgt_vocab_size, emb_size, learning_rate):
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.learning_rate = learning_rate
        with tf.variable_scope('embedding'):
            self.src_emb = tf.get_variable("src_embeddings", [src_vocab_size, emb_size], dtype=tf.float32)
            self.tgt_emb = tf.get_variable("tgt_embeddings", [tgt_vocab_size, emb_size], dtype=tf.float32)
        with tf.variable_scope('src_rnn'):
            self.src_rnn_cell = {
                'f': tf.nn.rnn_cell.GRUCell(enc_hidden_size),
                'w': tf.nn.rnn_cell.GRUCell(enc_hidden_size)}
        with tf.variable_scope('tgt_rnn'):
            self.tgt_rnn_cell = {
                'f': tf.nn.rnn_cell.GRUCell(enc_hidden_size),
                'w': tf.nn.rnn_cell.GRUCell(enc_hidden_size)}
        with tf.variable_scope('attention'):
            self.weight_a = tf.get_variable('W_a', shape=[2 * self.dec_hidden_size, 1], dtype=tf.float32,
                                            initializer=tf.initializers.random_normal(0.1))
            self.dense = tf.layers.Dense(units=1)

    def training(self, src, tgt, src_len, tgt_len, hter):
        with tf.variable_scope('training'):
            self.train_pred = self.predict(src, tgt, src_len, tgt_len)
            self.loss = tf.losses.mean_squared_error(
                labels=tf.expand_dims(hter, 1),
                predictions=self.train_pred)
            self.train_op = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)\
                .minimize(self.loss)

    def testing(self, src, tgt, src_len, tgt_len, hter):
        with tf.variable_scope('testing'):
            self.test_pred = self.predict(src, tgt, src_len, tgt_len)
            self.test_mse, self.test_mse_update, self.test_mse_reset = create_reset_metric(
                tf.metrics.mean_squared_error,
                'test_mse',
                labels=tf.expand_dims(hter, 1),
                predictions=self.test_pred,
                name="test_mse")
            self.test_pearson, self.test_pearson_update, self.test_pearson_reset = create_reset_metric(
                tf.contrib.metrics.streaming_pearson_correlation,
                'test_pearson',
                labels=tf.expand_dims(hter, 1),
                predictions=self.test_pred,
                name="test_pearson")

    def predict(self, src, tgt, src_len, tgt_len):
        embedded_src = tf.nn.embedding_lookup(self.src_emb, src)
        embedded_tgt = tf.nn.embedding_lookup(self.tgt_emb, tgt)
        with tf.variable_scope('src_birnn'):
            src_h = tf.nn.bidirectional_dynamic_rnn(
                self.src_rnn_cell['f'],
                self.src_rnn_cell['w'],
                embedded_src,
                dtype=tf.float32,  # 如果不给定RNN initial state，则必须给定dtype（是状态的dtype！）
                sequence_length=src_len)
            src_h = src_h[0]  # 原来是(outputs, output_states)
            src_h = tf.concat(src_h, 2)
        with tf.variable_scope('tgt_birnn'):
            tgt_h = tf.nn.bidirectional_dynamic_rnn(
                self.tgt_rnn_cell['f'],
                self.tgt_rnn_cell['w'],
                embedded_tgt,
                dtype=tf.float32,
                sequence_length=tgt_len)
            tgt_h = tgt_h[0]
            tgt_h = tf.concat(tgt_h, 2)
        h = tf.concat([src_h, tgt_h], 1)
        # 对h进行attention
        # [66, 1000]
        def unbatch_h(h):
            # [1000]
            def unpack_h(h):
                # 然而这个broadcast比我想象得难用
                return tf.reduce_sum(tf.multiply(h, self.weight_a))
            a = tf.map_fn(unpack_h, h)
            a = tf.nn.softmax(a)
            v = tf.reduce_sum(tf.multiply(tf.expand_dims(a, 1), h), 0)
            return v
        v = tf.map_fn(unbatch_h, h)
        v = self.dense(v)
        pred = tf.nn.sigmoid(v)
        return pred