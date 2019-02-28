# encoding: utf-8
import tensorflow as tf
import argparse


BATCH_SIZE = 32
STEPS = 2000
EVAL_STEPS = 50
EMB_SIZE = 500
HIDDEN_SIZE = 200
LR = 1e-3


# https://github.com/tensorflow/tensorflow/issues/4814
def create_reset_metric(metric, scope='reset_metrics', **metric_args):
  with tf.variable_scope(scope) as scope:
    metric_op, update_op = metric(**metric_args)
    vars = tf.contrib.framework.get_variables(
                 scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
    reset_op = tf.variables_initializer(vars)
  return metric_op, update_op, reset_op


def read_vocab(src, tgt):
    vocab_idx_src = tf.contrib.lookup.index_table_from_file(src, num_oov_buckets=1)
    vocab_idx_tgt = tf.contrib.lookup.index_table_from_file(tgt, num_oov_buckets=1)
    vocab_str_src = tf.contrib.lookup.index_to_string_table_from_file(src, default_value='<unk>')
    vocab_str_tgt = tf.contrib.lookup.index_to_string_table_from_file(tgt, default_value='<unk>')
    return vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt


def one_dataset_loader(src, tgt, hter, vocab_idx_src, vocab_idx_tgt, one_shot=False):
    src = tf.data.TextLineDataset(src)
    tgt = tf.data.TextLineDataset(tgt)
    hter = tf.data.TextLineDataset(hter)
    src = src.map(lambda string: tf.string_split([string]).values)
    tgt = tgt.map(lambda string: tf.string_split([string]).values)
    hter = hter.map(lambda x: tf.strings.to_number(x))
    src = src.map(lambda tokens: vocab_idx_src.lookup(tokens))
    src_len = src.map(lambda tokens: tf.size(tokens))
    tgt = tgt.map(lambda tokens: vocab_idx_tgt.lookup(tokens))
    tgt_len = tgt.map(lambda tokens: tf.size(tokens))
    dataset = tf.data.Dataset.zip({
        "src": src,
        "src_len": src_len,
        "tgt": tgt,
        "tgt_len": tgt_len,
        "hter": hter
    })
    src_pad_id = vocab_idx_src.lookup(tf.constant('<pad>'))
    tgt_pad_id = vocab_idx_tgt.lookup(tf.constant('<pad>'))
    padded_shapes = {
        'src': tf.TensorShape([None]),
        'src_len': [],
        'tgt': tf.TensorShape([None]),
        'tgt_len': [],
        'hter': []
    }
    padding_values = {
        'src': src_pad_id,
        'src_len': tf.constant(0),
        'tgt': tgt_pad_id,
        'tgt_len': tf.constant(0),
        'hter': tf.constant(0.0)
    }
    if not one_shot:
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat()
    dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes, padding_values=padding_values)
    return dataset


def data_loader(vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt):
    train_dataset = one_dataset_loader(
        src=args.train[0],
        tgt=args.train[1],
        hter=args.train[2],
        vocab_idx_src=vocab_idx_src,
        vocab_idx_tgt=vocab_idx_tgt)
    dev_dataset = one_dataset_loader(
        src=args.dev[0],
        tgt=args.dev[1],
        hter=args.dev[2],
        vocab_idx_src=vocab_idx_src,
        vocab_idx_tgt=vocab_idx_tgt,
        one_shot=True)
    test_dataset = one_dataset_loader(
        src=args.test[0],
        tgt=args.test[1],
        hter=args.test[2],
        vocab_idx_src=vocab_idx_src,
        vocab_idx_tgt=vocab_idx_tgt,
        one_shot=True)
    # iterator = training_dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    # with tf.Session() as sess:
    #     sess.run(tf.tables_initializer())  # 为何需要explicitly initialize这个呢？
    #     sess.run(iterator.initializer)
    #     for i in range(2):
    #         print(sess.run(next_element))
    return train_dataset, dev_dataset, test_dataset


class Model:
    def __init__(self, train_dataset, dev_dataset, hidden_size, src_vocab_size, tgt_vocab_size, emb_size,
                 learning_rate):
        self.hidden_size = hidden_size
        # 一次做完training dev test好像不太对，但现在先这样好了……
        with tf.variable_scope('inputs'):
            self.train_iter = train_dataset.make_initializable_iterator()
            train_ele = self.train_iter.get_next()
            self.dev_iter = dev_dataset.make_initializable_iterator()
            dev_ele = self.dev_iter.get_next()
            self.test_iter = test_dataset.make_initializable_iterator()
            test_ele = self.test_iter.get_next()
        with tf.variable_scope('embedding'):
            self.src_emb = tf.get_variable("src_embeddings", [src_vocab_size, emb_size], dtype=tf.float32)
            self.tgt_emb = tf.get_variable("tgt_embeddings", [tgt_vocab_size, emb_size], dtype=tf.float32)
        with tf.variable_scope('src_rnn'):
            self.src_rnn_cell = {
                'f': tf.nn.rnn_cell.GRUCell(hidden_size),
                'w': tf.nn.rnn_cell.GRUCell(hidden_size)}
        with tf.variable_scope('tgt_rnn'):
            self.tgt_rnn_cell = {
                'f': tf.nn.rnn_cell.GRUCell(hidden_size),
                'w': tf.nn.rnn_cell.GRUCell(hidden_size)}
        with tf.variable_scope('attention'):
            self.weight_a = tf.get_variable('W_a', shape=[2 * self.hidden_size, 1], dtype=tf.float32,
                                            initializer=tf.initializers.random_normal(0.1))
            self.dense = tf.layers.Dense(units=1)
        with tf.variable_scope('training'):
            self.train_pred = self.predict(
                src=train_ele['src'],
                tgt=train_ele['tgt'],
                src_len=train_ele['src_len'],
                tgt_len=train_ele['tgt_len'])
            # with tf.Session() as sess:
            #     sess.run(tf.tables_initializer())
            #     sess.run(self.train_iter.initializer)
            #     sess.run(tf.global_variables_initializer())
            #     print('pred:', sess.run(tf.shape(pred)))
            self.loss = tf.losses.mean_squared_error(
                labels=tf.expand_dims(train_ele['hter'], 1),
                predictions=self.train_pred)
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        with tf.variable_scope('dev'):
            self.dev_pred = self.predict(dev_ele['src'], dev_ele['tgt'], dev_ele['src_len'], dev_ele['tgt_len'])
            self.dev_mse, self.dev_mse_update, self.dev_mse_reset = create_reset_metric(
                tf.metrics.mean_squared_error,
                'dev_mse',
                labels=tf.expand_dims(dev_ele['hter'], 1),
                predictions=self.dev_pred,
                name="dev_mse")
            self.dev_pearson, self.dev_pearson_update, self.dev_pearson_reset = create_reset_metric(
                tf.contrib.metrics.streaming_pearson_correlation,
                'dev_pearson',
                labels=tf.expand_dims(dev_ele['hter'], 1),
                predictions=self.dev_pred,
                name="dev_pearson")
        with tf.variable_scope('test'):
            self.test_pred = self.predict(
                test_ele['src'], test_ele['tgt'], test_ele['src_len'], test_ele['tgt_len'])
            self.test_mse, self.test_mse_update = tf.metrics.mean_squared_error(
                labels=tf.expand_dims(test_ele['hter'], 1),
                predictions=self.test_pred)
            self.test_pearson, self.test_pearson_update = tf.contrib.metrics.streaming_pearson_correlation(
                labels=tf.expand_dims(test_ele['hter'], 1),
                predictions=self.test_pred)

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
            # with tf.Session() as sess:
            #     sess.run(tf.tables_initializer())
            #     sess.run(train_iter.initializer)
            #     sess.run(tf.global_variables_initializer())
            #     print('src_h: ')
            #     print(sess.run(tf.shape(src_h[0])))
            #     print(sess.run(tf.shape(src_h[1])))
            src_h = tf.concat(src_h, 2)
        with tf.variable_scope('tgt_birnn'):
            # with tf.Session() as sess:
            #     sess.run(tf.tables_initializer())
            #     sess.run(train_iter.initializer)
            #     print(sess.run(train_src))
            tgt_h = tf.nn.bidirectional_dynamic_rnn(
                self.tgt_rnn_cell['f'],
                self.tgt_rnn_cell['w'],
                embedded_tgt,
                dtype=tf.float32,
                sequence_length=tgt_len)
            tgt_h = tgt_h[0]
            tgt_h = tf.concat(tgt_h, 2)
        h = tf.concat([src_h, tgt_h], 1)
        # 打印h的形状
        # with tf.Session() as sess:
        #     sess.run(tf.tables_initializer())
        #     sess.run(train_iter.initializer)
        #     sess.run(tf.global_variables_initializer())
        #     print(sess.run(tf.shape(src_h)))
        #     print(sess.run(tf.shape(tgt_h)))
        #     print(sess.run(tf.shape(h)))
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
        # with tf.Session() as sess:
        #     sess.run(tf.tables_initializer())
        #     sess.run(self.train_iter.initializer)
        #     sess.run(tf.global_variables_initializer())
        #     print('v:', sess.run(tf.shape(v)))
        v = self.dense(v)
        pred = tf.nn.sigmoid(v)
        return pred


parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, nargs=3, help='Parallel training files and HTER score')
parser.add_argument('--vocab', type=str, nargs=2, help='Parallel vocab files (with ctrl)')
parser.add_argument('--dev', type=str, nargs=3, help='Parallel development files and HTER score')
parser.add_argument('--test', type=str, nargs=3, help='Parallel test files and HTER score')
args = parser.parse_args()

# 据说把数据预处理放在CPU上是best practice
with tf.device('/cpu:0'):
    print("Loading vocabulary from %s and %s ..." % (args.vocab[0], args.vocab[1]))
    vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt = read_vocab(args.vocab[0], args.vocab[1])
    # with tf.Session() as sess:
    # 为了debug
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #     sess.run(tf.tables_initializer())
    #     src_vocab_size = sess.run(vocab_idx_src.size())
    #     tgt_vocab_size = sess.run(vocab_idx_tgt.size())
    # 没法用sess来算，只好直接行数+1了！
    src_vocab_size = 1
    with open(args.vocab[0], 'r') as f:
        for line in f:
            src_vocab_size += 1
    tgt_vocab_size = 1
    with open(args.vocab[1], 'r') as f:
        for line in f:
            tgt_vocab_size += 1
    print('Loaded src vocabulary size %d' % src_vocab_size)
    print('Loaded tgt vocabulary size %d' % tgt_vocab_size)
    train_dataset, dev_dataset, test_dataset \
        = data_loader(vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt)
print('Building computation model...')
model = Model(train_dataset,
              dev_dataset,
              hidden_size=HIDDEN_SIZE,
              emb_size=EMB_SIZE,
              src_vocab_size=src_vocab_size,
              tgt_vocab_size=tgt_vocab_size,
              learning_rate=LR)
print('Debugging: Built computation model...')
with tf.Session() as sess:
# 还是为了debug
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print('Debugging: Running initialization...')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  # for pearson op
    sess.run(tf.tables_initializer())
    sess.run(model.train_iter.initializer)
    # 打印到tensorboard
    print('Debugging: Preparing tensorboard...')
    writer = tf.summary.FileWriter('logs', sess.graph)
    train_summary = tf.Summary()
    train_summary.value.add(tag='train loss', simple_value=None)
    dev_summary = tf.Summary()
    dev_summary.value.add(tag='dev mse', simple_value=None)
    dev_summary.value.add(tag='dev pearson', simple_value=None)

    print('Debugging: Ready to start training...')
    for step in range(STEPS):
        loss, _ = sess.run([model.loss, model.train_op])
        print('Step %d: loss=%f' % (step, loss))
        train_summary.value[0].simple_value = loss
        writer.add_summary(train_summary, step)
        if step % EVAL_STEPS == 0:
            try:
                sess.run(model.dev_iter.initializer)
                sess.run(model.dev_mse_reset)
                sess.run(model.dev_pearson_reset)
                while True:
                    mse, pearson = sess.run([model.dev_mse_update, model.dev_pearson_update])
                    # print('Eval (in the middle) %d: mse=%f, pearson=%f' % (step // EVAL_STEPS, mse, pearson))
            except tf.errors.OutOfRangeError:  # Thrown at the end of the epoch.
                pass
            mse, pearson = sess.run([model.dev_mse, model.dev_pearson])
            print('Eval %d: mse=%f, pearson=%f' % (step // EVAL_STEPS, mse, pearson))
            dev_summary.value[0].simple_value = mse
            dev_summary.value[1].simple_value = pearson
            writer.add_summary(dev_summary, step)
        writer.flush()

    # 等下，test不应该shuffle的。。。
    sess.run(model.test_iter.initializer)
    try:
        while True:
            mse, pearson = sess.run([model.test_mse_update, model.test_pearson_update])
    except tf.errors.OutOfRangeError:  # Thrown at the end of the epoch.
        pass
    mse, pearson = sess.run([model.test_mse, model.test_pearson])
    print('Test: mse=%f, pearson=%f' % (mse, pearson))
