import tensorflow as tf
import argparse


BATCH_SIZE=128
STEPS = 20000
EMB_SIZE = 1000
HIDDEN_SIZE = 500
OUT_SIZE = 100


def read_vocab(src, tgt):
    vocab_idx_src = tf.contrib.lookup.index_table_from_file(src, num_oov_buckets=1)
    vocab_idx_tgt = tf.contrib.lookup.index_table_from_file(tgt, num_oov_buckets=1)
    vocab_str_src = tf.contrib.lookup.index_to_string_table_from_file(src, default_value='<unk>')
    vocab_str_tgt = tf.contrib.lookup.index_to_string_table_from_file(tgt, default_value='<unk>')
    return vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt

def one_dataset_loader(src, tgt, hter, vocab_idx_src, vocab_idx_tgt):
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
    dataset = (dataset
               .shuffle(buffer_size=10000)
               .padded_batch(BATCH_SIZE, padded_shapes=padded_shapes, padding_values=padding_values)
            )
    return dataset

def data_loader():
    vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt = read_vocab(args.vocab[0], args.vocab[1])
    train_dataset = one_dataset_loader(args.train[0], args.train[1], args.train[2], vocab_idx_src, vocab_idx_tgt)
    dev_dataset = one_dataset_loader(args.dev[0], args.dev[1], args.dev[2], vocab_idx_src, vocab_idx_tgt)
    test_dataset = one_dataset_loader(args.test[0], args.test[1], args.test[2], vocab_idx_src, vocab_idx_tgt)
    # iterator = training_dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    # with tf.Session() as sess:
    #     sess.run(tf.tables_initializer())  # 为何需要explicitly initialize这个呢？
    #     sess.run(iterator.initializer)
    #     for i in range(2):
    #         print(sess.run(next_element))
    return vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt, train_dataset, dev_dataset, test_dataset


class Model:
    def __init__(self, train_dataset, dev_dataset, hidden_size, src_vocab_size, tgt_vocab_size, emb_size, out_size):
        with tf.variable_scope('inputs'):
            train_iter = train_dataset.make_initializable_iterator()
            train_ele = train_iter.get_next()
            train_src = train_ele['src']
            train_src_len = train_ele['src_len']
            train_tgt = train_ele['tgt']
            train_tgt_len = train_ele['tgt_len']
            train_hter = train_ele['hter']
        with tf.variable_scope('embedding'):
            self.src_emb = tf.get_variable("src_embeddings", [src_vocab_size, emb_size])
            embedded_train_src = tf.nn.embedding_lookup(self.src_emb, train_src)
            self.tgt_emb = tf.get_variable("tgt_embeddings", [tgt_vocab_size, emb_size])
            embedded_train_tgt = tf.nn.embedding_lookup(self.tgt_emb, train_tgt)
        with tf.variable_scope('training'):
            self.src_rnn_cell = {
                'f': tf.nn.rnn_cell.GRUCell(hidden_size),
                'w': tf.nn.rnn_cell.GRUCell(hidden_size)}
            self.tgt_rnn_cell = {'f': tf.nn.rnn_cell.GRUCell(hidden_size), 'w': tf.nn.rnn_cell.GRUCell(hidden_size)}
            with tf.Session() as sess:
                sess.run(tf.tables_initializer())
                sess.run(train_iter.initializer)
                print(sess.run(train_src))
            src_h = tf.nn.bidirectional_dynamic_rnn(
                self.src_rnn_cell['f'],
                self.src_rnn_cell['w'],
                embedded_train_src,
                dtype=tf.int64, # 如果不给定RNN initial state，则必须给定dtype
                sequence_length=train_src_len)
            src_h = tf.concat(src_h, 2)
            tgt_h = tf.nn.bidirectional_dynamic_rnn(
                self.tgt_rnn_cell['f'],
                self.tgt_rnn_cell['w'],
                embedded_train_tgt,
                dtype=tf.int64,
                sequence_length=train_tgt_len)
            tgt_h = tf.concat(tgt_h, 2)
            h = tf.concat([src_h, tgt_h])
            with tf.Session() as sess:
                sess.run(tf.tables_initializer())
                sess.run(train_iter.initializer)
                print(sess.run(h))
            self.dense = tf.layers.Dense(units=out_size)
            h = self.dense(h)

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, nargs=3, help='Parallel training files and HTER score')
parser.add_argument('--vocab', type=str, nargs=2, help='Parallel vocab files (with ctrl)')
parser.add_argument('--dev', type=str, nargs=3, help='Parallel development files and HTER score')
parser.add_argument('--test', type=str, nargs=3, help='Parallel test files and HTER score')
args = parser.parse_args()

vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt, train_dataset, dev_dataset, test_dataset \
    = data_loader()
model = Model(train_dataset,
              dev_dataset,
              hidden_size=HIDDEN_SIZE,
              emb_size=EMB_SIZE,
              src_vocab_size=vocab_idx_src.size(),
              tgt_vocab_size=vocab_idx_tgt.(),
              out_size=OUT_SIZE)
