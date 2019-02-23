import numpy as np
import tensorflow as tf
import argparse


def data_preprocess():
    training_src = tf.data.TextLineDataset(args.training[0])
    training_tgt = tf.data.TextLineDataset(args.training[1])
    training_src = training_src.map(lambda string: tf.string_split([string]).values)
    training_tgt = training_tgt.map(lambda string: tf.string_split([string]).values)
    vocab_src = tf.contrib.lookup.index_table_from_file(args.vocab[0], num_oov_buckets=1)
    vocab_tgt = tf.contrib.lookup.index_table_from_file(args.vocab[1], num_oov_buckets=1)
    tgt_go_id = vocab_src.lookup(tf.constant('<go>'))
    tgt_eos_id = vocab_src.lookup(tf.constant('<eos>'))
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        print(sess.run(tgt_go_id))
        print(sess.run(tgt_eos_id))
    training_src = training_src.map(lambda tokens: vocab_src.lookup(tokens))
    training_src_len = training_src.map(lambda tokens: tf.size(tokens))
    training_tgt_input = training_tgt.map(
        lambda tokens: tf.stack([tgt_go_id, vocab_tgt.lookup(tokens)], 0))
    training_tgt_output = training_tgt.map(
        lambda tokens: tf.stack([vocab_tgt.lookup(tokens), tgt_eos_id], 0))
    training_tgt_len = training_src.map(lambda tokens: tf.size(tokens) + 1)
    training_dataset = tf.data.Dataset.zip({
        "src": training_src,
        "src_len": training_src_len,
        "tgt_input": training_tgt_input,
        "tgt_output": training_tgt_output,
        "tgt_len": training_tgt_len
    })

    # 分batch并进行智能padding
    src_pad_id = vocab_src.lookup(tf.constant('<pad>'))
    tgt_pad_id = vocab_tgt.lookup(tf.constant('<pad>'))
    padded_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
    padding_values = (src_pad_id, tgt_pad_id)
    training_dataset = (training_dataset
                        .shuffle(buffer_size=10000)
                        .padded_batch(args.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
                        )
    iterator = training_dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())  # 为何需要explicitly initialize这个呢？
        sess.run(iterator.initializer)
        for i in range(2):
            print(sess.run(next_element))


def get_hparams():
    """
    返回本次训练所需的超参数，用户指定的参数将覆盖默认参数
    """
    hparams = tf.contrib.training.HParams(
        learning_rate=1.0,
        batch_size=128,  # 每个batch中句子个数
        max_length=256,  # 句子最大长度
        train_steps=100000,  # 总训练步数
        save_checkpoint_steps=1000,  # 多久保存一次checkpoint
        keep_checkpoint_max=20,  # 最多保存几个checkpoint
        keep_top_checkpoint_max=5,  # 最多保存几个表现最好的checkpoint
        eval_steps=2000,  # 多久在开发集上验证一次
        eval_batch_size=32,  # 验证时每个batch中句子个数
        embedding_size=1000,  # 源端和译端的embedding大小
        hidden_size=1000,
    )

    hparams.parse(args.hparams)
    return hparams


parser = argparse.ArgumentParser()
parser.add_argument('--training', type=str, nargs=2, help='Parallel training files')
parser.add_argument('--vocab', type=str, nargs=2, help='Vocabularies')
parser.add_argument('--dev', type=str, nargs=2, help='Parallel development files')
parser.add_argument('--hparams', type=str, default='', help='Hyperparameters')
args = parser.parse_args()
hparams = get_hparams()

# 数据预处理
data_preprocess()