# encoding: utf-8
import tensorflow as tf
import argparse

from qe_dataset import read_vocab, one_dataset_loader
from qe_model import Model

BATCH_SIZE = 50
MAX_EPOCH = 500
PATIENCE = 5
EMB_SIZE = 300
ENC_HIDDEN_SIZE = 50
DEC_HIDDEN_SIZE = 500
LR = 1.0


parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, nargs=3, help='Parallel training files and HTER score')
parser.add_argument('--vocab', type=str, nargs=2, help='Parallel vocab files (with ctrl)')
parser.add_argument('--dev', type=str, nargs=3, help='Parallel development files and HTER score')
args = parser.parse_args()

# 据说把数据预处理放在CPU上是best practice
with tf.device('/cpu:0'):
    print("Loading vocabulary from %s and %s ..." % (args.vocab[0], args.vocab[1]))
    vocab_idx_src, vocab_idx_tgt, vocab_str_src, vocab_str_tgt = read_vocab(args.vocab[0], args.vocab[1])
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        src_vocab_size = sess.run(vocab_idx_src.size())
        tgt_vocab_size = sess.run(vocab_idx_tgt.size())
    print('Loaded src vocabulary size %d' % src_vocab_size)
    print('Loaded tgt vocabulary size %d' % tgt_vocab_size)
    train_dataset = one_dataset_loader(
        src=args.train[0],
        tgt=args.train[1],
        hter=args.train[2],
        vocab_idx_src=vocab_idx_src,
        vocab_idx_tgt=vocab_idx_tgt,
        batch_size=BATCH_SIZE)
    dev_dataset = one_dataset_loader(
        src=args.dev[0],
        tgt=args.dev[1],
        hter=args.dev[2],
        vocab_idx_src=vocab_idx_src,
        vocab_idx_tgt=vocab_idx_tgt,
        batch_size=BATCH_SIZE)
# print('Building computation model...')
train_iter = train_dataset.make_initializable_iterator()
train_ele = train_iter.get_next()
dev_iter = dev_dataset.make_initializable_iterator()
dev_ele = dev_iter.get_next()
model = Model(enc_hidden_size=ENC_HIDDEN_SIZE,
              dec_hidden_size=DEC_HIDDEN_SIZE,
              emb_size=EMB_SIZE,
              src_vocab_size=src_vocab_size,
              tgt_vocab_size=tgt_vocab_size,
              learning_rate=LR)
# print('Debugging: Built computation model...')
model.training(src=train_ele['src'],
               tgt=train_ele['tgt'],
               src_len=train_ele['src_len'],
               tgt_len=train_ele['tgt_len'],
               hter=train_ele['hter'])
model.testing(src=dev_ele['src'],
              tgt=dev_ele['tgt'],
              src_len=dev_ele['src_len'],
              tgt_len=dev_ele['tgt_len'],
              hter=dev_ele['hter'])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())  # for pearson op
    sess.run(tf.tables_initializer())
    # 打印到tensorboard
    # print('Debugging: Preparing tensorboard...')
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('logs', sess.graph)
    train_summary = tf.Summary()
    train_summary.value.add(tag='train loss', simple_value=None)
    dev_summary = tf.Summary()
    dev_summary.value.add(tag='dev mse', simple_value=None)
    dev_summary.value.add(tag='dev pearson', simple_value=None)

    # print('Debugging: Ready to start training...')
    step = 0
    no_improve_epochs = 0
    best_pearson = None
    for epoch in range(MAX_EPOCH):
        print("Epoch %d" % epoch)
        sess.run(train_iter.initializer)

        while True:
            try:
                loss, _ = sess.run([model.loss, model.train_op])
                print('Step %d: loss=%f' % (step, loss))
                train_summary.value[0].simple_value = loss
                writer.add_summary(train_summary, step)
                step += 1
            except tf.errors.OutOfRangeError:  # 到达epoch最后
                sess.run(dev_iter.initializer)
                sess.run(model.test_mse_reset)
                sess.run(model.test_pearson_reset)
                try:
                    while True:
                        mse, pearson = sess.run([model.test_mse_update, model.test_pearson_update])
                except tf.errors.OutOfRangeError:  # Thrown at the end of the epoch.
                    mse, pearson = sess.run([model.test_mse, model.test_pearson])
                    print('Epoch %d: mse=%f, pearson=%f' % (epoch, mse, pearson))
                    dev_summary.value[0].simple_value = mse
                    dev_summary.value[1].simple_value = pearson
                    writer.add_summary(dev_summary, step)
                    saver.save(sess, "model/epoch_%d.ckpt" % epoch)
                    # perform early stopping
                    if best_pearson is not None and pearson <= best_pearson:
                        no_improve_epochs += 1
                    else:
                        no_improve_epochs = 0
                        best_pearson = pearson
                    print('Patience: %d' % no_improve_epochs)
                    break

        writer.flush()
        if no_improve_epochs >= PATIENCE:
            print('Training finished')
            break
