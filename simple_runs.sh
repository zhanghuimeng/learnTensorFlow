#!/usr/bin/env bash

data_dir=~/Documents/MT/projects/201810ineval_deen/data/processed
python_cmd="python -u"
set +x  # echo on

while getopts "bqst" arg
do
    case $arg in
        b)
            python seq2seq/build_vocab.py \
                --corpus "data_dir/corpus.tc.32k.en.shuf" \
                --name "data/test.en.vocab"
            ;;
        q)
            $python_cmd qe/qe_training.py \
                --train data/qe-2017/train.src data/qe-2017/train.mt data/qe-2017/train.hter \
                --vocab data/qe-2017/src.vocab data/qe-2017/tgt.vocab \
                --dev data/qe-2017/dev.src data/qe-2017/dev.mt data/qe-2017/dev.hter \
                --model_dir model/qe/
            ;;
        t)
            #--test data/qe-2017/dev.src data/qe-2017/dev.mt data/qe-2017/dev.hter \
            $python_cmd qe/qe_test.py \
                --vocab data/qe-2017/src.vocab data/qe-2017/tgt.vocab \
                 --test data/qe-2017/test.src data/qe-2017/test.mt data/qe-2017/test.hter \
                --model model/qe3/qe.ckpt-13800 \
                --output test.hter
            ;;
        s)
            $python_cmd seq2seq/seq2seq.py \
                --training "$data_dir/corpus.tc.32k.en.shuf" "$data_dir/corpus.tc.32k.de.shuf" \
                --vocab "$data_dir/vocab.32k.en.txt" "$data_dir/vocab.32k.de.txt" \
                --dev "$data_dir/newstest2014.tc.32k.en" "$data_dir/newstest2014.tc.32k.de"
            ;;
        ?)
            echo "unknown argument"
            exit 1
        ;;
    esac
done
