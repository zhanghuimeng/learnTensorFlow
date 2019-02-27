#!/usr/bin/env bash

data_dir=~/Documents/MT/projects/201810ineval_deen/data/processed
python_cmd="python"

while getopts "d:bqs" arg
do
    case $arg in
        d)
            if [[ "0" -le "$OPTARG" && "$OPTARG" -lt "8" ]]
            then
                export CUDA_VISIBLE_DEVICES=$OPTARG
                python_cmd="venv-gpu/bin/python -u"
            else
                python_cmd="venv/bin/python -u"
            fi
            ;;
        b)
            python seq2seq/build_vocab.py \
                --corpus "data_dir/corpus.tc.32k.en.shuf" \
                --name "data/test.en.vocab"
            ;;
        q)
            $python_cmd qe/qe.py \
                --train data/qe-2017/train.src data/qe-2017/train.mt data/qe-2017/train.hter \
                --vocab data/qe-2017/src.vocab data/qe-2017/tgt.vocab \
                --dev data/qe-2017/dev.src data/qe-2017/dev.mt data/qe-2017/dev.hter \
                --test data/qe-2017/test.src data/qe-2017/test.mt data/qe-2017/test.hter
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
