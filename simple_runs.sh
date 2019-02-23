#!/usr/bin/env bash

DATA_DIR=~/Documents/MT/projects/201810ineval_deen/data/processed

while getopts "bs" arg
do
    case $arg in
        b)
            python seq2seq/build_vocab.py \
                --corpus "$DATA_DIR/corpus.tc.32k.en.shuf" \
                --name "data/test.en.vocab"
            ;;
        s)
            python seq2seq/seq2seq.py \
                --training "$DATA_DIR/corpus.tc.32k.en.shuf" "$DATA_DIR/corpus.tc.32k.de.shuf" \
                --vocab "$DATA_DIR/vocab.32k.en.txt" "$DATA_DIR/vocab.32k.de.txt" \
                --dev "$DATA_DIR/newstest2014.tc.32k.en" "$DATA_DIR/newstest2014.tc.32k.de"
            ;;
        ?)
            echo "unknown argument"
            exit 1
        ;;
    esac
done
