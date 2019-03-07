#!/usr/bin/env bash

exclude=""
while getopts "e" arg
do
    case $arg in
        e)
            exclude="--exclude '.idea' --exclude '.git'"
        ;;
        ?)
            echo "unknown argument"
            exit 1
        ;;
    esac
done

rsync -avz $exclude \
    ~/Documents/learnTensorFlow/ 242:/data/disk2/private/zhm/201902_learnTF/learnTensorFlow
