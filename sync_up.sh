#!/usr/bin/env bash

exclude=""
addr="zhm@166.111.5.234:/data/disk4/private/zhm/201902learnTF/learnTensorFlow"
while getopts "et:" arg
do
    case $arg in
        e)
            exclude="--exclude '.idea' --exclude '.git'"
        ;;
        t)
            if [[ "242" -eq "$OPTARG" ]]
            then
                addr="242:/data/disk2/private/zhm/201902_learnTF/learnTensorFlow"
            fi
        ;;
        ?)
            echo "unknown argument"
            exit 1
        ;;
    esac
done

rsync -avz -e ssh $exclude \
    ~/Documents/learnTensorFlow/ $addr
