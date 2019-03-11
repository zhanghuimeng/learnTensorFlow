#!/usr/bin/env bash
rsync -avz --exclude '.idea' --exclude '.git' -e ssh \
    242:/data/disk2/private/zhm/201902_learnTF/learnTensorFlow/ ~/Documents/learnTensorFlow
