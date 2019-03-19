#!/usr/bin/env bash
model_base_dir="en_de.smt_models/estimator/target_1"
data_dir="data/WMT17/sentence_level/en_de"
#kiwi predict --config ${model_base_dir}/predict.yaml	\
#		 --experiment-name "Official run for OpenKiwi"	\
#		 --load-model ${model_base_dir}/model.torch		\
#		 --output-dir "kiwi_out"			\
#		 --gpu-id -1				\
#		 --test-source ${data_dir}/test.2017.src		\
#		 --test-target ${data_dir}/test.2017.mt		\
#		 # --quiet
kiwi evaluate					\
     --pred-sents "kiwi_out/sentence_scores"	\
     --gold-sents ${data_dir}/en-de_task1_test.2017.hter		\
     --format wmt17				\
     --pred-format wmt17