mkdir -p data &&																		\
    unzip -nd data WMT18\ Quality\ Estimation\ Shared\ Task\ Training\ and\ Development\ Data.zip  sentence_level_training.tar.gz word_level_training.tar.gz &&	\
    tar -xzf data/word_level_training.tar.gz -C data  &&													\
    tar -xzf data/sentence_level_training.tar.gz  -C data  &&													\
    cd en_de.smt_models/ &&																	\
    bash reproduce_numbers.sh
