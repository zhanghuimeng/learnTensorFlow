import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load data
with open("../data/small_vocab_en.txt", "r", encoding="utf-8") as f:
    source_text = f.read()
with open("../data/small_vocab_fr.txt", "r", encoding="utf-8") as f:
    target_text = f.read()

# Build en & fr vocabulary & dictionary
source_vocab = list(set(source_text.lower().split()))
target_vocab = list(set(target_text.lower().split()))

print("English vocabulary size: %d" % len(source_vocab))
print("French vocabulary size: %d" % len(target_vocab))

SOURCE_CODES = ["<PAD>", "<UNK>"]
TARGET_CODES = ["<PAD>", "<EOS>", "<UNK>", "<GO>"]

source_vocab_to_int = {word: idx for idx, word in enumerate(SOURCE_CODES + source_vocab)}