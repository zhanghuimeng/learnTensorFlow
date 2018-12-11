import numpy as np
from scipy.special import expit
import math
EPS = 1e-9
MAX_EXP = 709

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def sigmoid(x):
    x = np.clip(x, -MAX_EXP, None)
    return 1 / (1 + np.exp(-x))

def cross_entropy(y, y_hat):
    y_hat = np.clip(y_hat, EPS, 1-EPS)
    return -np.sum(y * np.log(y_hat))

def softmax_cross_entropy(x, y):
    max_x = np.max(x)
    log_exp = max_x + np.log(np.sum(np.exp(x - max_x)))
    return -np.sum(x * y) + np.sum(y) * log_exp

def sigmoid_cross_entropy(x, y):
    for xi in np.nditer(x, op_flags=['readwrite']):
        if xi < -MAX_EXP:
            xi[...] = -xi
        else:
            xi[...] = math.log(1 + math.exp(-xi))
    return np.sum(y * x)


x = np.array([1, 1, 1, 4000])
y = np.array([1, 0, 0, 0])
print(softmax(x))
print(cross_entropy(y, softmax(x)))
print(softmax_cross_entropy(x, y))
# outputs:
# [0. 0. 0. 1.]
# 20.72326583694641
# 3999.0

x = np.array([1, 1, -4000, -4000])
y = np.array([0, 0, 0, 1])
print(sigmoid(x))
print(expit(x))
print(cross_entropy(y, sigmoid(x)))
print(cross_entropy(y, expit(x)))
print(sigmoid_cross_entropy(x, y))
# outputs:
# [7.31058579e-001 7.31058579e-001 1.21678075e-308 1.21678075e-308]
# [0.73105858 0.73105858 0.         0.        ]
# 20.72326583694641
# 20.72326583694641
# 4000