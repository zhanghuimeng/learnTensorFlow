import tensorflow as tf
tf.enable_eager_execution()

a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)  # It's ok to write c = a + b

print(c)

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)

# Tensor: shape & dtype
