import tensorflow as tf

matrix1 = tf.constant([[2, 2]])
matrix2 = tf.constant([[3], [3]])
product = tf.matmul(matrix1, matrix2)

### method 1
sess = tf.Session()
print("method 1:", sess.run(product))
sess.close()

### method 2
with tf.Session() as sess:
    print("method 2:", sess.run(product))