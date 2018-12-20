import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(4.0)
# c = tf.multiply(a, b)
c = a*b
with tf.Session() as sess:
    result = sess.run(c)
    print(result)
