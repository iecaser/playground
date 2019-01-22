import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def model1(X, w):
    return tf.multiply(X, w)


def model2(X, w):
    terms = [tf.multiply(w[i], tf.pow(X, i)) for i in range(5)]
    return tf.add_n(terms)


def f1(x):
    y = 2 * x + np.random.randn(*x.shape) * 0.33
    return y


def f2(x):
    y = 0.0
    coes = np.arange(6)
    for i, c in enumerate(coes):
        y += c * x ** i
    y += np.random.randn(*x.shape) * 1.5
    return y


lr = 0.01
epochs = 100
x_train = np.linspace(-1, 1, 101)
y_train = f2(x_train)
# tf
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w = tf.Variable([0.0] * 5, trainable=True, name='weights')
y_ = model2(X, w)
loss = (tf.square(Y - y_) + 0.2 * tf.reduce_sum(tf.square(w))) / x_train.shape[0]
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(epochs):
    y_vals = []
    for (x, y) in zip(x_train, y_train):
        y_val, loss_val, __ = sess.run((y_, loss, train_op), feed_dict={X: x, Y: y})
        y_vals.append(y_val)
    y_vals = np.asarray(y_vals)
w_val = sess.run(w)
sess.close()
plt.figure()
plt.scatter(x_train, y_train)
y_learned = np.zeros_like(x_train)
plt.plot(x_train, y_vals, 'r')
plt.show()
