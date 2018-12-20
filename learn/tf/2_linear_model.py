import tensorflow as tf
import matplotlib.pyplot as plt
import pdb

tf.enable_eager_execution()


class Model(object):
    def __init__(self):
        self.W = tf.Variable(initial_value=0.0)
        self.b = tf.Variable(initial_value=0.0)

    def __call__(self, X):
        return self.W*X + self.b


def loss_fn(y_, y):
    return tf.reduce_mean(tf.square(y_-y))


# params
EPOCHS = 50
lr = .1

X = tf.random_normal(shape=(1000,))
W, b = 3.5, 2.5
y = W*X + b


Ws, bs = [], []
model = Model()
for epoch in range(EPOCHS):
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    # y_ = model(X) # WRONG PLACE!!
    with tf.GradientTape() as t:
        y_ = model(X)
        loss = loss_fn(y_, y)
    dW, db = t.gradient(loss, [model.W, model.b])
    model.W.assign_sub(dW * lr)
    model.b.assign_sub(db * lr)
plt.figure()
plt.plot(Ws, 'b')
plt.plot([W]*EPOCHS, 'b--')
plt.plot(bs, 'r')
plt.plot([b]*EPOCHS, 'r--')
plt.show()
plt.savefig('linear_model')
