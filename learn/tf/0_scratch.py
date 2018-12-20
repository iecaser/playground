import tensorflow as tf
import tempfile

tf.enable_eager_execution()
a = tf.add(1, 2)
b = tf.add([1, 2], [3, 4])
x = tf.random_uniform([3, 3])

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

file = tempfile.mktemp()
with open(file, 'w') as f:
    f.write('line1\n'
            'line2\n'
            'line3\n'
            )

ds_file = tf.data.TextLineDataset(file)
ds_file = ds_file.batch(2)

for df in ds_file:
    print(df)

x = tf.ones((2, 2))
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
dz_dx = t.gradient(z, x)


def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(x, output)
    return output


def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        output = f(x, y)
    return t.gradient(output, x)


x = tf.constant(2.0)
tf.convert_to_tensor(2.0)
r3 = grad(x, 4)
r1 = grad(x, 5)
r2 = grad(x, 6)

x = tf.zeros((10, 10))
x += 2
