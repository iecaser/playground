import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()


class MyDense(tf.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.kernel = None

    def build(self, input_shape):
        super().build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.num_outputs))

    def call(self, x):
        # return tf.multiply(x, self.kernel) # WRONG!!
        return tf.matmul(x, self.kernel)


layer = MyDense(10)
x = tf.zeros((10, 5))
y_ = layer(x)


class ResNetBlock(keras.models.Model):
    def __init__(self, kernel_size, filters):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1))
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=filters[1], kernel_size=kernel_size, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(filters=filters[2], kernel_size=(1, 1))
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, input_tensor, training=True):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += input_tensor
        x = tf.nn.relu(x)
        return x


block = ResNetBlock(kernel_size=1,
                    filters=(1, 2, 3))
x = tf.zeros(shape=(1, 2, 3, 3))
y_ = block(x)
