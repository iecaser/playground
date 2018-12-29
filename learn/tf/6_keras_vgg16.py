import tensorflow as tf


def conv3(filters):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                  padding='same', activation='relu')


def max_pool():
    return tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))


img_input = tf.keras.layers.Input(shape=(224, 224, 3))
# block 1
x = conv3(64)(img_input)
x = conv3(64)(x)
x = max_pool()(x)

# block 2
x = conv3(128)(x)
x = conv3(128)(x)
x = max_pool()(x)

# block 3
x = conv3(256)(x)
x = conv3(256)(x)
x = conv3(256)(x)
x = max_pool()(x)

# block 4
x = conv3(512)(x)
x = conv3(512)(x)
x = conv3(512)(x)
x = max_pool()(x)

# block 5
x = conv3(512)(x)
x = conv3(512)(x)
x = conv3(512)(x)
x = max_pool()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dense(1000, activation='relu')(x)
model = tf.keras.models.Model(img_input, x)
