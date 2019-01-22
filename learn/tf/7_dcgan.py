import tensorflow as tf

tf.enable_eager_execution()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
