import pdb
import numpy as np
import tensorflow as tf

MAX_LEN = 32
MAX_FEATURES = 128
EMBED_DIM = 64
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_FEATURES)
samples = ['The cat sat on the mat the.', 'The dog ate my homework cat.', 'Hello world.']
tokenizer.fit_on_texts(samples)
x_train = tokenizer.texts_to_sequences(samples)
x_train_ = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=MAX_FEATURES,
                                    output_dim=EMBED_DIM, input_length=MAX_LEN))
x = np.random.randint(MAX_FEATURES, size=(32, MAX_LEN))
model.compile('rmsprop', 'mse')
y = model.predict(x)
