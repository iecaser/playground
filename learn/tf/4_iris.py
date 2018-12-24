import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from sklearn.model_selection import train_test_split
tf.enable_eager_execution()

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                        origin=train_dataset_url)
test_url = "http://download.tensorflow.org/data/iris_test.csv"

test_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                          origin=test_url)
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
# data = pd.read_csv(train_dataset_fp)
# data.columns = column_names
# y, X = data.species, data.drop(columns='species')
# X_train, X_val, y_train, y_val = train_test_split(X, y)


class Model(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = keras.layers.Dense(units=10, activation=tf.nn.relu)
        self.fc2 = keras.layers.Dense(units=10, activation=tf.nn.relu)
        self.fc3 = keras.layers.Dense(units=3, activation=tf.nn.softmax)

    def call(self, input_tensor):
        x = self.fc1(input_tensor)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def pack_feature_vector(features, label):
    features = tf.stack(list(features.values()), axis=1)
    return features, label


model = Model()
dataset_train = tf.data.experimental.make_csv_dataset(file_pattern=train_dataset_fp,
                                                      batch_size=32,
                                                      column_names=column_names,
                                                      label_name=label_name,
                                                      num_epochs=1)

dataset_train = dataset_train.map(pack_feature_vector)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1)
global_step = tf.Variable(0)


def cal_acc(y, y_):
    batch_correct = tf.argmax(y_, axis=1).numpy() == y.numpy()
    acc = batch_correct.sum() / batch_correct.shape[0]
    return acc


EPOCHS = 100
for epoch in tqdm(range(EPOCHS)):
    epoch_loss = []
    epoch_acc = []
    for x, y in dataset_train:
        with tf.GradientTape() as tape:
            y_ = model(x)
            loss = tf.losses.sparse_softmax_cross_entropy(y, y_)
            acc = cal_acc(y, y_)
        gradient = tape.gradient(loss, model.trainable_variables)
        epoch_loss.append(loss.numpy())
        epoch_acc.append(acc)
        optimizer.apply_gradients(grads_and_vars=[*zip(gradient, model.trainable_variables)],
                                  global_step=global_step)
    tqdm.write('loss: {:.4f}, acc: {:.4f}'.format(np.mean(epoch_loss), np.mean(epoch_acc)))
