import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
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
        self.fc1 = keras.layers.Dense(units=32, activation=tf.nn.relu)
        self.fc2 = keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.fc3 = keras.layers.Dense(units=3, activation=tf.nn.softmax)

    def call(self, input_tensor):
        x = self.fc1(input_tensor)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = Model()
dataset_train = tf.data.experimental.make_csv_dataset(file_pattern=train_dataset_fp,
                                                      batch_size=32,
                                                      column_names=column_names,
                                                      label_name=label_name,
                                                      num_epochs=1)
for features, label in dataset_train:
    print(features)
    print(label)
    break


def pack_feature_vector(features, label):
    features = tf.stack(list(features.values()), axis=1)
    return features, label


dataset_train = dataset_train.map(pack_feature_vector)
