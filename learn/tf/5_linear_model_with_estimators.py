# sys.path.append('/home/zxf/workspace/models')
# from official.wide_deep import census_main
# from official.wide_deep import census_dataset
import tensorflow as tf
import pandas as pd
import tensorflow.feature_column as fc
import sys
import os
import functools
from loguru import logger
from pprint import pprint

tf.enable_eager_execution()


_CSV_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                'marital_status', 'occupation', 'relationship', 'race', 'gender',
                'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                'income_bracket']
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]
TRAIN_SIZE = 32561
data_path = '/home/zxf/data/census'
train_path = os.path.join(data_path, 'adult.data')
test_path = os.path.join(data_path, 'adult.test')
NUM_EPOCHS = 1
BATCH_SIZE = 64


def get_dataset(file_path, num_epochs, batch_size, shuffle=True):
    def parse_csv(records):
        data = tf.decode_csv(records=records,
                             record_defaults=_CSV_COLUMN_DEFAULTS)
        X = dict(zip(_CSV_COLUMNS, data))
        y = tf.equal(X.pop('income_bracket'), '>50K')
        return X, y
    ds = tf.data.TextLineDataset(filenames=[file_path])
    if shuffle:
        ds = ds.shuffle(TRAIN_SIZE)
    ds = ds.map(map_func=parse_csv, num_parallel_calls=5).repeat(num_epochs).batch(batch_size)
    return ds


get_train_dataset = functools.partial(get_dataset, file_path=train_path,
                                      num_epochs=3, batch_size=BATCH_SIZE, shuffle=True)
get_test_dataset = functools.partial(get_dataset, file_path=test_path,
                                     num_epochs=1, batch_size=BATCH_SIZE, shuffle=False)

# ds_train = get_train_dataset()
# ds_test = get_test_dataset()

# for Xb, yb in ds_train.take(1):
#     print(Xb.keys())
#     print(yb)

# numerical
age = fc.numeric_column('age')
age_bucket = fc.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 70])
education_num = fc.numeric_column('education_num')
capital_gain = fc.numeric_column('capital_gain')
capital_loss = fc.numeric_column('capital_loss')
hours_per_week = fc.numeric_column('hours_per_week')
numeric_columns = [age, age_bucket, education_num, capital_gain, capital_loss, hours_per_week]

# categorical
relationship = fc.categorical_column_with_vocabulary_list('relationship',
                                                          ['Husband', 'Not-in-family', 'Wife',
                                                           'Own-child', 'Unmarried', 'Other-relative'])
occupation = fc.categorical_column_with_hash_bucket('occupation',
                                                    hash_bucket_size=1000)
education = fc.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'
    ])
marital_status = fc.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'
    ])
workclass = fc.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'
    ])
categorical_columns = [relationship, occupation, education, marital_status, workclass]

# crossed
education_x_occupation = fc.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000)
crossed_columns = [education_x_occupation]

# train
classifier = tf.estimator.LinearClassifier(
    feature_columns=numeric_columns + categorical_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
                                     l1_regularization_strength=0.1,
                                     l2_regularization_strength=0.1))
classifier.train(get_train_dataset)
result = classifier.evaluate(get_test_dataset)
pprint(result)
