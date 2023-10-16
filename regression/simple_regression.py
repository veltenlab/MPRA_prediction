import os 
import getopt
import sys

import numpy as np
import h5py
import pickle
import random
import copy
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, concatenate, Bidirectional, Dense, Dropout, Flatten, Conv1D,BatchNormalization,  MaxPooling1D, Bidirectional, GRU, TimeDistributed
import tensorflow as tf
from tensorflow import keras


args = argument_parser(argv_list=sys.argv[1:])

##################################################
### Initialize hyper parameters, CUDA and seed ###
##################################################

np.random.seed(1337) # for reproducibility
vocab = ["A", "G", "C", "T"]
indices = tf.range(len(vocab), dtype = tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab,indices)
table = tf.lookup.StaticVocabularyTable(table_init, 1)
defs = [0.] * 1 + [tf.constant([], dtype = "string")]

# Nadav dataset


def data_reader(file, batch_size=100, n_parse_threads = 4):
    dataset = tf.data.TextLineDataset(file).skip(1)
    dataset=dataset.map(preprocess, num_parallel_calls = n_parse_threads)
    return dataset.batch(batch_size).prefetch(1)

def preprocess(record):
    fields = tf.io.decode_csv(record, record_defaults=defs)
    chars = tf.strings.bytes_split(fields[2])
    chars_indeces = table.lookup(chars)
    X = tf.one_hot(chars_indeces, depth = len(vocab))
    Y = fields[1]
    ID = fields[0]
    
    return X,Y,ID

input_path_train = "/home/felix/cluster/fpacheco/Data/Nadav_lab/K562/mean_with_sequence_ENCFF616IAQ_2col_train.csv"
input_path_test = '/home/felix/cluster/fpacheco/Data/Nadav_lab/K562/mean_with_sequence_ENCFF616IAQ_test.csv'
input_path_valid = '/home/felix/cluster/fpacheco/Data/Nadav_lab/K562/mean_with_sequence_ENCFF616IAQ_validation.csv'


# Get first item of the dataset to get the shape of the input data
for element in data_reader(input_path_train):
    input_shape = element[0].shape

inputs = Input(shape=(input_shape[1],input_shape[2]), name="inputs")
layer = Conv1D(250, kernel_size=7, strides=1, activation='relu', name="conv1")(inputs)  # 250 7 relu
layer = Dropout(0.5)(layer)
layer = BatchNormalization()(layer)
layer = Conv1D(250, 8, strides=1, activation='softmax', name="conv2")(layer)  # 250 8 softmax
layer = BatchNormalization()(layer)
layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
layer = Dropout(0.5)(layer)
layer = Conv1D(250, 3, strides=1, activation='softmax', name="conv3")(layer)  # 250 3 softmax
layer = BatchNormalization()(layer)
layer = Dropout(0.5)(layer)
layer = Conv1D(100, 2, strides=1, activation='softmax', name="conv4")(layer)  # 100 3 softmax
layer = BatchNormalization()(layer)
layer = MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
layer = Dropout(0.5)(layer)
layer = Flatten()(layer)
layer = Dense(300, activation='sigmoid')(layer)  # 300
layer = Dropout(0.5)(layer)
layer = Dense(200, activation='sigmoid')(layer)  # 300
predictions = Dense(1, activation='linear')(layer)

model = Model(inputs=inputs, outputs=predictions)
model.summary()

model.compile(optimizer="adam",
              loss="mean_squared_error",
              metrics=["mse", "mae", "mape"],
              )

history=model.fit(data_reader(input_path_train, batch_size=1024),
                        epochs=50,
                        validation_data=data_reader(input_path_valid,batch_size=100),
                        callbacks=None,
                        verbose=1)

predicted = model.predict(data_reader(input_path_test,
                                            batch_size=100))

test_data = data_reader(input_path_test,batch_size=100)
test_tensor = X = np.empty(shape=[0,1])
for batch in test_data:
    test_tensor = np.append(test_tensor, batch[1])

import math
def pearson_correlation(x, y):
    n = len(x)
    # Calculate the mean of x and y
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate the numerator and denominators of the correlation coefficient
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denominator_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    # Calculate the correlation coefficient
    correlation = numerator / (denominator_x * denominator_y)
    return correlation
    
corr_coefficient = pearson_correlation(predicted.flatten(), test_tensor)