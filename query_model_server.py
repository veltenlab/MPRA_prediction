import os 
import getopt
import getopt
import sys


import random
import re

import numpy as np
import h5py
import pickle
import random
import copy
import pandas as pd
import math 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Masking, Lambda, concatenate, Bidirectional, Dense, Dropout, Flatten, Conv1D,BatchNormalization,  MaxPooling1D, Bidirectional, GRU, TimeDistributed, Concatenate
import tensorflow as tf
from tensorflow import keras

import socket


options, remainder = getopt.getopt(sys.argv[1:], 'p:', ['port='])

port = 4568
host = '0.0.0.0'  # Localhost

for opt, arg in options:
    if opt in ('-p', '--port'):
        port = int(arg)
        
### 1. Setup functions ####

np.random.seed(1337) # for reproducibility
vocab = ["A", "G", "C", "T"]
indices = tf.range(len(vocab), dtype = tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab,indices)
table = tf.lookup.StaticVocabularyTable(table_init, 1)


batches = ["LibA","LibH","LibB","LibV"]
bindices = tf.range(len(batches), dtype = tf.int64)
btable_init = tf.lookup.KeyValueTensorInitializer(batches,bindices)
btable = tf.lookup.StaticVocabularyTable(btable_init, 1)

record_defaults = [
    tf.constant([''], dtype=tf.string),
    tf.constant([''], dtype=tf.string),
    tf.constant([''], dtype=tf.string),
    tf.constant([''], dtype=tf.string),
    tf.constant([''], dtype=tf.string),  
    tf.constant([''], dtype=tf.string),
    tf.constant([''], dtype=tf.string),
    tf.constant([''], dtype=tf.string),  
    tf.constant([''], dtype=tf.string),
    tf.constant([''], dtype=tf.string),
]

# Nadav dataset

def data_reader(file, batch_size=100, n_parse_threads=8):
    """Method for reading the data in an optimized way, can be used inside model.fit()
    
    Args:
        file (_type_): path to csv file
        batch_size (int, optional): _description_. Defaults to 100.
        n_parse_threads (int, optional): _description_. Defaults to 4.

    Returns:
        dataset.batch: batch dataset object 
    """
    dataset = tf.data.TextLineDataset(file).skip(1)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    return dataset.batch(batch_size).prefetch(1)

def preprocess(record):
    """Preprocessing method of a dataset object, one-hot-encodes the data

    Args:
        record (_type_): _description_

    Returns:
        X (2D np.array): one-hot-encoded input sequence
        Y (1D np.array): MPRA measurements for each cell state
    """
    # Get fields from the data
    fields = tf.io.decode_csv(record, record_defaults=record_defaults)
    
    # One-hot-encode data
    chars = tf.strings.bytes_split(fields[0])
    chars_indeces = table.lookup(chars)
    batch_indeces = btable.lookup(fields[2])
    
    X = tf.one_hot(chars_indeces, depth = len(vocab))
    B = tf.one_hot(batch_indeces, depth = len(batches))
    # Combine y for each cell type into one vector 
    Y = tf.stack(fields[3:])
    
    # Replace missing values with -1
    Y= tf.where(tf.equal(Y,  "nan"), ["-1"], Y)
    Y= tf.where(tf.equal(Y,  "NA"), ["-1"], Y)
    Y = tf.strings.to_number(Y, tf.float32)
    return (X,B),Y

# Get first item of the dataset to get the shape of the input data
for element in data_reader("data.all/complete_data.csv"):
    input_shape = element[0][0].shape
    output_shape = element[1].shape
    batch_shape = element[0][1].shape
    
# Here we initialize the df where each fold test prediction will be appended to
# the list containing the correlations of each fold is also initialized
corr_list = []

# We define a custom normalization layer to then compile on the model
class CustomNormalization(Layer):
    """Custom normalization layer that normalizes the output of the neural network"""
    def __init__(self, **kwargs):
        super(CustomNormalization, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Add trainable variables for mean and standard deviation
        self.mean = self.add_weight("mean", shape=(1,), initializer="zeros", trainable=True)
        self.stddev = self.add_weight("stddev", shape=(1,), initializer="ones", trainable=True)
        super(CustomNormalization, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        # Normalize the inputs using the learned mean and standard deviation
        return (inputs - self.mean) / (self.stddev + 1e-8)

# We define the method to compute the pearson correlation between prediction and ground truth in the multi_head case
def pearson_correlation_multi_head(predictions, ground_truth, mask_value=-1):
    """Computes Pearson Correlation between predictions and ground truth for each column
    Args:
        predictions (np.array): 2D array of prediction values (N, 7)
        ground_truth (np.array): 2D array of ground truth values (N, 7)
        mask_value (float): Value in ground truth to be ignored in correlation computation

    Returns:
        correlations (np.array): 1D array of Pearson correlations for each column
    """
    # Ensure predictions and ground_truth have the same shape
    assert predictions.shape == ground_truth.shape, "Input shapes do not match"

    n_columns = predictions.shape[1]
    correlations = np.zeros(n_columns)

    for col in range(n_columns):
        x = predictions[:, col]
        y = ground_truth[:, col]

        # Exclude values in ground truth equal to mask_value
        valid_indices = (y != mask_value)
        x = x[valid_indices]
        y = y[valid_indices]

        if len(x) == 0 or len(y) == 0:
            # If no valid values, set correlation to NaN
            correlations[col] = np.nan
        else:
            # Calculate mean of x and y
            mean_x = np.mean(x)
            mean_y = np.mean(y)

            # Calculate the numerator and denominators of the correlation coefficient
            numerator = np.sum((x - mean_x) * (y - mean_y))
            denominator_x = np.sqrt(np.sum((x - mean_x) ** 2))
            denominator_y = np.sqrt(np.sum((y - mean_y) ** 2))

            # Calculate the correlation coefficient
            correlation = numerator / (denominator_x * denominator_y)
            correlations[col] = correlation

    return correlations



import matplotlib.pyplot as plt
# Define plotting function of loss
def create_plots(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.clf()


# Define a custom loss function
class MaskedMSE(tf.keras.losses.Loss):
    """Computes the MSE loss and prevents missing values backpropagation (previously replaced by -1.0)

    Args:
        tf (_type_): _description_
    """
    def __init__(self, mask_value=-1, **kwargs):
        super(MaskedMSE, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, y_true, y_pred):
        # Create a mask for valid elements (not equal to the specified mask_value)
        mask = tf.math.not_equal(y_true, self.mask_value)

        # Compute MSE loss only for valid elements
        loss = tf.where(mask, tf.square(y_true - y_pred), 0.0)

        # Calculate the mean loss
        mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32))

        return mean_loss




from tensorflow.keras.optimizers import Adam

corr_coefficients = {}

df_test_10folds  = pd.DataFrame(columns=['State_3E',
                                         "seq",
                                         "avg_prediction_State_1M",
                                         "avg_prediction_State_2D",
                                         "avg_prediction_State_3E",
                                         "avg_prediction_State_4M",
                                         "avg_prediction_State_5M",
                                         "avg_prediction_State_6N",
                                         "avg_prediction_State_7M"])

# We iterate through each of the train folds to train, test and validate the model

#this is just to determine the input shape
i=1
input_path_train = "data.all/all_train_"+str(i)+".csv"
input_path_valid = "data.all/all_valid_"+str(i)+".csv"
input_path_test = "data.all/all_test_"+str(i)+".csv"

# Read test data to then predict
df_test = pd.read_csv(input_path_test)
df_test["fold"] = str(i)

predictions_sum = np.zeros((df_test.shape[0], 7))

# Define and compile model
inputs = Input(shape=(input_shape[1],input_shape[2]), name="inputs")
batchinput = Input(shape=(batch_shape[1]))
layer = Conv1D(250, kernel_size=7, strides=1, activation='relu', name="conv1")(inputs)  # 250 7 relu
layer = Dropout(0.5)(layer)
layer = BatchNormalization()(layer)
layer = Conv1D(250, 8, strides=1, activation='softmax', name="conv2")(layer)  # 250 8 softmax
layer = BatchNormalization()(layer)
layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
layer = Dropout(0.3)(layer)
layer = Conv1D(250, 3, strides=1, activation='softmax', name="conv3")(layer)  # 250 3 softmax
layer = BatchNormalization()(layer)
layer = Dropout(0.5)(layer)
layer = Conv1D(100, 2, strides=1, activation='softmax', name="conv4")(layer)  # 100 3 softmax
layer = BatchNormalization()(layer)
layer = MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
layer = Dropout(0.5)(layer)
layer = Flatten()(layer)
layer = Concatenate()([layer, batchinput])

layer = Dense(300, activation='sigmoid')(layer)  # 300
layer = Dropout(0.5)(layer)

layer = Dense(200, activation='sigmoid')(layer)  # 300
predictions = Dense(7, activation='linear')(layer)
norm_predictions = CustomNormalization()(predictions)  # Assuming "predictions" is your existing output

model = Model(inputs=[inputs,batchinput], outputs=norm_predictions)

# compile model
model.compile(optimizer= Adam(clipvalue=1.0), loss=MaskedMSE(mask_value=-1),metrics=["mse"])

#the idea here is to run an ensembl of 10 models , trained on random subsets of the data and with random inits

wpath='weights.all/'
weights = os.listdir(wpath)
weights = random.sample(weights, 10)
#weights = ["fold_1_ens_"+str(i)+".h5" for i in range(1,11)]

ensembl = [keras.models.clone_model(model) for i in range(10)]
for i in range(10):
    ensembl[i].load_weights(wpath+weights[i])

def seq2tensors(s):
    chars = tf.strings.bytes_split(s)
    chars_indeces = table.lookup(chars)
    X = tf.one_hot(chars_indeces, depth = len(vocab))
    return(X)
  
def batch2tensors(s):
    chars_indeces = btable.lookup(tf.constant(s, dtype = tf.string))
    B = tf.one_hot(chars_indeces, depth = len(batches))
    return(B)

def tensor_to_string(tensor):
    # Ensure the tensor is 1D
    if len(tensor.shape) != 1:
        raise ValueError("Input tensor must be 1D")

    # Convert tensor to a numpy array and then to a list
    tensor_list = tensor.numpy().tolist()

    # Convert list elements to strings and join them with commas
    return ','.join(map(str, tensor_list)) + '\n'

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen()
    print(f"Server is listening on {host}:{port}", file=sys.stderr)

    while True:
        conn, addr = s.accept()
        conn.settimeout(60)
        with conn:
            try:
                print(f"Connected by {addr}")
                data = conn.recv(2) #2 characters: integer
                recdata = data.decode().rstrip()
                ensembl_size = int(recdata)
                while True:
                    data = conn.recv(280)
                    recdata = data.decode().rstrip().split(",")
                    if not data:
                        break
                    x = seq2tensors(recdata[0])
                    x = x[tf.newaxis, ...]
                    b = batch2tensors(recdata[1])
                    b = b[tf.newaxis, ...]
                    for i in range(ensembl_size):
                        conn.sendall(tensor_to_string(ensembl[i]((x, b))[0,:]).encode())
            except socket.timeout:
                print(f"Connection timed out for {addr}")
            except Exception as e:
                print(f"An error occurred: {e}")
