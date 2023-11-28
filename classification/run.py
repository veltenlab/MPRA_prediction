import sys
import optparse
from array import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import tensorflow
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.utils import class_weight, shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from functools import partial

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Bidirectional, Concatenate, PReLU, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers import Layer, average, Input, Lambda, concatenate
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from client_fly_brain.client import argument_parser
from keras import backend as K

#import keras_tuner as kt
from tensorflow import keras
import glob
import pandas as pd
import os
import re
import shap

args = argument_parser(argv_list=sys.argv[1:])

##################################################
### Initialize hyper parameters, CUDA and seed ###
##################################################

# Hyperparams
SEED = args.seed
BATCH = args.batch_size
INFILE = args.infile
OUT = args.out
DATASET = args.name
PATIENCE = args.patience
NUM_CLASSES = args.classes
CV = args.cv
selected_classes = np.array(list(range(NUM_CLASSES)))

EPOCH = args.max_epochs
PWM_PATH = args.pwm
CHANNELS = args.channels_conv
WIDTH_DENSE = args.width_dense
REGRESSION = args.regression
MULTI_REGRESSION = args.multiheadregression
CNN_NUM = int(args.cnn_num)
DENSE_NUM = int(args.dense_num)
GPU = args.gpu
HP_TUNNING= args.hp_tunning


# Set seed
tf.random.set_random_seed(SEED)

if GPU is False :
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

CLASSIFICATION = False
if REGRESSION is False and MULTI_REGRESSION is False:
    CLASSIFICATION  = True