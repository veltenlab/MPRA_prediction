## Define functions
import sys
import optparse
from array import *

import tensorflow
import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import average_precision_score, roc_auc_score

from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Bidirectional, BatchNormalization
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers import Input, Lambda, concatenate
from keras.models import Model
import keras.backend as K
import tensorflow as tf

from keras.optimizers import Adam

from sklearn.model_selection import KFold

from keras import backend as K
import glob
import pandas as pd
import os
import re
import math

## Define functions
def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

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

def pearson_r(y_true, y_pred):
    # use smoothing for not resulting in NaN values
    # pearson correlation coefficient
    # https://github.com/WenYanger/Keras_Metrics
    epsilon = 10e-5
    x = y_true
    y = y_pred[:,0]
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)

def build_model(CHANNELS, width_dense, REGRESSION, MULTI_REGRESSION, CNN_NUM, DENSE_NUM, CLASSIFICATION, seq_shape, PWM_PATH):
    reverse_lambda_ax2 = Lambda(lambda x: K.reverse(x,axes=2))
    reverse_lambda_ax1 = Lambda(lambda x: K.reverse(x,axes=1))
    forward_input = Input(shape=seq_shape)
    reverse_input = Input(shape=seq_shape)
        
    if CNN_NUM == 1 and DENSE_NUM==1:
        layer0 = [
        Conv1D(CHANNELS, kernel_size=24, padding="valid", activation='relu', kernel_initializer='random_uniform'),
        MaxPooling1D(pool_size=12, strides=12, padding='valid'),
        Dropout(0.5),
        TimeDistributed(Dense(128, activation='relu')),
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
        Dropout(0.5),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),]
        
    elif CNN_NUM==2 and DENSE_NUM==1:
        layer0 = [
            Conv1D(CHANNELS, kernel_size=24, padding="valid", activation='relu', kernel_initializer='random_uniform'),
            MaxPooling1D(pool_size=12, strides=12, padding='valid'),
            Dropout(0.5),
            Conv1D(int(CHANNELS/2), kernel_size=12, padding="valid", activation='relu', kernel_initializer='random_uniform'),
            MaxPooling1D(pool_size=6, strides=12, padding='valid'),
            Dropout(0.5),
            TimeDistributed(Dense(128, activation='relu')),
            Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
            Dropout(0.5),
            Flatten(),
            Dense(width_dense, activation='relu'),
            Dropout(0.5),]
        
    elif CNN_NUM==2 and DENSE_NUM==2:
        layer0 = [
        Conv1D(CHANNELS, kernel_size=24, padding="valid", activation='relu', kernel_initializer='random_uniform'),
        MaxPooling1D(pool_size=12, strides=12, padding='valid'),
        Dropout(0.5),
        Conv1D(int(CHANNELS/2), kernel_size=12, padding="valid", activation='relu', kernel_initializer='random_uniform'),
        MaxPooling1D(pool_size=12, strides=12, padding='valid'),
        Dropout(0.5),
        TimeDistributed(Dense(128, activation='relu')),
        # Initialize variables (if any)
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
        Dropout(0.5),
        Flatten(),
        Dense(width_dense, activation='relu'),
        Dropout(0.5),
        Dense(int(width_dense/2), activation='relu'),
        Dropout(0.5),]
    
    elif CNN_NUM==1 and DENSE_NUM==2:
        layer0 = [
            Conv1D(CHANNELS, kernel_size=24, padding="valid", activation='relu', kernel_initializer='random_uniform'),
            MaxPooling1D(pool_size=12, strides=12, padding='valid'),
            Dropout(0.5),
            TimeDistributed(Dense(128, activation='relu')),
            Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
            Dropout(0.5),
            Flatten(),
            Dense(width_dense, activation='relu'),
            Dropout(0.5),]
    
    if CLASSIFICATION is True :
        # Classification
        layer1 = [
            Dense(NUM_CLASSES, activation='sigmoid')]
    
    elif MULTI_REGRESSION is True:
        # Multi head regression
        layer1 = [
            Dense(NUM_CLASSES)]
    
    else:
        # Simple regression
        layer1 = [Dense(1)]

    forward_output_f = get_output(forward_input, layer0)
    reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), layer0)
    merged_output = concatenate([forward_output_f,reverse_output_r],axis=1)
    output = get_output(merged_output, layer1)
    model = Model(input=forward_input, output=output)
    
    w=24
    f = open(PWM_PATH, "rb")
    motif_dict = pickle.load(f)
    f.close()
    conv_weights = model.layers[3].get_weights()
    for i, name in enumerate(motif_dict):
        if i == conv_weights[0].shape[2]:
            break
        conv_weights[0][int((w-len(motif_dict[name]))/2):int((w-len(motif_dict[name]))/2) + len(motif_dict[name]), :, i] = motif_dict[name]
    model.layers[3].set_weights(conv_weights)
    
    model.summary()
    
    if CLASSIFICATION is True :
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        metric = "val_accuracy"
        
    if REGRESSION is True :
        model.compile(optimizer='adam', loss=[rmse], metrics=[pearson_r])
        metric = "val_pearson_r"
    
    if MULTI_REGRESSION is True:
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        metric="val_mae"
        
    return model, metric


def build_nadav_hybrid_model(CHANNELS, width_dense, REGRESSION, MULTI_REGRESSION, CNN_NUM, DENSE_NUM, CLASSIFICATION, seq_shape, PWM_PATH):
    reverse_lambda_ax2 = Lambda(lambda x: K.reverse(x,axes=2))
    reverse_lambda_ax1 = Lambda(lambda x: K.reverse(x,axes=1))
    
    forward_input = Input(shape=seq_shape)
    reverse_input = Input(shape=seq_shape)
    
    layer0 = [
    Conv1D(250, kernel_size=7, padding="valid", activation='relu', kernel_initializer='random_uniform'),
    BatchNormalization(),
    Conv1D(250, kernel_size=8, padding="valid", activation='relu', kernel_initializer='random_uniform'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, strides=None, padding='valid'),
    Dropout(0.7),
    Conv1D(250, 3, strides=1, activation='softmax'),
    BatchNormalization(),
    Conv1D(100, 2, strides=1, activation='softmax'),
    BatchNormalization(),
    MaxPooling1D(pool_size=1, strides=None),
    Dropout(0.7),
    TimeDistributed(Dense(128, activation='relu')),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
    Dropout(0.7),
    Flatten(),
    Dense(300, activation='sigmoid'),
    Dropout(0.7),
    Dense(200, activation='sigmoid'),
   ]
    
    if CLASSIFICATION is True :
        # Classification
        layer1 = [
            Dense(NUM_CLASSES, activation='sigmoid')]
    
    elif MULTI_REGRESSION is True:
        # Multi head regression
        layer1 = [
            Dense(NUM_CLASSES)]
    
    else:
        # Simple regression
        layer1 = [Dense(1)]

    forward_output_f = get_output(forward_input, layer0)
    reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), layer0)
    merged_output = concatenate([forward_output_f,reverse_output_r],axis=1)
    output = get_output(merged_output, layer1)
    model = Model(input=forward_input, output=output)
    '''
    w=24
    f = open(PWM_PATH, "rb")
    motif_dict = pickle.load(f)
    f.close()
    conv_weights = model.layers[3].get_weights()
    for i, name in enumerate(motif_dict):
        if i == conv_weights[0].shape[2]:
            break
        conv_weights[0][int((w-len(motif_dict[name]))/2):int((w-len(motif_dict[name]))/2) + len(motif_dict[name]), :, i] = motif_dict[name]
    model.layers[3].set_weights(conv_weights)
    '''
    model.summary()
    
    
    if CLASSIFICATION is True :
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        metric = "val_accuracy"
        
    if REGRESSION is True :
        model.compile(optimizer="adam", loss=[rmse], metrics=[pearson_r])
        metric = "val_pearson_r"
    
    if MULTI_REGRESSION is True:
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        metric="val_mae"
        
    return model, metric

def build_nadav_model(CHANNELS, width_dense, REGRESSION, MULTI_REGRESSION, CNN_NUM, DENSE_NUM, CLASSIFICATION, seq_shape, PWM_PATH):
    reverse_lambda_ax2 = Lambda(lambda x: K.reverse(x,axes=2))
    reverse_lambda_ax1 = Lambda(lambda x: K.reverse(x,axes=1))
    
    forward_input = Input(shape=seq_shape)
    reverse_input = Input(shape=seq_shape)
    
    layer0 = [
    Conv1D(250, kernel_size=7, strides=1, activation='relu', name="conv1", kernel_initializer='random_uniform'),
    BatchNormalization(),
    Conv1D(250, 8, strides=1, activation='softmax', name="conv2", kernel_initializer='random_uniform'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2, strides=None, name="maxpool1"),
    Dropout(0.5),
    Conv1D(250, 3, strides=1, activation='softmax', name="conv3"),
    BatchNormalization(),
     Conv1D(100, 2, strides=1, activation='softmax', name="conv4"),
    BatchNormalization(),
    MaxPooling1D(pool_size=1, strides=None, name="maxpool2"),
    Dropout(0.5),
    Flatten(),
    Dense(300, activation='sigmoid'),
    Dropout(0.5),
    Dense(200, activation='sigmoid'),
   ]
    
    if CLASSIFICATION is True :
        # Classification
        layer1 = [
            Dense(NUM_CLASSES, activation='sigmoid')]
    
    elif MULTI_REGRESSION is True:
        # Multi head regression
        layer1 = [
            Dense(NUM_CLASSES)]
    
    else:
        # Simple regression
        layer1 = [Dense(1)]

    forward_output_f = get_output(forward_input, layer0)
    reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), layer0)
    merged_output = concatenate([forward_output_f,reverse_output_r],axis=1)
    output = get_output(merged_output, layer1)
    model = Model(input=forward_input, output=output)
    '''
    w=24
    f = open(PWM_PATH, "rb")
    motif_dict = pickle.load(f)
    f.close()
    conv_weights = model.layers[3].get_weights()
    for i, name in enumerate(motif_dict):
        if i == conv_weights[0].shape[2]:
            break
        conv_weights[0][int((w-len(motif_dict[name]))/2):int((w-len(motif_dict[name]))/2) + len(motif_dict[name]), :, i] = motif_dict[name]
    model.layers[3].set_weights(conv_weights)
    '''
    model.summary()
    
    if CLASSIFICATION is True :
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        metric = "val_accuracy"
        
    if REGRESSION is True :
        model.compile(optimizer='adam', loss=rmse, metrics=[pearson_r])
        metric = "val_pearson_r"
    
    if MULTI_REGRESSION is True:
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        metric="val_mae"
        
    return model, metric
        
def readfile_fasta(filename, NUM_CLASSES):
    ids = []
    ids_d = {}
    seqs = {}
    classes = {}
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    seq = []
    for line in lines:
        if line[0] == '>':
            # Append the whole id line to list of ids (including, seq_id, class and any other extra annotation)
            ids.append(line[1:].rstrip('\n'))
            
            if line[1:].rstrip('\n').split('_')[0] not in seqs:
                # If sequence id not encountered add to seqs dictionary
                seqs[line[1:].rstrip('\n').split('_')[0]] = []
                
            if line[1:].rstrip('\n').split('_')[0] not in ids_d:
                # If sequence id not encountered add to ids_d dictionary
                ids_d[line[1:].rstrip('\n').split('_')[0]] = line[1:].rstrip('\n').split('_')[0]
                
            if line[1:].rstrip('\n').split('_')[0] not in classes:
                # Add the class of the id sequence
                classes[line[1:].rstrip('\n').split('_')[0]] = np.zeros(NUM_CLASSES)
            classes[line[1:].rstrip('\n').split('_')[0]][int(line[1:].rstrip('\n').split('_')[1])-1] = 1        
            
            if seq != []: 
                # If sequence is not empty
                seqs[ids[-2].split('_')[0]]= ("".join(seq))
            seq = []
        
        else:
            # Append sequence of current loop
            seq.append(line.rstrip('\n').upper())
            
    if seq != []:
        seqs[ids[-1].split('_')[0]]=("".join(seq))
    return ids,ids_d,seqs,classes

def readfile_multihead_regression(filename):
    """Reads a csv file and outputs ids,ids_d,seqs, vector of expressions"""
    df = pd.read_csv(filename)
    ids = df.CRS.values.tolist()
    ids_d = dict(zip(df.CRS, df.CRS))
    seqs = dict(zip(df.CRS, df.seq))
    array_expression = dict(zip(df.CRS, df[["DivEry", "EoBaso", "MegEry", "Monocytes", "Neutrophil"]].values))
    return ids, ids_d, seqs, array_expression

def readfile_regression(filename):
    """Reads a csv file and outputs ids,ids_d,seqs, vector of expressions"""
    df = pd.read_csv(filename, sep="\t")
    ids = df.name.values.tolist()
    ids_d = dict(zip(df.name, df.name))
    seqs = dict(zip(df.name, df.sequence))
    values = dict(zip(df.name, df.meanVal.values))
    return ids, ids_d, seqs, values

def one_hot_encode_along_row_axis(sequence):
    to_return = np.zeros((1,len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return[0],
                                 sequence=sequence, one_hot_axis=1)
    return to_return

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1): 
        assert zeros_array.shape[0] == len(sequence)
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1

def create_plots(history, metric, flag, foldername):
    
    if flag == "regression":
        plt.plot(history.history["pearson_r"])
        plt.plot(history.history["val_pearson_r"])
    else :
        plt.plot(history.history[metric[4:]])
        plt.plot(history.history[metric])
    plt.title('model metric')
    plt.ylabel(metric[4:])
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(foldername + 'metric.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(foldername + 'loss.png')
    plt.clf()
    
def json_hdf5_to_model(json_filename, hdf5_filename):  
    with open(json_filename, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(hdf5_filename)
    return model

def shuffle_label(label):
    for i in range(len(label.T)):
        label.T[i] = shuffle(label.T[i])
    return label

def calculate_roc_pr(score, label):
    output = np.zeros((len(label.T), 2))
    for i in range(len(label.T)):
        roc_ = roc_auc_score(label.T[i], score.T[i])
        pr_ = average_precision_score(label.T[i], score.T[i])
        output[i] = [roc_, pr_]
    return output

def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))

def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))

def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))

def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}

def plot_weights_given_ax(ax, array,
                 height_padding_factor,
                 length_padding,
                 subticks_frequency,
                 highlight,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
            
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
    return ax

def plot_weights_modified(array, fig, n,n1,n2, title='', ylab='',
                              figsize=(20,2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=20,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):
    ax = fig.add_subplot(n,n1,n2) 
    ax.set_title(title)
    ax.set_ylabel(ylab)
    y = plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)
    return fig,ax


def path_to_id_dict(path):
    filename = path
    ids, ids_d, seqs, classes = readfile_wolabel(filename)
    result = {'ids':ids, 'ids_d':ids_d, 'seqs':seqs, 'classes':classes, }
    return result

def path_to_X_id_dict(path):
    filename = path
    ids, ids_d, seqs = readfile_wolabel(filename)
    X = np.array([one_hot_encode_along_row_axis(seqs[id]) for id in ids_d]).squeeze(axis=1)
    X_rc = [X,  X[:,::-1,::-1]]
    result = {'ids':ids, 'ids_d':ids_d, 'seqs':seqs, 'X':X, 'X_rc':X_rc }
    return result

def path_to_X_id_dict_label(path, selected_classes, flag):
    train_filename = path   
    if flag=="classification":
        train_ids,train_ids_d, train_seqs, train_classes = readfile_fasta(train_filename)
    
    if flag=="regression":
            train_ids,train_ids_d, train_seqs, train_classes = readfile_regression(train_filename)

    else:
        train_ids,train_ids_d, train_seqs, train_classes = read_multi_regression(train_filename)
        
    X = np.array([one_hot_encode_along_row_axis(train_seqs[id]) for id in train_ids_d]).squeeze(axis=1)
    y = np.array([train_classes[id] for id in train_ids_d])

    y = y[:,selected_classes]
    X = X[y.sum(axis=1)>0]
    ids = np.array([id for id in train_ids_d])
    ids = ids[y.sum(axis=1)>0]
    y = y[y.sum(axis=1)>0]
    X_rc = [X,  X[:,::-1,::-1]]
    y_single=y[y.sum(axis=1)==1]
    
    result = {'ids':train_ids, 'ids_d':train_ids_d, 'seqs':train_seqs, 'classes':train_classes, 'X':X, 'X_rc':X_rc, 'y':y ,"y_single":y_single}
    return result   

def readfile_wolabel(filename):
    ids = []
    ids_d = {}
    seqs = {}
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    seq = []
    for line in lines:
        if line[0] == '>':
            ids.append(line[1:].rstrip('\n'))
            if line[1:].rstrip('\n') not in seqs:
                seqs[line[1:].rstrip('\n')] = []
            if line[1:].rstrip('\n') not in ids_d:
                ids_d[line[1:].rstrip('\n')] = line[1:].rstrip('\n')    
            if seq != []: seqs[ids[-2]]= ("".join(seq))
            seq = []
        else:
            seq.append(line.rstrip('\n').upper())
    if seq != []:
        seqs[ids[-1]]=("".join(seq))

    return ids,ids_d,seqs

def loc_to_model_loss(foldername):
    """Parses history and selects the best model, saves it as model_best_loss and opens it"""
    file_list = os.listdir(foldername)
    highest_string = max(filter(lambda x: "model_best_loss" in x, file_list), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return json_hdf5_to_model(foldername + 'model.json', foldername + highest_string)
