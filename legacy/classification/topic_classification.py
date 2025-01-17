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
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Bidirectional, Concatenate, PReLU 
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers import Layer, average, Input, Lambda, concatenate
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K 

reverse_lambda_ax2 = Lambda(lambda x: K.reverse(x,axes=2))
reverse_lambda_ax1 = Lambda(lambda x: K.reverse(x,axes=1))

def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

def build_model():
    forward_input = Input(shape=seq_shape)
    reverse_input = Input(shape=seq_shape)

    layer0 = [
        Conv1D(1024, kernel_size=24, padding="valid", activation='relu', kernel_initializer='random_uniform'),
        MaxPooling1D(pool_size=12, strides=12, padding='valid'),
        Dropout(0.5),
        TimeDistributed(Dense(128, activation='relu')),
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
        Dropout(0.5),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),]
    layer1 = [
        Dense(81, activation='sigmoid')]
    forward_output_f = get_output(forward_input, layer0)
    reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), layer0)
    merged_output = concatenate([forward_output_f,reverse_output_r],axis=1)
    output = get_output(merged_output, layer1)
    model = Model(input=forward_input, output=output)


    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def readfile(filename):
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
            ids.append(line[1:].rstrip('\n'))
            if line[1:].rstrip('\n').split('_')[0] not in seqs:
                seqs[line[1:].rstrip('\n').split('_')[0]] = []
            if line[1:].rstrip('\n').split('_')[0] not in ids_d:
                ids_d[line[1:].rstrip('\n').split('_')[0]] = line[1:].rstrip('\n').split('_')[0]
            if line[1:].rstrip('\n').split('_')[0] not in classes:
                classes[line[1:].rstrip('\n').split('_')[0]] = np.zeros(NUM_CLASSES)
            classes[line[1:].rstrip('\n').split('_')[0]][int(line[1:].rstrip('\n').split('_')[1])-1] = 1        
            if seq != []: seqs[ids[-2].split('_')[0]]= ("".join(seq))
            seq = []
        else:
            seq.append(line.rstrip('\n').upper())
    if seq != []:
        seqs[ids[-1].split('_')[0]]=("".join(seq))
    return ids,ids_d,seqs,classes

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

def create_plots(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(foldername + 'accuracy.png')
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

def loc_to_model_loss(loc):
    return json_hdf5_to_model(loc + 'model.json', loc + 'model_best_loss.hdf5')

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

NUM_CLASSES = 81
selected_classes = np.array(list(range(NUM_CLASSES)))
SEQ_LEN = 500
SEQ_DIM = 4
seq_shape = (SEQ_LEN, SEQ_DIM)
EPOCH = 50
BATCH = 
PATIENCE = 10

foldername = '/home/felix/cluster/fpacheco/CNN_RNN/src/results_automated_script/'
train_filename = '/home/felix/cluster/fpacheco/Data/Peerlab_10topics_encoded/fasta_splits/Peerlab_500bp_regions_train.fa'
valid_filename = '/home/felix/cluster/fpacheco/Data/Peerlab_10topics_encoded/fasta_splits/Peerlab_500bp_regions_validation.fa'
test_filename = '/home/felix/cluster/fpacheco/Data/Peerlab_10topics_encoded/fasta_splits/Peerlab_500bp_regions_test.fa'

PATH_TO_SAVE_ARC = foldername + 'model.json'
PATH_TO_SAVE_BEST_LOST_WEIGHTS = foldername + 'model_best_loss.hdf5'
PATH_TO_SAVE_BEST_ACC_WEIGHTS = foldername + 'model_best_acc.hdf5'
PATH_TO_SAVE_END_WEIGHTS = foldername + 'model_end.hdf5'

print("Prepare input...")
train_ids, train_ids_d, train_seqs, train_classes = readfile(train_filename)
X_train = np.array([one_hot_encode_along_row_axis(train_seqs[id]) for id in train_ids_d]).squeeze(axis=1)
y_train = np.array([train_classes[id] for id in train_ids_d])
y_train = y_train[:,selected_classes]
X_train = X_train[y_train.sum(axis=1)>0]
y_train = y_train[y_train.sum(axis=1)>0]
train_data = X_train

valid_ids, valid_ids_d, valid_seqs, valid_classes = readfile(valid_filename)
X_valid = np.array([one_hot_encode_along_row_axis(valid_seqs[id]) for id in valid_ids_d]).squeeze(axis=1)
y_valid = np.array([valid_classes[id] for id in valid_ids_d])
y_valid = y_valid[:,selected_classes]
X_valid = X_valid[y_valid.sum(axis=1)>0]
y_valid = y_valid[y_valid.sum(axis=1)>0]
valid_data = X_valid

test_ids, test_ids_d, test_seqs, test_classes = readfile(test_filename)
X_test = np.array([one_hot_encode_along_row_axis(test_seqs[id]) for id in test_ids_d]).squeeze(axis=1)
y_test = np.array([test_classes[id] for id in test_ids_d])
y_test = y_test[:,selected_classes]
X_test = X_test[y_test.sum(axis=1)>0]
y_test = y_test[y_test.sum(axis=1)>0]
test_data = X_test

print("Compile model...")
model = build_model()

model_json = model.to_json()
with open(PATH_TO_SAVE_ARC, "w") as json_file:
    json_file.write(model_json)
print("Model architecture saved to..", PATH_TO_SAVE_ARC)

checkpoint1 = ModelCheckpoint(PATH_TO_SAVE_BEST_LOST_WEIGHTS+"_{epoch:02d}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint2 = ModelCheckpoint(PATH_TO_SAVE_BEST_ACC_WEIGHTS+"_{epoch:02d}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint3 = EarlyStopping(monitor='val_loss', patience=PATIENCE)
callbacks_list = [checkpoint1, checkpoint2, checkpoint3]

print("Train model...")
history = model.fit( train_data, y_train, nb_epoch=EPOCH, batch_size=BATCH, shuffle=True, validation_data=(valid_data, y_valid), verbose=1, callbacks=callbacks_list)
create_plots(history)
model.save_weights(PATH_TO_SAVE_END_WEIGHTS)
print("Model weights saved to..", PATH_TO_SAVE_END_WEIGHTS)
plot_model(model, to_file=foldername + 'model.png')

score, acc = model.evaluate(test_data, y_test, batch_size=BATCH)
print('Test score:', score)
print('Test accuracy:', acc)