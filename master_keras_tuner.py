# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:00:00 2021


@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re

"""
We use the standard convention, where intances in a dataset runs in row, and features run in columns

# implement all we know about neural network on various example problems (binary/multi-class classification, regression)

# use keras_tuner to build a model with tunable hyperparameters

https://keras.io/api/keras_tuner/
https://keras.io/guides/keras_tuner/getting_started/
https://keras.io/api/keras_tuner/hyperparameters/


save and load a model



# most important hyperparameters to tune
# most important: learning rate, 
# then momentum beta (good default is 0.9), mini-batch size, number of hidden units,
# then number of layers and learning rate decay
# (beta1, beta2, epsilon) no need to tune really

# DO NOT do a grid search
# DO try random values
# because you may not know in advance which hyperparameters are more important
# use coarse-to-fine scheme
# re-test hyperparameters once in a while

# choosing the right scale to sample hyperparameters
# number of units and number of layers: reasonable to sample uniformly
# learning rate: uniformly in log scale
# 1-beta=0.1, 0.01, ...

# bias and variance (Google for definitions)
# high bias = high training set error
# high variance = high dev set error (compared to training set error); sensitive to change of training data point
# This assumes optimal (Bayes) error of 0%. Counter-example: blurry images for cat photo classsification

# basic recipe for machine learning
# 1. high bias (training error)?
# 2. <<bigger network>>, train longer, search for another neural network architecture
# 3. repeat until bias is low
# 4. high variance (dev error) (can we generalise)?
# 5. <<more data>>, regularisation, search for another neural network architecture
# 6. repeat until variance is low
# 7. done =)

Summary: 
    # 

"""


# %% Preamble

# Make the output look better
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.initializers import glorot_uniform
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, Concatenate, BatchNormalization
from keras.models import Model
from keras_tuner import HyperModel
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import tensorflow_addons as tfa # import tfa which contains metrics for regression
from tensorflow.keras import layers
from tensorflow import keras
import os
import re
import matplotlib.pyplot as plt
import seaborn as sn
import h5py
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\9_nlp_disaster_tweets')


# %% tensorflow.keras


# %% load embedding matrix pre-trained using glove

# https://nlp.stanford.edu/projects/glove/
# Twitter (2B tweets, 27B tokens, 1,193,514 vocab, uncased, 25d, 50d, 100d, & 200d vectors)
# read in the glove file. Each line in the text file is a string containing a word followed by the embedding vector, all separated by a whitespace
# word_to_vec_map is a dict of words to their embedding vectors
# (1,193,514 words, with the valid indices starting from 1 to 1193514

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            # line = line.strip().split()
            # !!! in glove.twitter.27B, the 38522-nd vocab is '\x85' which strip(), split() in Python and re.split(r'\s') in regex interpret as unicode whitespace.
            line = re.split(r' ', line.rstrip())
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        # starts from 1
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


# index is valued  from 1 to 1193514
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(
    'glove.twitter.27B/glove.twitter.27B.100d.txt')


# %% LSTM with embedding layer


"""
2. The Embedding layer

# In Keras, the embedding matrix is represented as a "layer".
# The embedding matrix maps word indices to embedding vectors.
    # The word indices are [positive] integers.
    # The embedding vectors are dense vectors of fixed size.
    # When we say a vector is "dense", in this context, it means that most of the values are non-zero. As a counter-example, a one-hot encoded vector is not "dense."
# The embedding matrix can be derived in two ways:
    # Training a model to derive the embeddings from scratch.
    # Using a pretrained embedding

# Using and updating pre-trained embeddings
# In this part, you will learn how to create an Embedding() layer in Keras
# You will initialize the Embedding layer with the GloVe 50-dimensional vectors.
# In the code below, we'll show you how Keras allows you to either train or leave fixed this layer.
# Because our training set is quite small, we will leave the GloVe embeddings fixed instead of updating them.
"""

"""build an embedding layer"""


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    # Embedding() in Keras:
    # input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
    vocab_len = len(word_to_index) + 1
    # output_dim in Embedding()
    # dimensionality of GloVe word vectors
    emb_dim = list(word_to_vec_map.values())[0].shape[0]

    # using word_to_vec_map from pre-trained Glove, construct the embedding matrix
    # Initialize the embedding matrix as a numpy array of zeros.
    embedding_matrix = np.zeros([vocab_len, emb_dim])
    # Set each row "idx" of the embedding matrix to be the word vector representation of the idx-th word of the vocabulary
    # idx starts from 1 to len(vocab), so the first row of embedding_matrix is zero representing unknown words
    for word, idx in word_to_index.items():
        embedding_matrix[idx, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # Weights are created when the Model is first called on inputs or build() is called with an input_shape.
    # somehow embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix) does not work...
    # in our model, embedding_layer takes the input layer with input shape (max_len,)
    # here embedding_layer does not know max_len yet, so use (None,)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    return embedding_layer


embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)


# %% HyperModel

"""# define the hypermodel with HyperModel class in keras_tuner"""


class HyperModel(HyperModel):
    def __init__(self, sent_indices_input_shape, non_text_features_input_shape, embedding_layer, label_cols, output_activation, loss, metrics):
        self.sent_indices_input_shape = sent_indices_input_shape
        self.non_text_features_input_shape = non_text_features_input_shape
        self.embedding_layer = embedding_layer
        self.label_cols = label_cols
        self.output_activation = output_activation
        self.loss = loss
        self.metrics = metrics

    def build(self, hp):

        #####################################################
        """# hyperparameters for layer definition"""

        # number of <Bi-LSTM> layers
        num_BLSTM = hp.Int('num_BLSTM', min_value=2, max_value=5, step=1)
        # number of <non-text> layers
        num_non_text = hp.Int('num_non_text', min_value=1, max_value=3, step=1)
        # number of <hidden> layers
        num_dense_hidden = hp.Int(
            'num_dense_hidden', min_value=1, max_value=4, step=1)

        # # activation function for hidden layers
        # hidden_activation = hp.Choice('learning_rate', values = ['relu', 'tanh'], default = 'relu')
        # dropout
        dropout = hp.Boolean("dropout", default=False)
        if dropout == True:
            dropout_rate = hp.Float(
                'dropout_rate', min_value=0.1, max_value=0.3, step=0.1, default=0.2)
        # batch normalisation
        batch_normalize = hp.Boolean("batch_normalize", default=False)

        # regulariser for kernel W. we do not consider bias and activation regularisers
        # kernel_regularizer_which = hp.Choice('kernel_regularizer_which', values = ['None', 'l2', 'l1', 'l1_l2'], default = 'None')
        # if kernel_regularizer_which == 'None' :
        #     kernel_regularizer = None
        # elif kernel_regularizer_which == 'l2' :
        #     W_l2 = hp.Choice('W_l2', values = [0.1, 1e-2, 1e-3], default = 1e-2)
        #     kernel_regularizer = regularizers.l2(l2 = W_l2)
        # elif kernel_regularizer_which == 'l1' :
        #     W_l1 = hp.Choice('W_l1', values = [0.1, 1e-2, 1e-3], default = 1e-2)
        #     kernel_regularizer = regularizers.l1(l1 = W_l1)
        # elif kernel_regularizer_which == 'l1_l2' :
        #     W_l1 = hp.Choice('W_l1', values = [0.1, 1e-2, 1e-3], default = 1e-2)
        #     W_l2 = hp.Choice('W_l2', values = [0.1, 1e-2, 1e-3], default = 1e-2)
        #     kernel_regularizer = regularizers.l1_l2(l1 = W_l1, l2 = W_l2)

        #####################################################
        """# layer definition"""

        # Define sentence_indices as the input of the graph
        # It should be of shape input_shape and dtype 'int32' (as it contains indices).
        sentence_indices = Input(
            shape=self.sent_indices_input_shape, dtype='int32')
        non_text_features = Input(shape=self.non_text_features_input_shape)

        """embedding + Bi-LSTM for text input"""
        # Propagate sentence_indices through your embedding layer, you get back the embeddings
        X_sent = self.embedding_layer(sentence_indices)

        for l in range(num_BLSTM - 1):
            LSTM_units = hp.Int('LSTM_units' + str(l),
                                min_value=128, max_value=1024, step=128)
            X_sent = Bidirectional(
                LSTM(LSTM_units, return_sequences=True), name='BLSTM_layer' + str(l))(X_sent)

            if dropout == True:
                X_sent = Dropout(rate=dropout_rate,
                                 name='LSTM_dropout'+str(l))(X_sent)
            if batch_normalize == True:
                X_sent = BatchNormalization(
                    name='LSTM_batch_normalize'+str(l))(X_sent)

        LSTM_units_last = hp.Int(
            'LSTM_units' + str(num_BLSTM - 1), min_value=128, max_value=1024, step=128)
        X_sent = Bidirectional(LSTM(LSTM_units_last, return_sequences=False),
                               name='BLSTM_layer' + str(num_BLSTM - 1))(X_sent)
        if dropout == True:
            X_sent = Dropout(rate=dropout_rate,
                             name='LSTM_dropout' + str(num_BLSTM - 1))(X_sent)
        if batch_normalize == True:
            X_sent = BatchNormalization(
                name='LSTM_batch_normalize' + str(num_BLSTM - 1))(X_sent)

        """Dense layers for non-text features """
        non_text_units0 = hp.Int(
            'non_text_units0', min_value=32, max_value=512, step=64)
        X_non_text = Dense(non_text_units0, activation='tanh',
                           name='non_text_dense0')(non_text_features)

        if dropout == True:
            X_non_text = Dropout(rate=dropout_rate,
                                 name='non_text_dropout0')(X_non_text)
        if batch_normalize == True:
            X_non_text = BatchNormalization(
                name='non_text_batch_normalize0')(X_non_text)

        if num_non_text > 1:
            for l in np.arange(1, num_non_text):
                units = hp.Int('units' + str(l), min_value=16,
                               max_value=1024, step=16)
                X_non_text = Dense(units=units,
                                   activation='tanh',
                                   name='non_text_dense'+str(l))(X_non_text)

                if dropout == True:
                    X_non_text = Dropout(rate=dropout_rate,
                                         name='non_text_dropout'+str(l))(X_non_text)
                if batch_normalize == True:
                    X_non_text = BatchNormalization(
                        name='non_text_batch_normalize'+str(l))(X_non_text)

        """concatenate text and non-text features, and pass to FC layers"""
        X = Concatenate()([X_sent, X_non_text])
        X = Dropout(.2)(X)
        X = BatchNormalization()(X)

        FC_units0 = hp.Int('FC_units0', min_value=32, max_value=512, step=64)
        X = Dense(FC_units0, activation='tanh', name='FC_dense0')(X)

        if dropout == True:
            X = Dropout(rate=dropout_rate,
                        name='FC_dropout0')(X)
        if batch_normalize == True:
            X = BatchNormalization(name='FC_batch_normalize0')(X)

        if num_dense_hidden > 1:
            for l in np.arange(1, num_dense_hidden):
                units = hp.Int('FC_units' + str(l), min_value=32,
                               max_value=512, step=64)
                X = Dense(units, activation='tanh', name='FC_dense'+str(l))(X)

                if dropout == True:
                    X = Dropout(rate=dropout_rate,
                                name='FC_dropout'+str(l))(X)
                if batch_normalize == True:
                    X = BatchNormalization(name='FC_batch_normalize'+str(l))(X)

        X = Dense(units=self.label_cols, name='output')(X)
        # Add a sigmoid activation
        X = Activation(self.output_activation, name='output_activation')(X)

        # Create Model instance which converts sentence_indices into X.
        model = Model(inputs=[sentence_indices, non_text_features], outputs=X)

        #####################################################
        """# learning rate decay schedule """

        # we only consider constant, exponential decay and power law decay
        learning_rate_decay = hp.Choice('learning_rate_decay', values=[
                                        'None', 'ExponentialDecay', 'PolynomialDecay'], default='None')
        learning_rate_initial = hp.Choice(
            'learning_rate_initial ', values=[1e-4], default=1e-4)
        if learning_rate_decay == 'None':
            learning_rate_schedule = learning_rate_initial
        elif learning_rate_decay == 'ExponentialDecay':
            learning_rate_decay_rate = hp.Fixed(
                'learning_rate_decay_rate', value=0.96)
            decay_steps = hp.Int('decay_steps', min_value=50,
                                 max_value=1000, step=100)
            learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate_initial,
                decay_steps=decay_steps,
                decay_rate=learning_rate_decay_rate)
        elif learning_rate_decay == 'PolynomialDecay':
            end_learning_rate = hp.Fixed('end_learning_rate', value=1e-6)
            decay_steps = hp.Int('decay_steps', min_value=50,
                                 max_value=1000, step=100)
            decay_power = hp.Float(
                'decay_power', min_value=0.5, max_value=2.5, step=0.5)
            learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate_initial,
                decay_steps=decay_steps,
                end_learning_rate=end_learning_rate,
                power=decay_power)

        #####################################################
        """# model optimiser"""

        # we only consider SGD, RMSprop and Adam with default betas
        optimizer_which = hp.Choice(
            'optimizer_which', values=['RMSprop', 'Adam'])
        if optimizer_which == 'SGD':
            momentum = hp.Choice('momentum', values=[0., 0.9])
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate_schedule)
        elif optimizer_which == 'RMSprop':
            rho = hp.Fixed('rho', value=0.9)
            momentum = hp.Choice('momentum', values=[0., 0.9])
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate_schedule, rho=rho, momentum=momentum)
        elif optimizer_which == 'Adam':
            beta1 = hp.Fixed('beta1', value=0.9)
            beta2 = hp.Fixed('beta2', value=0.999)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_schedule, beta_1=beta1, beta_2=beta2)

        #####################################################
        """# model optimiser definition and compilation """

        model.compile(optimizer=optimizer,
                      loss=self.loss,
                      metrics=self.metrics)

        # model.summary()

        return model

#     elif my_learning_rate_decay == 'PiecewiseConstantDecay' :
#         my_learning_rate_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
#         boundaries = my_piecewise_decay_boundaries,
#         values = my_piecewise_decay_values)

#     elif my_learning_rate_decay == 'InverseTimeDecay' :
#         my_learning_rate_schedule = keras.optimizers.schedules.InverseTimeDecay(
#         initial_learning_rate = my_learning_rate_initial,
#         decay_steps = my_decay_steps,
#         decay_rate = my_learning_rate_decay_rate)

# %% early stopping


def my_callbacks(early_stopping=False, early_stopping_monitor='val_loss', min_delta=0.0001, patience=100, restore_best_weights=True, mode='min'):
    if early_stopping == False:
        return None
    else:
        early_stopping_scheme = [EarlyStopping(
            monitor=early_stopping_monitor,
            mode=mode,
            min_delta=min_delta,  # minimium amount of change to count as an improvement
            patience=patience,  # how many epochs to wait before stopping
            restore_best_weights=restore_best_weights,
        )]
        # print('early stopping done')
        return early_stopping_scheme


# %% load engineered datasets

# !!!
X_train_indices = np.load('engineered_datasets/X_train_indices.npy')
X_valid_indices = np.load('engineered_datasets/X_valid_indices.npy')
X_test_indices = np.load('engineered_datasets/X_test_indices.npy')

X_train_no_text_encoded2 = np.load(
    'engineered_datasets/X_train_no_text_encoded2.npy')
X_valid_no_text_encoded2 = np.load(
    'engineered_datasets/X_valid_no_text_encoded2.npy')
X_test_no_text_encoded2 = np.load(
    'engineered_datasets/X_test_no_text_encoded2.npy')

scaler = MinMaxScaler()

X_train_no_text_encoded2 = scaler.fit_transform(X_train_no_text_encoded2)
X_valid_no_text_encoded2 = scaler.transform(X_valid_no_text_encoded2)
X_test_no_text_encoded2 = scaler.transform(X_test_no_text_encoded2)

y_train = np.load('engineered_datasets/y_train.npy')
y_valid = np.load('engineered_datasets/y_valid.npy')
y_train1 = np.expand_dims(y_train, -1)
y_valid1 = np.expand_dims(y_valid, -1)


# X_train_indices.shape
# Out[3]: (6090, 128)

max_len = X_train_indices.shape[-1]
# 128

# X_train_no_text_encoded2.shape
num_features = X_train_no_text_encoded2.shape[-1]


###############################################
"""# dataset input and output metadata"""

sent_indices_input_shape = (max_len,)
non_text_features_input_shape = (num_features,)


label_cols = y_train1.shape[1]
# label_cols = y_train.shape[1]
classification_problem = True
# 1 for binary classification or regression, C (int, > 1) for multi-class C classification <<after one-hot encoding>>

# activation function of output layer: sigmoid for binary classification, softmax for multi-class classification, linear for regression
if label_cols == 1:
    if classification_problem == True:  # binary classification
        output_activation = 'sigmoid'
    else:  # regression
        output_activation = None
elif (type(label_cols) == int) & (label_cols > 1):  # multi-class classification
    output_activation = 'softmax'

###############################################
"""# hypermodel tuning: finding the best set of hyperparameters"""

###############################################
"""# define the hypermodel """

# specify the loss and metrics to train the model with
my_loss = "binary_crossentropy"
my_metrics = ['accuracy', tfa.metrics.F1Score(num_classes=1, threshold=0.5)]

hypermodel = HyperModel(sent_indices_input_shape=sent_indices_input_shape, non_text_features_input_shape=non_text_features_input_shape,
                        embedding_layer=embedding_layer, label_cols=label_cols, output_activation=output_activation, loss=my_loss, metrics=my_metrics)

###############################################
"""# if desired, choose a subset of hyperparameters to search or fix"""

# hp = kt.HyperParameters()

# to override e.g. the `learning_rate` parameter with our own selection of choices
# hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

# to fix a hyperparameter
# hp.Fixed("learning_rate", value=1e-4)

###############################################
"""# call the tuner"""

tuner = kt.tuners.BayesianOptimization(
    hypermodel,
    objective=kt.Objective('val_f1_score', direction="max"),
    # hyperparameters = hp, # overriding existing hyperparameters
    # `tune_new_entries: False to tune <only> the hyperparameters specified above, True to tune <all other> hyperparameters <not> fixed above
    # tune_new_entries = False,
    max_trials=35,  # Set to 5 to run quicker, but need 100+ for good results
    # loss="mse", # overriding existing loss
    overwrite=True)

tuner.search_space_summary()

###############################################
"""# specify mini-batch size and epoch, and begin search"""

#!!! ideally we want to do this search with cross validation
# mini-batch size
# batch_size= 512
num_epochs = 20

tuner.search([X_train_indices, X_train_no_text_encoded2], y_train1,
             validation_data=([X_valid_indices, X_valid_no_text_encoded2],
                              y_valid1),
             # batch_size = batch_size,
             epochs=num_epochs
             )

tuner.results_summary()

# !!!
###############################################
"""# pick the best set of hyperparameters"""

best_model = tuner.get_best_models()[0]

best_model.summary()
best_model.get_config()
best_model.optimizer.get_config()
best_model.get_weights()

###############################################
"""# <re-train> the model with the best set of hyperparameters"""

# best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
# # build a hypermodel
# build_my_model = HyperModel(sent_indices_input_shape=sent_indices_input_shape, non_text_features_input_shape = non_text_features_input_shape, embedding_layer = embedding_layer, label_cols=label_cols, output_activation=output_activation, loss=my_loss, metrics=my_metrics)
# # build the model with the best hyperparameters
# my_model = build_my_model.build(best_hyperparameters)

"""
The best model suffers from overfitting. Add some regularisations to the model
"""


def tweet_classification_model(sent_indices_input_shape, non_text_features_input_shape, word_to_vec_map, word_to_index, num_classes=1):
    """

    Arguments:
    sent_indices_input_shape -- shape of the padded text input, usually (max_len,)
    non_text_features_input_shape -- shape of the non-text features input (num_features, )
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,000 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=sent_indices_input_shape, dtype='int32')
    non_text_features = Input(shape=non_text_features_input_shape)

    """embedding + Bi-LSTM for text input"""
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(
        word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    # You must set return_sequences = True when stacking LSTM layers (except the last LSTM layer), so that the next LSTM layer has as input a sequence of length = num_timesteps
    # LSTM layer 0
    X_sent = Bidirectional(LSTM(512, return_sequences=True))(embeddings)
    # this is the last LSTM layer; it should only output the final state for the next (non-LSTM) layer
    X_sent = Dropout(.2)(X_sent)
    X_sent = BatchNormalization()(X_sent)
    # LSTM layer 1
    X_sent = Bidirectional(LSTM(1024, return_sequences=True))(X_sent)
    X_sent = Dropout(.2)(X_sent)
    X_sent = BatchNormalization()(X_sent)
    # LSTM layer 2
    X_sent = Bidirectional(LSTM(256, return_sequences=True))(X_sent)
    X_sent = Dropout(.2)(X_sent)
    X_sent = BatchNormalization()(X_sent)
    # LSTM layer 3
    X_sent = Bidirectional(LSTM(256, return_sequences=False))(X_sent)
    X_sent = Dropout(.2)(X_sent)
    X_sent = BatchNormalization()(X_sent)

    """Dense layers for non-text features; add l2 regularisation to reduce overfitting """
    # non-text Dense 0
    X_non_text = Dense(512, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(
        l2=0.01))(non_text_features)
    X_non_text = Dropout(.2)(X_non_text)
    X_non_text = BatchNormalization()(X_non_text)
    # non-text Dense 1
    X_non_text = Dense(16, activation='tanh',
                       kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))(X_non_text)
    X_non_text = Dropout(.2)(X_non_text)
    X_non_text = BatchNormalization()(X_non_text)

    """concatenate text and non-text features, and pass to FC layers"""
    X = Concatenate()([X_sent, X_non_text])
    X = Dropout(.2)(X)
    X = BatchNormalization()(X)
    # FC 0; add l2 regularisation to reduce overfitting
    X = Dense(512, activation='tanh',
              kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))(X)
    X = Dropout(.2)(X)
    X = BatchNormalization()(X)
    # FC 1
    X = Dense(32, activation='tanh',
              kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))(X)
    X = Dropout(.2)(X)
    X = BatchNormalization()(X)
    # FC 2
    X = Dense(32, activation='tanh',
              kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))(X)
    X = Dropout(.2)(X)
    X = BatchNormalization()(X)
    X = Dense(num_classes)(X)
    if num_classes == 1:
        # Add a sigmoid activation
        X = Activation('sigmoid')(X)
    else:
        # Add a softmax activation
        X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=[sentence_indices, non_text_features], outputs=X)

    return model


my_model = tweet_classification_model(
    (max_len,), (num_features,), word_to_vec_map, word_to_index)


lr_decay = PolynomialDecay(initial_learning_rate=0.0001,
                           decay_steps=20, end_learning_rate=0.00001, power=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)
my_model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, threshold=0.5)]
)

# my_model.summary()
# my_model.get_config()
# my_model.optimizer.get_config()

"""# re-train the model with train-validation sets"""

# mini-batch size
# batch_size= 512
num_epochs = 1000

# early stopping
early_stopping = True
early_stopping_monitor = "val_f1_score"
min_delta = 0.00001
patience = 50
restore_best_weights = True
mode = 'max'


history = my_model.fit([X_train_indices, X_train_no_text_encoded2], y_train1,
                       validation_data=([X_valid_indices, X_valid_no_text_encoded2],
                                        y_valid1),
                       epochs=num_epochs,
                       callbacks=my_callbacks(early_stopping,
                                              early_stopping_monitor,
                                              min_delta, patience,
                                              restore_best_weights,
                                              mode=mode),
                       verbose='auto')


# Epoch 59/1000
# 191/191 [==============================] - 43s 225ms/step - loss: 2.7831 - accuracy: 0.9571 - f1_score: 0.9504 - val_loss: 3.5519 - val_accuracy: 0.7886 - val_f1_score: 0.7399
# still overfitting with the use of regularisations

history_df = pd.DataFrame(history.history)
history_df.columns

fig = plt.figure(dpi=150)
plt.plot(history.history['f1_score'], color='red', label='f1_score')
plt.plot(history.history['val_f1_score'], color='blue', label='val_f1_score')
ax = plt.gca()
ax.set_xlabel('Epochs')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
# plt.savefig('LSTM_2_f1.png', dpi = 150)

fig = plt.figure(dpi=150)
plt.plot(history.history['loss'], color='red', label='loss')
plt.plot(history.history['val_loss'], color='blue', label='val_loss')
ax = plt.gca()
ax.set_xlabel('Epochs')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
# plt.savefig('LSTM_2_loss', dpi = 150)

fig = plt.figure(dpi=150)
plt.plot(history.history['accuracy'], color='red', label='accuracy')
plt.plot(history.history['val_accuracy'], color='blue', label='val_accuracy')
ax = plt.gca()
ax.set_xlabel('Epochs')
ax.set_ylabel(None)
# ax.set_yticks([])
ax.set_title(None)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
# plt.savefig('LSTM_2_accuracy', dpi = 150)


"""# save the model"""

# default directory is found by os.getcwd()
my_model.save('my_LSTM_model.h5')

# my_loaded_model = keras.models.load_model('my_model')


"""# finally, evaluate the model performance with the holdout test set """

# print(my_model.evaluate([X_train_indices, X_train_no_text_encoded2], y_train1))
# print(my_model.evaluate([X_valid_indices, X_valid_no_text_encoded2],
#                                        y_valid1))


# R2 scores
# metric = tfa.metrics.r_square.RSquare()

# metric.update_state(y_train.squeeze(), my_model.predict(encoded_X_train_scaled).squeeze())
# result = metric.result()
# print(result.numpy())
# metric.update_state(y_valid.squeeze(), my_model.predict(encoded_X_valid_scaled).squeeze())
# result = metric.result()
# print(result.numpy())


"""predict"""

prediction = my_model.predict(
    [X_test_indices, X_test_no_text_encoded2]).squeeze()
prediction_bool = (prediction > 0.5).astype(int)
test_df = pd.read_csv("test.csv", header=[0])
LSTM_pred = pd.Series(prediction_bool, index=test_df['id'], name='target')
LSTM_pred.to_csv('LSTM_pred.csv')
# 0.79987

# %%

# tuner.results_summary()
# Results summary
# Results in .\untitled_project
# Showing 10 best trials
# Objective(name='val_f1_score', direction='max')
# Trial summary
# Hyperparameters:
# num_BLSTM: 5
# num_non_text: 2
# num_dense_hidden: 3
# dropout: True
# batch_normalize: True
# LSTM_units0: 256
# LSTM_units1: 512
# non_text_units0: 480
# FC_units0: 416
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: Adam
# rho: 0.9
# momentum: 0.0
# dropout_rate: 0.2
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 128
# units1: 16
# FC_units1: 32
# FC_units2: 32
# beta1: 0.9
# beta2: 0.999
# learning_rate_decay_rate: 0.96
# decay_steps: 50
# units2: 16
# Score: 0.7783955335617065
# Trial summary
# Hyperparameters:
# num_BLSTM: 2
# num_non_text: 3
# num_dense_hidden: 4
# dropout: True
# batch_normalize: False
# LSTM_units0: 128
# LSTM_units1: 128
# non_text_units0: 480
# FC_units0: 480
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: RMSprop
# rho: 0.9
# momentum: 0.9
# dropout_rate: 0.30000000000000004
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 640
# units1: 768
# FC_units1: 32
# FC_units2: 352
# beta1: 0.9
# beta2: 0.999
# learning_rate_decay_rate: 0.96
# decay_steps: 950
# units2: 16
# end_learning_rate: 1e-06
# decay_power: 0.5
# FC_units3: 32
# Score: 0.7768730521202087
# Trial summary
# Hyperparameters:
# num_BLSTM: 5
# num_non_text: 2
# num_dense_hidden: 3
# dropout: True
# batch_normalize: True
# LSTM_units0: 256
# LSTM_units1: 384
# non_text_units0: 416
# FC_units0: 416
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: Adam
# rho: 0.9
# momentum: 0.0
# dropout_rate: 0.2
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 128
# units1: 16
# FC_units1: 32
# FC_units2: 32
# beta1: 0.9
# beta2: 0.999
# Score: 0.7755101919174194
# Trial summary
# Hyperparameters:
# num_BLSTM: 2
# num_non_text: 3
# num_dense_hidden: 4
# dropout: True
# batch_normalize: False
# LSTM_units0: 128
# LSTM_units1: 128
# non_text_units0: 480
# FC_units0: 480
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: RMSprop
# rho: 0.9
# momentum: 0.9
# dropout_rate: 0.30000000000000004
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 384
# units1: 832
# FC_units1: 32
# FC_units2: 224
# beta1: 0.9
# beta2: 0.999
# learning_rate_decay_rate: 0.96
# decay_steps: 950
# units2: 16
# end_learning_rate: 1e-06
# decay_power: 0.5
# FC_units3: 416
# Score: 0.7749195694923401
# Trial summary
# Hyperparameters:
# num_BLSTM: 4
# num_non_text: 2
# num_dense_hidden: 4
# dropout: True
# batch_normalize: False
# LSTM_units0: 128
# LSTM_units1: 128
# non_text_units0: 480
# FC_units0: 480
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: RMSprop
# rho: 0.9
# momentum: 0.9
# dropout_rate: 0.2
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 896
# units1: 656
# FC_units1: 32
# FC_units2: 480
# beta1: 0.9
# beta2: 0.999
# learning_rate_decay_rate: 0.96
# decay_steps: 950
# units2: 16
# end_learning_rate: 1e-06
# decay_power: 0.5
# FC_units3: 32
# Score: 0.7734139561653137
# Trial summary
# Hyperparameters:
# num_BLSTM: 2
# num_non_text: 3
# num_dense_hidden: 4
# dropout: True
# batch_normalize: False
# LSTM_units0: 128
# LSTM_units1: 128
# non_text_units0: 480
# FC_units0: 480
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: RMSprop
# rho: 0.9
# momentum: 0.9
# dropout_rate: 0.30000000000000004
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 1024
# units1: 48
# FC_units1: 32
# FC_units2: 416
# beta1: 0.9
# beta2: 0.999
# learning_rate_decay_rate: 0.96
# decay_steps: 950
# units2: 16
# end_learning_rate: 1e-06
# decay_power: 0.5
# FC_units3: 352
# Score: 0.7726199626922607
# Trial summary
# Hyperparameters:
# num_BLSTM: 2
# num_non_text: 3
# num_dense_hidden: 4
# dropout: True
# batch_normalize: False
# LSTM_units0: 128
# LSTM_units1: 128
# non_text_units0: 96
# FC_units0: 480
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: RMSprop
# rho: 0.9
# momentum: 0.9
# dropout_rate: 0.30000000000000004
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 896
# units1: 880
# FC_units1: 32
# FC_units2: 352
# beta1: 0.9
# beta2: 0.999
# learning_rate_decay_rate: 0.96
# decay_steps: 950
# units2: 16
# end_learning_rate: 1e-06
# decay_power: 0.5
# FC_units3: 224
# Score: 0.7717908024787903
# Trial summary
# Hyperparameters:
# num_BLSTM: 5
# num_non_text: 3
# num_dense_hidden: 4
# dropout: True
# batch_normalize: False
# LSTM_units0: 128
# LSTM_units1: 128
# non_text_units0: 480
# FC_units0: 480
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: RMSprop
# rho: 0.9
# momentum: 0.9
# dropout_rate: 0.30000000000000004
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 1024
# units1: 1024
# FC_units1: 32
# FC_units2: 480
# beta1: 0.9
# beta2: 0.999
# learning_rate_decay_rate: 0.96
# decay_steps: 950
# units2: 16
# end_learning_rate: 1e-06
# decay_power: 0.5
# FC_units3: 352
# Score: 0.7705977559089661
# Trial summary
# Hyperparameters:
# num_BLSTM: 5
# num_non_text: 2
# num_dense_hidden: 3
# dropout: True
# batch_normalize: True
# LSTM_units0: 128
# LSTM_units1: 512
# non_text_units0: 480
# FC_units0: 352
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: Adam
# rho: 0.9
# momentum: 0.0
# dropout_rate: 0.2
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 128
# units1: 16
# FC_units1: 32
# FC_units2: 32
# beta1: 0.9
# beta2: 0.999
# learning_rate_decay_rate: 0.96
# decay_steps: 50
# units2: 96
# Score: 0.7700701355934143
# Trial summary
# Hyperparameters:
# num_BLSTM: 2
# num_non_text: 3
# num_dense_hidden: 4
# dropout: True
# batch_normalize: True
# LSTM_units0: 128
# LSTM_units1: 128
# non_text_units0: 480
# FC_units0: 480
# learning_rate_decay: None
# learning_rate_initial : 0.0001
# optimizer_which: RMSprop
# rho: 0.9
# momentum: 0.9
# dropout_rate: 0.30000000000000004
# LSTM_units2: 128
# LSTM_units3: 128
# LSTM_units4: 1024
# units1: 560
# FC_units1: 32
# FC_units2: 480
# beta1: 0.9
# beta2: 0.999
# learning_rate_decay_rate: 0.96
# decay_steps: 950
# units2: 16
# end_learning_rate: 1e-06
# decay_power: 0.5
# FC_units3: 32
# Score: 0.7683429718017578


# best_model.summary()
# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            [(None, 128)]        0
# __________________________________________________________________________________________________
# embedding (Embedding)           (None, 128, 100)     119351500   input_1[0][0]
# __________________________________________________________________________________________________
# BLSTM_layer0 (Bidirectional)    (None, 128, 512)     731136      embedding[68][0]
# __________________________________________________________________________________________________
# LSTM_dropout0 (Dropout)         (None, 128, 512)     0           BLSTM_layer0[0][0]
# __________________________________________________________________________________________________
# LSTM_batch_normalize0 (BatchNor (None, 128, 512)     2048        LSTM_dropout0[0][0]
# __________________________________________________________________________________________________
# BLSTM_layer1 (Bidirectional)    (None, 128, 1024)    4198400     LSTM_batch_normalize0[0][0]
# __________________________________________________________________________________________________
# LSTM_dropout1 (Dropout)         (None, 128, 1024)    0           BLSTM_layer1[0][0]
# __________________________________________________________________________________________________
# LSTM_batch_normalize1 (BatchNor (None, 128, 1024)    4096        LSTM_dropout1[0][0]
# __________________________________________________________________________________________________
# BLSTM_layer2 (Bidirectional)    (None, 128, 256)     1180672     LSTM_batch_normalize1[0][0]
# __________________________________________________________________________________________________
# LSTM_dropout2 (Dropout)         (None, 128, 256)     0           BLSTM_layer2[0][0]
# __________________________________________________________________________________________________
# LSTM_batch_normalize2 (BatchNor (None, 128, 256)     1024        LSTM_dropout2[0][0]
# __________________________________________________________________________________________________
# input_2 (InputLayer)            [(None, 11)]         0
# __________________________________________________________________________________________________
# BLSTM_layer3 (Bidirectional)    (None, 128, 256)     394240      LSTM_batch_normalize2[0][0]
# __________________________________________________________________________________________________
# non_text_dense0 (Dense)         (None, 480)          5760        input_2[0][0]
# __________________________________________________________________________________________________
# LSTM_dropout3 (Dropout)         (None, 128, 256)     0           BLSTM_layer3[0][0]
# __________________________________________________________________________________________________
# non_text_dropout0 (Dropout)     (None, 480)          0           non_text_dense0[0][0]
# __________________________________________________________________________________________________
# LSTM_batch_normalize3 (BatchNor (None, 128, 256)     1024        LSTM_dropout3[0][0]
# __________________________________________________________________________________________________
# non_text_batch_normalize0 (Batc (None, 480)          1920        non_text_dropout0[0][0]
# __________________________________________________________________________________________________
# BLSTM_layer4 (Bidirectional)    (None, 256)          394240      LSTM_batch_normalize3[0][0]
# __________________________________________________________________________________________________
# non_text_dense1 (Dense)         (None, 16)           7696        non_text_batch_normalize0[0][0]
# __________________________________________________________________________________________________
# LSTM_dropout4 (Dropout)         (None, 256)          0           BLSTM_layer4[0][0]
# __________________________________________________________________________________________________
# non_text_dropout1 (Dropout)     (None, 16)           0           non_text_dense1[0][0]
# __________________________________________________________________________________________________
# LSTM_batch_normalize4 (BatchNor (None, 256)          1024        LSTM_dropout4[0][0]
# __________________________________________________________________________________________________
# non_text_batch_normalize1 (Batc (None, 16)           64          non_text_dropout1[0][0]
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 272)          0           LSTM_batch_normalize4[0][0]
#                                                                  non_text_batch_normalize1[0][0]
# __________________________________________________________________________________________________
# dropout (Dropout)               (None, 272)          0           concatenate[0][0]
# __________________________________________________________________________________________________
# batch_normalization (BatchNorma (None, 272)          1088        dropout[0][0]
# __________________________________________________________________________________________________
# FC_dense0 (Dense)               (None, 416)          113568      batch_normalization[0][0]
# __________________________________________________________________________________________________
# FC_dropout0 (Dropout)           (None, 416)          0           FC_dense0[0][0]
# __________________________________________________________________________________________________
# FC_batch_normalize0 (BatchNorma (None, 416)          1664        FC_dropout0[0][0]
# __________________________________________________________________________________________________
# FC_dense1 (Dense)               (None, 32)           13344       FC_batch_normalize0[0][0]
# __________________________________________________________________________________________________
# FC_dropout1 (Dropout)           (None, 32)           0           FC_dense1[0][0]
# __________________________________________________________________________________________________
# FC_batch_normalize1 (BatchNorma (None, 32)           128         FC_dropout1[0][0]
# __________________________________________________________________________________________________
# FC_dense2 (Dense)               (None, 32)           1056        FC_batch_normalize1[0][0]
# __________________________________________________________________________________________________
# FC_dropout2 (Dropout)           (None, 32)           0           FC_dense2[0][0]
# __________________________________________________________________________________________________
# FC_batch_normalize2 (BatchNorma (None, 32)           128         FC_dropout2[0][0]
# __________________________________________________________________________________________________
# output (Dense)                  (None, 1)            33          FC_batch_normalize2[0][0]
# __________________________________________________________________________________________________
# output_activation (Activation)  (None, 1)            0           output[0][0]
# ==================================================================================================
# Total params: 126,405,853
# Trainable params: 7,047,249
# Non-trainable params: 119,358,604
# __________________________________________________________________________________________________

# {'name': 'Adam',
#  'learning_rate': 0.0001,
#  'decay': 0.0,
#  'beta_1': 0.9,
#  'beta_2': 0.999,
#  'epsilon': 1e-07,
#  'amsgrad': False}
