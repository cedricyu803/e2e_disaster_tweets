# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:00:00 2021

@author: Cedric Yu
"""

# import matplotlib.pyplot as plt
# import math
# import re


"""
#####################################

# https://www.kaggle.com/c/nlp-getting-started/overview

Competition Description
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. 

#####################################

# Metric
Submissions are evaluated using [F1 between the predicted and expected answers].

Submission Instructions
For each ID in the test set, you must predict 1 if the tweet is describing a real disaster, and 0 otherwise. The file should contain a header and have the following format:

id,target
0,0
2,0
3,1
9,0
11,0

#####################################

# Dataset

Files

# train.csv - the training set
# test.csv - the test set
# sample_submission.csv - a sample submission file in the correct format

Columns

# id - a unique identifier for each tweet
# text - the text of the tweet
# location - the location the tweet was sent from (may be blank)
# keyword - a particular keyword from the tweet (may be blank)
# target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

"""

# %% This file

"""
uses embedding vector from GloVe glove.twitter.27B
train LSTM neural networks
"""

# %% Workflow

"""
# load engineered datasets
# load GloVe vectors
# map words in pre-processed datasets to GloVe indices, save to files 
    # unknown/padded words are set to index = 0, which will be mapped to zero embedding vectors
# define LSTM model and optimise with Adam
# predict

"""


# %% Preamble

# Make the output look better
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import keras
from keras.initializers import glorot_uniform
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, Concatenate, BatchNormalization
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import os
import seaborn as sns
import re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None

os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\9_nlp_disaster_tweets')


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


# [(i, index_to_word[i]) for i, vec in enumerate(list(word_to_vec_map.values())) if len(vec) != 100]
# []


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


# %% LSTM with embedding layer


"""
Overview of model:
Embedding ---> Bidirectional LSTM ---> Bidirectional LSTM ---> Dropout ---> Fully Connected with tanh ---> output with sigmoid
"""


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

    # !!! Weights are created when the Model is first called on inputs or build() is called with an input_shape.
    # somehow embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix) does not work...
    # in our model, embedding_layer takes the input layer with input shape (max_len,)
    # here embedding_layer does not know max_len yet, so use (None,)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    return embedding_layer


# embedding_layer00 = pretrained_embedding_layer(word_to_vec_map, word_to_index)
# print("weights[0][1][3] =", embedding_layer00.get_weights()[0][1][3])
# weights[0][1][3] = 0.21751


"""
3. Building an LSTM model
architecture optimised with Keras tuner, plus some regularisations to reduce overfitting
"""
# !!!


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

    # !!! You must set return_sequences = True when stacking LSTM layers (except the last LSTM layer), so that the next LSTM layer has as input a sequence of length = num_timesteps
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

# %% fit model


model = tweet_classification_model(
    (max_len,), (num_features,), word_to_vec_map, word_to_index)


lr_decay = PolynomialDecay(initial_learning_rate=0.0001,
                           decay_steps=20, end_learning_rate=0.00001, power=2)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, threshold=0.5)]
)


callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_f1_score', mode='max', patience=30, min_delta=0.00001, restore_best_weights=True)


# input_shape = [(m, max_len), (m, num_features)]
history = model.fit([X_train_indices, X_train_no_text_encoded2], y_train1,
                    validation_data=([X_valid_indices, X_valid_no_text_encoded2],
                                     y_valid1),
                    epochs=1000, callbacks=[callback])


# Epoch 59/1000
# 191/191 [==============================] - 43s 225ms/step - loss: 2.7831 - accuracy: 0.9571 - f1_score: 0.9504 - val_loss: 3.5519 - val_accuracy: 0.7886 - val_f1_score: 0.7399
# still overfitting with the use of regularisations

# del model

# tf.keras.backend.clear_session()

# from numba import cuda
# device = cuda.get_current_device()
# device.reset()


history_df = pd.DataFrame(history.history)
history_df.columns

model.evaluate([X_valid_indices, X_valid_no_text_encoded2], y_valid1)


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


"""predict"""

prediction = model.predict([X_test_indices, X_test_no_text_encoded2]).squeeze()
prediction_bool = (prediction > 0.5).astype(int)
test_df = pd.read_csv("test.csv", header=[0])
LSTM_pred = pd.Series(prediction_bool, index=test_df['id'], name='target')
LSTM_pred.to_csv('LSTM_pred.csv')
# 0.80784


"""# save the model"""

model.save('LSTM_model_2.h5')
