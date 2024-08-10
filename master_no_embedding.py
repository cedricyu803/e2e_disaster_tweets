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
First trial: without using embedding vectors
uses CountVectorizer and TfidfVectorizer from nltk, combined with other engineered features

"""

# %% Workflow

"""
# load pre-processed text and engineered features

# uses CountVectorizer/TfidfVectorizer from nltk, with stop_words='english', ngram_range = (1,5), and experiments with max_df and min_df
# LogisticRegression, MultinomialNB, XGBClassifier
# predict

"""


# %% Preamble

# Make the output look better
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None
# import re

os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\9_nlp_disaster_tweets')

# %% load engineered datasets

X_test = pd.read_csv("test.csv", header=[0])

X_train_text_preprocessed = np.load(
    'engineered_datasets/X_train_text_preprocessed.npy', allow_pickle=True).squeeze().tolist()
X_valid_text_preprocessed = np.load(
    'engineered_datasets/X_valid_text_preprocessed.npy', allow_pickle=True).squeeze().tolist()
X_test_text_preprocessed = np.load(
    'engineered_datasets/X_test_text_preprocessed.npy', allow_pickle=True).squeeze().tolist()

# X_train_hashmention = np.load('engineered_datasets/X_train_hashmention.npy', allow_pickle=True).squeeze().tolist()
# X_valid_hashmention = np.load('engineered_datasets/X_valid_hashmention.npy', allow_pickle=True).squeeze().tolist()
# X_test_hashmention = np.load('engineered_datasets/X_test_hashmention.npy', allow_pickle=True).squeeze().tolist()

X_train_no_text_encoded2 = np.load(
    'engineered_datasets/X_train_no_text_encoded2.npy', allow_pickle=True)
X_valid_no_text_encoded2 = np.load(
    'engineered_datasets/X_valid_no_text_encoded2.npy', allow_pickle=True)
X_test_no_text_encoded2 = np.load(
    'engineered_datasets/X_test_no_text_encoded2.npy', allow_pickle=True)

y_train = np.load('engineered_datasets/y_train.npy')
y_valid = np.load('engineered_datasets/y_valid.npy')


# !!!
# X_train_added = hstack([X_train, csr_matrix(cols_to_add9)], 'csr')


# %% CountVectorizer with baseline models


"""fit CountVectorizer and transform"""


"""# grid search for CountVectorizer hyperparameters"""
# min_df_list = np.arange(4,5,4)
# max_df_list = np.arange(0.2,0.9,0.2)
# F1_count = []

# from tqdm.notebook import tqdm
# for min_df in min_df_list:
#     for max_df in tqdm(max_df_list):
#         # Fit the CountVectorizer to the training data
#         # minimum document frequency of 10 and max_df = 0.2, ignore stop words, ngram_range = (1,5)
#         vect_count0 = CountVectorizer(min_df=min_df, max_df=max_df, stop_words='english', ngram_range = (1,3)).fit(X_train_text_preprocessed)
#         # vect_count.get_feature_names()[::2000]
#         X_train_text_countvec0 = vect_count0.transform(X_train_text_preprocessed).toarray()
#         X_valid_text_countvec0 = vect_count0.transform(X_valid_text_preprocessed).toarray()

#         X_train_count_added0 = np.hstack([X_train_text_countvec0, X_train_no_text_encoded2])
#         X_valid_count_added0 = np.hstack([X_valid_text_countvec0, X_valid_no_text_encoded2])

#         scaler_count0 = MinMaxScaler()

#         scaler_count0.fit(X_train_count_added0)

#         X_train_count_added_scaled0 = scaler_count0.transform(X_train_count_added0)
#         X_valid_count_added_scaled0 = scaler_count0.transform(X_valid_count_added0)

#         # logistic regression
#         model_count0 = LogisticRegression()
#         model_count0.fit(X_train_count_added_scaled0, y_train)

#         # Predict the transformed test documents
#         predictions_count0 = model_count0.predict(X_valid_count_added_scaled0)
#         F1_count.append((min_df, max_df, f1_score(y_valid, predictions_count0)))


vect_count = CountVectorizer(min_df=3, max_df=0.3, stop_words='english', ngram_range=(
    1, 3)).fit(X_train_text_preprocessed)

X_train_text_countvec = vect_count.transform(
    X_train_text_preprocessed).toarray()
X_valid_text_countvec = vect_count.transform(
    X_valid_text_preprocessed).toarray()
X_test_text_countvec = vect_count.transform(X_test_text_preprocessed).toarray()


print(vect_count.get_feature_names()[::1000])
# ['aba', 'caused structural', 'entire pond', 'hashtag science', 'lose republicans like', 'outdoor', 'punc violent', 'stats', 'war punc']
len(vect_count.get_feature_names())
# 8390

inv_vocab = {vect_count.vocabulary_[
    word]: word for word in vect_count.get_feature_names()}


"""combine with other features"""

X_train_count_added = np.hstack(
    [X_train_text_countvec, X_train_no_text_encoded2])
X_valid_count_added = np.hstack(
    [X_valid_text_countvec, X_valid_no_text_encoded2])
X_test_count_added = np.hstack([X_test_text_countvec, X_test_no_text_encoded2])


"""scaler """

scaler_count = MinMaxScaler()

scaler_count.fit(X_train_count_added)

X_train_count_added_scaled = scaler_count.transform(X_train_count_added)
X_valid_count_added_scaled = scaler_count.transform(X_valid_count_added)
X_test_count_added_scaled = scaler_count.transform(X_test_count_added)


#################################
"""baseline models that do not require feature normalisation"""


# logistic regression


model_count = LogisticRegression(max_iter=10000)
model_count.fit(X_train_count_added_scaled, y_train)

# Predict the transformed test documents
predictions_count = model_count.predict(X_valid_count_added_scaled)

# !!!
print('F1 score: ', f1_score(y_valid, predictions_count))
# F1 score:  0.7564422277639236

# get the feature names as numpy array
feature_names_count = np.array(vect_count.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index_count = model_count.coef_[0][:8390].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1]
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}'.format(
    feature_names_count[sorted_coef_index_count[:10]]))
print('Largest Coefs: \n{}'.format(
    feature_names_count[sorted_coef_index_count[:-11:-1]]))
# Smallest Coefs:
# ['mode' 'best' 'buy' 'special' 'super' 'august number punc' 'cake'
#  'user punc' 'entertainment' 'longer']
# Largest Coefs:
# ['hiroshima' 'california' 'storm' 'crash' 'plane' 'war' 'riots' 'train'
#  'fires' 'tornado']

model_count_pred = model_count.predict(X_test_count_added_scaled)
model_count_pred = pd.Series(
    model_count_pred, index=X_test['id'], name='target')
model_count_pred.to_csv('y_test_pred_count_logreg.csv')
# 0.79190

#################################


MNB_clf_count = MultinomialNB(alpha=0.1)
MNB_clf_count.fit(X_train_count_added_scaled, y_train)

# !!!
print('F1 score: ', f1_score(
    y_valid, MNB_clf_count.predict(X_valid_count_added_scaled)))
# F1 score:  0.711590296495957


#################################

# !!!
RFC_model_count = RandomForestClassifier(n_jobs=6)
RFC_model_count.fit(X_train_count_added_scaled, y_train)
print('F1 score: ', f1_score(
    y_valid, RFC_model_count.predict(X_valid_count_added_scaled)))
# F1 score:  0.7205750224618149


#################################


XGBC_model_count = XGBClassifier(eval_metric='auc', n_jobs=6)
XGBC_model_count.fit(X_train_count_added_scaled, y_train, eval_set=[(
    X_train_count_added_scaled, y_train), (X_valid_count_added_scaled, y_valid)], early_stopping_rounds=40)
# [99]	validation_0-auc:0.94412	validation_1-auc:0.84282
# !!!
print('F1 score: ', f1_score(
    y_valid, XGBC_model_count.predict(X_valid_count_added_scaled)))
# F1 score:  0.7297071129707113

index0 = [word for word in list(inv_vocab.values())] + ['char_count', 'punc_ratio', 'cap_ratio',
                                                        'sentence_count_freq_encoded', 'stopword_num_freq_encoded',
                                                        'hashtag_num_freq_encoded', 'at_num_freq_encoded',
                                                        'url_num_freq_encoded', 'country_mention_num_freq_encoded',
                                                        'token_count_freq_encoded', 'keyword_mean_encoded']

XGBR_feature_importances_count = pd.Series(
    XGBC_model_count.feature_importances_, index=index0).sort_values(ascending=False)

# keyword_mean_encoded                0.038112
# country_mention_num_freq_encoded    0.016730
# hiroshima                           0.012004
# url_num_freq_encoded                0.010691
# california                          0.008889
# number                              0.008023
# killed                              0.007949
# hashtag jobs                        0.007540
# chile                               0.007475
# near                                0.006474
# rd                                  0.006356
# love                                0.006328
# road                                0.006246
# alabama                             0.006164
# train                               0.005970
# putin                               0.005889
# breaking                            0.005852
# wedding                             0.005852
# warning                             0.005731
# ebola                               0.005673
# storm                               0.005652
# new                                 0.005580
# lives                               0.005503
# report                              0.005479
# repeat url                          0.005468
# said                                0.005427
# rain                                0.005416
# city                                0.005406
# japanese                            0.005348
# sentence_count_freq_encoded         0.005240
# dtype: float32


# XGBC_model_count_pred = XGBC_model_count.predict(vect_count.transform(X_test))
# XGBC_model_count_pred = pd.Series(XGBC_model_count_pred, index = testData['id'], name = 'sentiment')
# XGBC_model_count_pred.to_csv('y_test_pred_count_XGBC.csv')


# %% TfidfVectorizer with baseline models

""" fit TfidfVectorizer and transform"""


vect_tfidf = TfidfVectorizer(min_df=3, max_df=0.3, stop_words='english', ngram_range=(
    1, 3)).fit(X_train_text_preprocessed)

X_train_text_tfidfvec = vect_tfidf.transform(
    X_train_text_preprocessed).toarray()
X_valid_text_tfidfvec = vect_tfidf.transform(
    X_valid_text_preprocessed).toarray()
X_test_text_tfidfvec = vect_tfidf.transform(X_test_text_preprocessed).toarray()


print(vect_tfidf.get_feature_names()[::1000])
# ['aba', 'caused structural', 'entire pond', 'hashtag science', 'lose republicans like', 'outdoor', 'punc violent', 'stats', 'war punc']
len(vect_tfidf.get_feature_names())
# 8390


"""combine with other features"""

X_train_tfidf_added = np.hstack(
    [X_train_text_tfidfvec, X_train_no_text_encoded2])
X_valid_tfidf_added = np.hstack(
    [X_valid_text_tfidfvec, X_valid_no_text_encoded2])
X_test_tfidf_added = np.hstack([X_test_text_tfidfvec, X_test_no_text_encoded2])


"""scaler """

scaler_tfidf = MinMaxScaler()

scaler_tfidf.fit(X_train_tfidf_added)

X_train_tfidf_added_scaled = scaler_tfidf.transform(X_train_tfidf_added)
X_valid_tfidf_added_scaled = scaler_tfidf.transform(X_valid_tfidf_added)
X_test_tfidf_added_scaled = scaler_tfidf.transform(X_test_tfidf_added)


#################################
"""baseline models that do not require feature normalisation"""


# logistic regression


model_tfidf = LogisticRegression(max_iter=10000)
model_tfidf.fit(X_train_tfidf_added_scaled, y_train)

# Predict the transformed test documents
predictions_tfidf = model_tfidf.predict(X_valid_tfidf_added_scaled)

# !!!
print('F1 score: ', f1_score(y_valid, predictions_tfidf))
# F1 score:  0.7495854063018241

# get the feature names as numpy array
feature_names_tfidf = np.array(vect_tfidf.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index_tfidf = model_tfidf.coef_[0][:8390].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1]
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}'.format(
    feature_names_tfidf[sorted_coef_index_tfidf[:10]]))
print('Largest Coefs: \n{}'.format(
    feature_names_tfidf[sorted_coef_index_tfidf[:-11:-1]]))
# Smallest Coefs:
# ['best' 'special' 'mode' 'good' 'super' 'august number punc' 'new'
#  'better' 'user punc' 'united']
# Largest Coefs:
# ['hiroshima' 'california' 'crash' 'report' 'storm' 'train' 'war' 'riots'
# 'cos' 'plane']

model_tfidf_pred = model_tfidf.predict(X_test_count_added_scaled)
model_tfidf_pred = pd.Series(
    model_tfidf_pred, index=X_test['id'], name='target')
model_tfidf_pred.to_csv('y_test_pred_tfidf_logreg.csv')

#################################


MNB_clf_tfidf = MultinomialNB(alpha=0.1)
MNB_clf_tfidf.fit(X_train_tfidf_added_scaled, y_train)

# !!!
print('F1 score: ', f1_score(
    y_valid, MNB_clf_tfidf.predict(X_valid_tfidf_added_scaled)))
# F1 score:  0.7118942731277533


#################################

# !!!
RFC_model_tfidf = RandomForestClassifier(n_jobs=6)
RFC_model_tfidf.fit(X_train_tfidf_added_scaled, y_train)
print('F1 score: ', f1_score(
    y_valid, RFC_model_tfidf.predict(X_valid_tfidf_added_scaled)))
# F1 score:  0.7235188509874326


#################################


XGBC_model_tfidf = XGBClassifier(eval_metric='auc', n_jobs=6)
XGBC_model_tfidf.fit(X_train_tfidf_added_scaled, y_train, eval_set=[(
    X_train_tfidf_added_scaled, y_train), (X_valid_tfidf_added_scaled, y_valid)], early_stopping_rounds=40)
# [99]	validation_0-auc:0.94412	validation_1-auc:0.84282
# !!!
print('F1 score: ', f1_score(
    y_valid, XGBC_model_tfidf.predict(X_valid_tfidf_added_scaled)))
# F1 score:  0.7483108108108107
print('AUC: ', roc_auc_score(
    y_valid, XGBC_model_tfidf.predict(X_valid_tfidf_added_scaled)))
# AUC:  0.7890329599455688


index0 = [word for word in list(inv_vocab.values())] + ['char_count', 'punc_ratio', 'cap_ratio',
                                                        'sentence_count_freq_encoded', 'stopword_num_freq_encoded',
                                                        'hashtag_num_freq_encoded', 'at_num_freq_encoded',
                                                        'url_num_freq_encoded', 'country_mention_num_freq_encoded',
                                                        'token_count_freq_encoded', 'keyword_mean_encoded']

XGBR_feature_importances = pd.Series(
    XGBC_model_tfidf.feature_importances_, index=index0).sort_values(ascending=False)

# keyword_mean_encoded                0.035902
# country_mention_num_freq_encoded    0.015010
# url_num_freq_encoded                0.010903
# hiroshima                           0.009294
# repeat url                          0.007829
# killed                              0.007638
# allcaps                             0.006751
# allcaps punc                        0.006738
# california                          0.006686
# number punc number                  0.006595
# rain                                0.006551
# putin                               0.006156
# video                               0.006079
# train                               0.005936
# ebola                               0.005910
# alabama                             0.005850
# sentence_count_freq_encoded         0.005782
# warning                             0.005689
# repeat punc                         0.005592
# user hashtag                        0.005569
# hashtag jobs                        0.005521
# near                                0.005271
# rd                                  0.005161
# explosion                           0.005082
# army                                0.005063
# media                               0.005038
# report                              0.004992
# car                                 0.004983
# wedding                             0.004961
# severe                              0.004803
# dtype: float32

# XGBC_model_tfidf_pred = XGBC_model_tfidf.predict(vect_tfidf.transform(X_test))
# XGBC_model_tfidf_pred = pd.Series(XGBC_model_tfidf_pred, index = testData['id'], name = 'sentiment')
# XGBC_model_tfidf_pred.to_csv('y_test_pred_tfidf_XGBC.csv')
