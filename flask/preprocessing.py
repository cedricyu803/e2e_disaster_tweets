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

#%% This file

"""
text pre-processing, feature extractions, engineering including encoding

"""


#%% Workflow

"""
# load dataset and train-validation split

# we will use GloVe Twitter embeddings for LSTM networks; see script on the website

Text pre-processing:
# remove HTML syntax
# strip front and end whitespaces
# expand abbreviation n't to not and 's to  s
# record the ratios of capital letters and punctuation marks to total characters
# replace url by '<url>' token
# replace @mention by '<user>' token
# replace # by <hashtag> token, and split the hashtag_body by capital letters unless it is all cap
# represent numbers by '<number>' token
# replace repeated punctuation marks by punctuation mark + <repeat> token
# replace punctuation marks (except ?!.) by <punc> token
# remove extra whitespaces
# (we do not handle emojis) in this exercise
# (remove stop words? Do not!)
# for nltk vectorisers:
# lower case
# lemmatize
for LSTM:
# lower case
# pad sequences by hand to max_len=? (max char length is 280)
for transformer: 
# not much else to do


Feature extractions and engineering:
# number of characters
# punctuation and capital letter ratio
# sentence count: frequency encode
# number of stopwords: frequency encode
# number of #, @, urls: frequency encode
# keyword: target encode
# country mentions with GeoText(text).country_mentions: frequency encode

# encoding and fillna

hashtag and mentions lists for each Tweet: also pre-process and feed to LSTM?

"""


#%% Preamble


import pandas as pd
# Make the output look better
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None  # default='warn' # ignores warning about dropping columns inplace
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns

import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Works\9_nlp_disaster_tweets')

#%% load datasets


train = pd.read_csv ("train.csv", header = [0])
X_test = pd.read_csv ("test.csv", header = [0])


#%% train-validation split


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train.drop(['target'], axis = 1), train['target'], random_state=0, train_size = 0.8)


#%% pre-processing

"""
text pre-processing.
# we loosely follow 
# https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""


"""remove mojibake"""

from bs4 import BeautifulSoup

def no_mojibake(row):
    row['text_no_mojibake'] = BeautifulSoup(row['text']).get_text().strip()
    return row

X_train = X_train.apply(no_mojibake, axis = 1)
X_valid = X_valid.apply(no_mojibake, axis = 1)
X_test = X_test.apply(no_mojibake, axis = 1)

""" text pre-processing"""

def text_preprocess(text):
    # remove html syntax and strip front and end whitespace (already did)
    # text1 = BeautifulSoup(text).get_text() 
    
    # expand abbreviation n't to not and 's to  s
    text1 = re.sub(r"n't", ' not', text)
    text1 = re.sub(r"'s", ' s', text1)
    
    
    # replace url by '<url>' token
    text1 = re.sub(r'(https?://[\S]*)', '<url>', text1)
    # replace @mention by '<user>' token
    text1 = re.sub(r'@\w+', "<user>", text1)
    
    # replace # by <hashtag> token, and split the hashtag_body by capital letters unless it is all cap
    hash_iter = list(filter(None, re.findall(r'#(\w+)', text1)))
    if len(hash_iter) > 0:
        for item in hash_iter:
            if item.isupper() == True:
                hash_words = item + ' <allcaps>'
            else:
                hash_words = ' '.join(list(filter(None, re.split(r'([A-Z]?[a-z]*)', item))))
            text1 = re.sub(item, hash_words, text1)
            text1 = re.sub(r'#', '<hashtag> ', text1)
    
    # represent numbers by '<number>' token
    text1 = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' <number> ', text1)
    
    # replace repeated punctuation marks (!?. only) by punctuation mark + <repeat> token
    text1 = re.sub(r'([!?\.]){2,}', r'\1' + ' <repeat>', text1)
    # add spaces before and after (!?.)
    text1 = re.sub(r'([!?\.])', r' \1 ', text1)
    
    # replace punctuation marks (except ?!.) by <punc> token
    text1 = re.sub(r'[\"#$%\&\'\(\)\*\+,\-/:;=@\[\]\^_`\{\|\}~\\]+', ' <punc> ', text1)
    text1 = re.sub(r'(<[^A-Za-z0-9_-]+|[^A-Za-z0-9_-]+>)', ' <punc> ', text1)

    # remove extra whitespaces
    text1 = text1.strip()
    text1 = re.sub('\s+', ' ', text1)

    # lemmatise
    # text1 = WNlemma_n.lemmatize(text1)

    # lower case
    text1 = text1.lower()
    
    return text1

def pd_text_preprocess(row):
    row['text_processed'] = text_preprocess(row['text_no_mojibake'])
    return row



X_train = X_train.apply(pd_text_preprocess, axis = 1)
X_valid = X_valid.apply(pd_text_preprocess, axis = 1)
X_test = X_test.apply(pd_text_preprocess, axis = 1)



"""
The pre-processed texts (without other features) will be fed to the LSTM/transformer parts of the neural networks separately from the other features
"""

X_train_text_preprocessed = X_train[['text_processed']].to_numpy()
X_valid_text_preprocessed = X_valid[['text_processed']].to_numpy()
X_test_text_preprocessed = X_test[['text_processed']].to_numpy()
# X_train_text_preprocessed.shape
# Out[73]: (6090, 1)

# np.save('X_train_text_preprocessed.npy', X_train_text_preprocessed)
# np.save('X_valid_text_preprocessed.npy', X_valid_text_preprocessed)
# np.save('X_test_text_preprocessed.npy', X_test_text_preprocessed)

""" if you want lemmatised text"""

import nltk
WNlemma_n = nltk.WordNetLemmatizer()

def nltk_lemmatize(text):
    # text1 = nltk.word_tokenize(text)
    text1 = WNlemma_n.lemmatize(text)
    return text1

def pd_nltk_lemmatize(row):
    row['text_processed'] = nltk_lemmatize(row['text_processed'])
    return row


X_train_text_preprocessed_lemmatized = X_train[['text_processed']].apply(pd_nltk_lemmatize, axis = 1).to_numpy()
X_valid_text_preprocessed_lemmatized = X_valid[['text_processed']].apply(pd_nltk_lemmatize, axis = 1).to_numpy()
X_test_text_preprocessed_lemmatized = X_test[['text_processed']].apply(pd_nltk_lemmatize, axis = 1).to_numpy()


# np.save('X_train_text_preprocessed_lemmatized.npy', X_train_text_preprocessed_lemmatized)
# np.save('X_valid_text_preprocessed_lemmatized.npy', X_valid_text_preprocessed_lemmatized)
# np.save('X_test_text_preprocessed_lemmatized.npy', X_test_text_preprocessed_lemmatized)
# np.save('y_train.npy', y_train)
# np.save('y_valid.npy', y_valid)



# """ get token count from processed text"""

# import nltk

# def token_count(row):
#     row['token_count'] = len(nltk.word_tokenize(row['text_processed']))
#     return row

# X_train = X_train.apply(token_count, axis = 1)
# X_valid = X_valid.apply(token_count, axis = 1)
# X_test = X_test.apply(token_count, axis = 1)
# # max number of tokens is 107 in the processed datasets


#%% load embedding matrix pre-trained using glove

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
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.twitter.27B/glove.twitter.27B.100d.txt')

# [(i, index_to_word[i]) for i, vec in enumerate(list(word_to_vec_map.values())) if len(vec) != 100]
# []


# contains <hashtag>, <user>, <url>, <repeat>, <number>


#%% tokenise and pad text for neural networks

# max number of tokens is 107 in the processed datasets
# there are special tokens such as <hashtag>; using ntk.word_tokenized will break them apart. tokens in pre-processed text are separated by a single space

def tokenize_pad_text(X, max_len = 128):
    
    X_ = X.squeeze()
    Xtokenized = []
    
    for m in range(len(X_)):
        # tokenise
        text1 = X_[m].split(' ')
        # pad to max_len by '-1 empty'
        if len(text1) < max_len:
            text1 += ['-1 empty' for i in range(max_len - len(text1))]
        if len(text1) > max_len:
            text1 = [word for i, word in enumerate(text1) if i < max_len]
        Xtokenized.append(text1)
    
    Xtokenized = np.array(Xtokenized)
    return Xtokenized


max_len = 128

X_train_tokenize_pad = tokenize_pad_text(X_train_text_preprocessed, max_len = max_len)
X_valid_tokenize_pad = tokenize_pad_text(X_valid_text_preprocessed, max_len = max_len)
X_test_tokenize_pad = tokenize_pad_text(X_test_text_preprocessed, max_len = max_len)


#%% function that converts training sentences into indices with padding

# set unknown and padded tokens to 0; the embedding vectors will be zero-vectors

padded_token_index = 0
unknown_token_index = 0

def sentences_to_indices(X, word_to_index):
    """
    Converts an array of [padded, tokenised sentences] into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to 'Embedding()'
    
    Arguments:
    X -- array of padded, tokenised sentences, of shape (m, max_len)
    word_to_index -- a dictionary containing the each word mapped to its index
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    # set padded tokens to index = padded_token_index
    word_to_index0 = word_to_index.copy()
    word_to_index0['-1 empty'] = padded_token_index
    
    # set unknown tokens to index = unknown_token_index
    def word_to_index00(key):
        return word_to_index0.get(key, unknown_token_index)
    
    # speed up computation with vectorisation
    X_indices = np.vectorize(word_to_index00)(X)
    
    return X_indices


# if you want larger max_len, just vstack more zeros
X_train_indices = sentences_to_indices(X_train_tokenize_pad, word_to_index)
X_valid_indices = sentences_to_indices(X_valid_tokenize_pad, word_to_index)
X_test_indices = sentences_to_indices(X_test_tokenize_pad, word_to_index)
# X_train_indices.shape
# Out[74]: (6090, 128)


# np.save('X_train_indices.npy', X_train_indices)
# np.save('X_valid_indices.npy', X_valid_indices)
# np.save('X_test_indices.npy', X_test_indices)


# X_train_indices = np.load('X_train_indices.npy')
# X_valid_indices = np.load('X_valid_indices.npy')
# X_test_indices = np.load('X_test_indices.npy')

#%% feature extractions 


import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

from geotext import GeoText


def get_features(row):
    
    # character count
    row['char_count'] = len(row['text_no_mojibake'])
    
    # punctuation mark and capital letter to character length ratios
    row['punc_ratio'] = len(''.join(re.findall(r'[\.\?!\"#$%\&\'\(\)\*\+,\-/:;=@\[\]\^_`\{\|\}~\\]+', row['text_no_mojibake']))) / len(row['text_no_mojibake'])
    row['cap_ratio'] = len(''.join(re.findall(r'[A-Z]', row['text_no_mojibake']))) / len(row['text_no_mojibake'])
    
    # number of sentences
    row['sentence_count'] = len(nltk.sent_tokenize(row['text_no_mojibake']))
    
    # number of stopwords
    tokens = nltk.word_tokenize(row['text_processed'])
    stop_wordsn = len([token for token in tokens if token in stops])
    row['stopword_num'] = stop_wordsn
    
    # hashtags
    hashtags_list = re.findall(r'#(\w+)', row['text_no_mojibake'])
    hashtags_list = list(filter(None, hashtags_list))
    row['hashtags'] = hashtags_list
    row['hashtag_num'] = len(hashtags_list)
    
    # mentions
    at_list = re.findall(r'@(\w+)', row['text_no_mojibake'])
    at_list = list(filter(None, at_list))
    row['at'] = at_list
    row['at_num'] = len(at_list)
    
    # URLs
    url_list = re.findall(r'(https?://[\S]*)', row['text_no_mojibake'])
    url_list = list(filter(None, url_list))
    row['url'] = url_list
    row['url_num'] = len(url_list)
    
    # country mentions
    if len(list(zip(*GeoText(row['text_no_mojibake']).country_mentions.items()))) > 0:
        row['country_mention_num'] = np.array(list(zip(*GeoText(row['text_no_mojibake']).country_mentions.items()))[1]).sum()
    else: 
        row['country_mention_num'] = 0
    
    # token count
    row['token_count'] = len(nltk.word_tokenize(row['text_processed']))
    
    return row

X_train = X_train.apply(get_features, axis = 1)
X_valid = X_valid.apply(get_features, axis = 1)
X_test = X_test.apply(get_features, axis = 1)


X_train_hashmention = X_train[['hashtags', 'at']]
X_valid_hashmention = X_valid[['hashtags', 'at']]
X_test_hashmention = X_test[['hashtags', 'at']]


X_train_no_text = X_train[['id', 'keyword', 'char_count', 'punc_ratio', 'cap_ratio', 'sentence_count', 'stopword_num', 'hashtag_num', 'at_num', 'url_num', 'country_mention_num', 'token_count']]
X_valid_no_text = X_valid[['id', 'keyword', 'char_count', 'punc_ratio', 'cap_ratio', 'sentence_count', 'stopword_num', 'hashtag_num', 'at_num', 'url_num', 'country_mention_num', 'token_count']]
X_test_no_text = X_test[['id', 'keyword', 'char_count', 'punc_ratio', 'cap_ratio', 'sentence_count', 'stopword_num', 'hashtag_num', 'at_num', 'url_num', 'country_mention_num', 'token_count']]



# np.save('X_train_hashmention.npy', X_train_hashmention)
# np.save('X_valid_hashmention.npy', X_valid_hashmention)
# np.save('X_test_hashmention.npy', X_test_hashmention)
# np.save('X_train_no_text.npy', X_train_no_text)
# np.save('X_valid_no_text.npy', X_valid_no_text)
# np.save('X_test_no_text.npy', X_test_no_text)



#%% feature encoding


import category_encoders as ce


"""# frequency-encoding"""


freq_cols = ['sentence_count', 'stopword_num', 'hashtag_num', 'at_num', 'url_num', 'country_mention_num', 'token_count']

# change data type to object before feeding it into the encoder
X_train_no_text[freq_cols] = X_train_no_text[freq_cols].astype(object)
X_valid_no_text[freq_cols] = X_valid_no_text[freq_cols].astype(object)
X_test_no_text[freq_cols] = X_test_no_text[freq_cols].astype(object)


freq_encoder = ce.count.CountEncoder()
freq_encoder.fit(X_train_no_text[freq_cols])


X_train_no_text_encoded1 = pd.concat([X_train_no_text, freq_encoder.transform(X_train_no_text[freq_cols]).rename(columns = dict(zip(freq_cols, [col + '_freq_encoded' for col in freq_cols])))], axis = 1).drop(freq_cols, axis = 1)


X_valid_no_text_encoded1 = pd.concat([X_valid_no_text, freq_encoder.transform(X_valid_no_text[freq_cols]).rename(columns = dict(zip(freq_cols, [col + '_freq_encoded' for col in freq_cols])))], axis = 1).drop(freq_cols, axis = 1)


X_test_no_text_encoded1 = pd.concat([X_test_no_text, freq_encoder.transform(X_test_no_text[freq_cols]).rename(columns = dict(zip(freq_cols, [col + '_freq_encoded' for col in freq_cols])))], axis = 1).drop(freq_cols, axis = 1)



"""# target mean encoding"""

target_mean_cols = ['keyword']


# these columns have been converted to object type for TargetEncoder
target_mean_encoder = ce.target_encoder.TargetEncoder()
target_mean_encoder.fit(X_train_no_text_encoded1[target_mean_cols], y_train)

X_train_no_text_encoded2 = pd.concat([X_train_no_text_encoded1, target_mean_encoder.transform(X_train_no_text_encoded1[target_mean_cols]).rename(columns = dict(zip(target_mean_cols, [col + '_mean_encoded' for col in target_mean_cols])))], axis = 1).drop(target_mean_cols, axis = 1)


X_valid_no_text_encoded2 = pd.concat([X_valid_no_text_encoded1, target_mean_encoder.transform(X_valid_no_text_encoded1[target_mean_cols]).rename(columns = dict(zip(target_mean_cols, [col + '_mean_encoded' for col in target_mean_cols])))], axis = 1).drop(target_mean_cols, axis = 1)


X_test_no_text_encoded2 = pd.concat([X_test_no_text_encoded1, target_mean_encoder.transform(X_test_no_text_encoded1[target_mean_cols]).rename(columns = dict(zip(target_mean_cols, [col + '_mean_encoded' for col in target_mean_cols])))], axis = 1).drop(target_mean_cols, axis = 1)


"""drop columns"""

cols_to_keep = ['char_count', 'punc_ratio', 'cap_ratio',
                'sentence_count_freq_encoded', 'stopword_num_freq_encoded',
                'hashtag_num_freq_encoded', 'at_num_freq_encoded',
                'url_num_freq_encoded', 'country_mention_num_freq_encoded',
                'token_count_freq_encoded', 'keyword_mean_encoded']



X_train_no_text_encoded2 = X_train_no_text_encoded2[cols_to_keep]
X_valid_no_text_encoded2 = X_valid_no_text_encoded2[cols_to_keep]
X_test_no_text_encoded2 = X_test_no_text_encoded2[cols_to_keep]
# X_train_no_text_encoded2.shape
# Out[75]: (6090, 11)

"""fillna with [training set] mean"""

# X_valid_no_text_encoded2.isnull().sum()
# char_count                          0
# punc_ratio                          0
# cap_ratio                           0
# sentence_count_freq_encoded         1
# stopword_num_freq_encoded           0
# hashtag_num_freq_encoded            0
# at_num_freq_encoded                 0
# url_num_freq_encoded                0
# country_mention_num_freq_encoded    1
# token_count_freq_encoded            0
# keyword_mean_encoded                0
# dtype: int64

# X_test_no_text_encoded2.isnull().sum()
# char_count                          0
# punc_ratio                          0
# cap_ratio                           0
# sentence_count_freq_encoded         0
# stopword_num_freq_encoded           1
# hashtag_num_freq_encoded            1
# at_num_freq_encoded                 1
# url_num_freq_encoded                0
# country_mention_num_freq_encoded    2
# token_count_freq_encoded            1
# keyword_mean_encoded                0
# dtype: int64


X_valid_no_text_encoded2 = X_valid_no_text_encoded2.fillna(X_train_no_text_encoded2.mean())
X_test_no_text_encoded2 = X_test_no_text_encoded2.fillna(X_train_no_text_encoded2.mean())

# !!! not yet normalised

# np.save('X_train_no_text_encoded2.npy', X_train_no_text_encoded2)
# np.save('X_valid_no_text_encoded2.npy', X_valid_no_text_encoded2)
# np.save('X_test_no_text_encoded2.npy', X_test_no_text_encoded2)





# y_train = np.load('y_train.npy')
# y_valid = np.load('y_valid.npy')
# y_train1 = np.expand_dims(y_train, -1)
# y_valid1 = np.expand_dims(y_valid, -1)







