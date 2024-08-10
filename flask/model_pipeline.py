# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 20:00:00 2021

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
text pre-processing and model fitting pipeline. we only use the processed text without other features

"""

"""
Pipeline with ColumnTransformer for pre-processing

We use sklearn.compose.ColumnTransformer to define pre-processing, which is to be passed to a pipeline.


# We also define custom estimators to do any column operations we want
# We create a custom estimator as a new estimator class; see 'Intro to Data Science' 1.6.
# We import from sklearn.base <<BaseEstimator>>, which provides a base class that enables to set and get parameters of the estimator, and <<TransformerMixin>>, which implements the combination of fit and transform, i.e. fit_transform

# A custom estimator consists of three methods:
# 1. __init__ : This is the constructor. Called when pipeline is initialized.
# 2. fit() : Called when we fit the pipeline.
# 3. transform() : Called when we use fit or transform on the pipeline.

# After that, the custom estimator is called in ColumnTransformer. The syntax is 
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('estimator1', Transformer_Example1(), <list of columns>), ...)
#     ], remainder='passthrough')
# Here, by default, remainder='drop' which drops all remaining columns. Here we can specify 'passthrough'. 
# The ColumnTransformer returns an array (each time the processed column is moved the the left)

# Then we pass the ColumnTransformer to a Pipeline


Summary: 

from sklearn.base import BaseEstimator, TransformerMixin
class Transformer_Example1(BaseEstimator, TransformerMixin) : 
    def __init__(self): # This will be called when the ColumnTransformer is called
        # print('__init__ is called.\n')
        pass
    
    def fit(self, X, y=None) : 
        # print('fit is called.\n')
        return self
    
    def transform(self, X, y=None) : 
        # print('return is called.\n')
        X_ = X.copy() # create a copy to avoid changing the original dataset
        X_ =  2 * X_  # cook up some manipulation to column X2
        return X_

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('estimator1', Transformer_Example1(), <list of columns>), ...)
    ], remainder='passthrough')


"""

#%% Workflow

"""
# load dataset and train-validation split

# for simplicity, we will only use the 'text' feature column and new features therefrom, and use tf-idf + logistic regression

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

-------------------------
Not used: 

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
import re
import matplotlib.pyplot as plt
import seaborn as sns

# import os
# os.chdir(r'C:\Users\Cedric Yu\Desktop\Data Science\flask\ML deployments\nlp_disaster_tweets')

# for pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# for pre-processing
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from geotext import GeoText
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


#%% define custom estimators for pre-processing


"""
text pre-processing.
# we loosely follow 
# https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

non_text_cols = ['keyword', 'char_count', 'punc_ratio', 'cap_ratio', 'sentence_count', 'stopword_num', 'hashtag_num', 'at_num', 'url_num', 'country_mention_num', 'token_count']

def text_preprocess(X): 
    
    X_ = X.copy()
    
    """remove mojibake"""
    def no_mojibake(row):
        row['text_no_mojibake'] = BeautifulSoup(row['text']).get_text().strip()
        return row
    
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
    
    
    """feature extractions """
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
    
    X_ = X_.apply(no_mojibake, axis = 1)
    X_ = X_.apply(pd_text_preprocess, axis = 1)
    # X_ = X_.apply(get_features, axis = 1)
    
    # X_ = X_.drop(['location', 'text', 'text_no_mojibake', 'hashtags', 'at', 'url'], axis = 1)
    
    X_text = X_['text_processed']
    # X_notext = X_[non_text_cols]
    
    # return X_text, X_notext
    return X_text



"""feature encoding"""
"""frequency-encoding"""


freq_cols = ['sentence_count', 'stopword_num', 'hashtag_num', 'at_num', 'url_num', 'country_mention_num', 'token_count']

# do not use custom class when pickling pipeline!!! Move .astype(object) to the text_preprocess function above
class freq_encode_pipe(BaseEstimator, TransformerMixin): 
    def __init__(self): # This will be called when the ColumnTransformer is called
        # frequency encoder
        self.freq_encoder = ce.count.CountEncoder()
        pass
    
    def fit(self, X, y=None): 
        X_ = X.copy()
        X_ = X_.astype(object)
        self.freq_encoder.fit(X_)
        return self
    
    def transform(self, X, y=None): 
        
        X_ = X.copy() # create a copy to avoid changing the original dataset
        X_ = X_.astype(object)
        X_ = self.freq_encoder.transform(X_)
        
        
        return X_



"""# target mean encoding"""

target_mean_cols = ['keyword']


class target_encode_pipe(BaseEstimator, TransformerMixin): 
    def __init__(self): # This will be called when the ColumnTransformer is called
        # frequency encoder
        self.target_mean_encoder = ce.target_encoder.TargetEncoder()
        pass
    
    def fit(self, X, y=None): 
        X_ = X.copy()
        self.target_mean_encoder.fit(X_, y)
        return self
    
    def transform(self, X, y=None): 
        
        X_ = X.copy() # create a copy to avoid changing the original dataset
        X_ = self.target_mean_encoder.transform(X_)
        
        return X_

# columns to drop
cols_to_drop = ['id', 'keyword', 'sentence_count', 'stopword_num', 'hashtag_num', 'at_num', 'url_num', 'country_mention_num', 'token_count', 'text_processed']


#%% pre-processing with ColumnTransformer and Pipeline


# pipeline_text = Pipeline([
#     ('tfidf', TfidfVectorizer(min_df=3, max_df=0.3, stop_words='english', ngram_range = (1,3))),
#     # ('scaler', MinMaxScaler())
#     ])


# encoders_nontext = ColumnTransformer(
#     transformers=[
#         ('freq_encode', freq_encode_pipe(), freq_cols),
#         ('target_encode', target_encode_pipe(), target_mean_cols),
#         ],
#         remainder='passthrough')

# # TfidfVectorizer(min_df=3, max_df=0.3, stop_words='english', ngram_range = (1,3))



# """# define pipelines with the above ColumnTransformer"""
# pipeline_nontext = Pipeline([
#     ('encoders_nontext', encoders_nontext),
#     ('imputer', SimpleImputer(strategy='mean')),
#     # ('scaler', MinMaxScaler())
#     ])



# from sklearn.linear_model import LogisticRegression

# pipeline_model = Pipeline([
#     ('scaler', MinMaxScaler()),
#     ('logreg', LogisticRegression(max_iter = 10000, class_weight = 'balanced'))
#     ])

"""

#%% load datasets


train = pd.read_csv ("train.csv", header = [0])
X_test = pd.read_csv ("test.csv", header = [0])


#%% train-validation split


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train.drop(['target'], axis = 1), train['target'], random_state=0, train_size = 0.8)



# X_train_text, X_train_nontext = text_preprocess(X_train)
X_train_text = text_preprocess(X_train)

X_train_textp = pipeline_text.fit_transform(X_train_text, y_train).toarray()
# X_train_nontextp = pipeline_nontext.fit_transform(X_train_nontext, y_train)
# X_all = np.hstack([X_train_textp, X_train_nontextp])

pipeline_model.fit(X_train_textp, y_train)


# import joblib
from joblib import dump

# dump the pipeline model
dump(pipeline_text, filename="pipeline_text.joblib")
# dump(pipeline_nontext, filename="pipeline_nontext.joblib")
dump(pipeline_model, filename="pipeline_model.joblib")



#%% load model

# import joblib
from joblib import load

test = '@KatieKatCubs you already know how this shit goes. World Series or Armageddon.'

X_test = pd.DataFrame([test], columns = ['text'])

X_test_text = text_preprocess(X_test)

X_test_textp = pipeline_text.transform(X_test_text).toarray()

pipeline_model.predict(X_test_textp)


"""

