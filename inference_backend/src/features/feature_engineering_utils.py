# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:00:00 2021
@author: Cedric Yu
This file:
preprocessed text feature extraction
"""


import os
import re

import nltk
import numpy as np
import pandas as pd
from geotext import GeoText
from nltk.corpus import stopwords

pd.set_option('future.no_silent_downcasting', True)


NO_TEXT_COLS = ['id', 'keyword', 'char_count', 'punc_ratio', 'cap_ratio',
                'sentence_count', 'stopword_num', 'hashtag_num', 'at_num',
                'url_num', 'country_mention_num', 'token_count']


# download required nltk data
MODEL_ASSETS_DIR = 'model_assets'


def get_features(row,
                 stops: list,
                 ):
    '''Extracts features from preprocessed df containing
    HTML syntax-processed text (text_no_mojibake) and
    processed text (text_processed).
    1. Character count
    2. punctuation mark and capital letter to character length ratios
    3. number of sentences, stopwords
    4. list of hashtags, mentions
    5. URLs
    6. country mentions
    7. token count

    Args:
        row (pd.Series):
        stops (list): nltk's list of stopwords

    Returns:
        row (pd.Series): with extracted features
    '''

    # character count
    row['char_count'] = len(row['text_no_mojibake'])

    # punctuation mark and capital letter to character length ratios
    row['punc_ratio'] = len(''.join(re.findall(
        r'[\.\?!\"#$%\&\'\(\)\*\+,\-/:;=@\[\]\^_`\{\|\}~\\]+',
        row['text_no_mojibake']))) / len(row['text_no_mojibake'])
    row['cap_ratio'] = len(''.join(re.findall(
        r'[A-Z]', row['text_no_mojibake']))) / len(row['text_no_mojibake'])

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
    if len(list(zip(*GeoText(row['text_no_mojibake'])
                    .country_mentions.items()))) > 0:
        row['country_mention_num'] = np.array(
            list(zip(*GeoText(row['text_no_mojibake'])
                     .country_mentions.items()))[1]).sum()
    else:
        row['country_mention_num'] = 0

    # token count
    row['token_count'] = len(nltk.word_tokenize(row['text_processed']))

    return row


def df_get_features(df: pd.DataFrame,
                    output_cols: list = NO_TEXT_COLS,
                    model_assets_dir: str = MODEL_ASSETS_DIR,
                    data_output_dir: str = None,
                    output_filename: str = 'X_no_text.npy'):
    '''Extracts features from text-preprocessed df.

    Args:
        df (pd.DataFrame): text-preprocessed
        output_cols (list, optional): Defaults to NO_TEXT_COLS.
        model_assets_dir (str, optional): contains nltk's data.
            Defaults to MODEL_ASSETS_DIR.
        data_output_dir (str, optional): Defaults to None.
        output_filename (str, optional): Defaults to 'pred.npy'.

    Returns:
        df (pd.DataFrame): df with new columns for extracted features
    '''

    os.makedirs(model_assets_dir, exist_ok=True)
    nltk.data.path.append(model_assets_dir)
    stops = set(stopwords.words('english'))

    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df = df.apply(lambda x: get_features(x, stops=stops), axis=1)
    df = df[[col for col in output_cols if col in df]]

    if data_output_dir not in [None, '']:
        os.makedirs(data_output_dir, exist_ok=True)
        text_output_np = df.to_numpy()
        np.save(os.path.join(data_output_dir, output_filename),
                text_output_np)

    return df
