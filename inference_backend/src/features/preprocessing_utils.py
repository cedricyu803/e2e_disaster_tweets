# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:00:00 2021
@author: Cedric Yu
This file:
text pre-processing
"""
import os
import re
import warnings

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None



# %% pre-processing

"""
text pre-processing.
# we loosely follow
# https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""


def process_html_syntax(text: str):
    '''Uses BeautifulSoup to remove HTML syntax (mojibake) from a text string

    Args:
        text (str):

    Returns:
        text_no_mojibake (str): processed string from text
    '''
    text_no_mojibake = (BeautifulSoup(text, features='html.parser')
                        .get_text().strip())
    return text_no_mojibake


def expand_abbrv(text: str):
    '''Expands abbreviation n't to not and 's to  s

    Args:
        text (str):

    Returns:
        text_processed (str):
    '''
    # expand abbreviation n't to not and 's to  s
    text_processed = re.sub(r"n't", ' not', text)
    text_processed = re.sub(r"'s", ' s', text_processed)
    return text_processed


def tokenise_url_mention(text: str):
    '''Replaces url by '<url>' token, and @mention by '<user>' token

    Args:
        text (str):

    Returns:
        text_processed (str):
    '''
    text_processed = re.sub(r'(https?://[\S]*)', '<url>', text)
    text_processed = re.sub(r'@\w+', "<user>", text_processed)
    return text_processed


def process_hashtag(text: str):
    '''Replaces # by <hashtag> token, and
    split the hashtag_body by capital letters unless it is all cap

    Args:
        text (str):

    Returns:
        text_processed (str):
    '''
    text_processed = text
    hash_iter = list(filter(None, re.findall(r'#(\w+)', text)))
    if len(hash_iter) > 0:
        for item in hash_iter:
            if item.isupper():
                hash_words = item + ' <allcaps>'
            else:
                hash_words = ' '.join(
                    list(filter(None, re.split(r'([A-Z]?[a-z]*)', item))))
            text_processed = re.sub(item, hash_words, text_processed)
            text_processed = re.sub(r'#', '<hashtag> ', text_processed)
    return text_processed


def tokenise_num_punc(text: str):
    '''Represents numbers by '<number>' token,
    replaces repeated punctuation marks (!?. only) by
    punctuation mark + <repeat> token,
    adds spaces before and after (!?.),
    represents punctuation marks (except ?!.) by <punc> token.
    Also remove extra whitespaces.

    Args:
        text (str):

    Returns:
        text_processed (str):
    '''
    # represent numbers by '<number>' token
    text_processed = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' <number> ', text)

    # replace repeated punctuation marks (!?. only) by
    # punctuation mark + <repeat> token
    text_processed = re.sub(r'([!?\.]){2,}', r'\1' + ' <repeat>',
                            text_processed)
    # add spaces before and after (!?.)
    text_processed = re.sub(r'([!?\.])', r' \1 ', text_processed)

    # replace punctuation marks (except ?!.) by <punc> token
    text_processed = re.sub(
        r'[\"#$%\&\'\(\)\*\+,\-/:;=@\[\]\^_`\{\|\}~\\]+',
        ' <punc> ', text_processed)
    text_processed = re.sub(r'(<[^A-Za-z0-9_-]+|[^A-Za-z0-9_-]+>)',
                            ' <punc> ', text_processed)

    # remove extra whitespaces
    text_processed = text_processed.strip()
    text_processed = re.sub(r'\s+', ' ', text_processed)
    return text_processed


def nltk_lemmatize(text: str):
    '''Lemmatises text.

    Args:
        text (str):

    Returns:
        text_processed (str):
    '''
    WNlemma_n = nltk.WordNetLemmatizer()
    return WNlemma_n.lemmatize(text)


def preprocess_text(text: str):
    '''Processes text str:
    1. Process HTML syntax
    2. Expand abbreviations
    3. tokenise URL and @mention
    4. Process hashtag
    5. Tokenise numbers and punctuations
    6. remove extra whitespaces

    Args:
        text (str): _description_

    Returns:
        text_no_mojibake (str): HTML syntax-processed text
        text_processed (str): processed text
    '''
    text_no_mojibake = process_html_syntax(text)
    text_processed = expand_abbrv(text_no_mojibake)
    text_processed = tokenise_url_mention(text_processed)
    text_processed = process_hashtag(text_processed)
    text_processed = tokenise_num_punc(text_processed)

    return text_no_mojibake, text_processed


def df_preprocess_text(df: pd.DataFrame,
                       text_col: str = 'text',
                       text_processed_col: str = 'text_processed',
                       data_output_dir: str = None,
                       output_filename: str = 'X_processed.npy'):
    '''Processes text in the text_col of df.

    Args:
        df (pd.DataFrame):
        text_col (str, optional): Defaults to 'text'.
        text_processed_col (str, optional): Defaults to 'text_processed'.
        data_output_dir (str, optional): Defaults to None.
        output_filename (str, optional): Defaults to 'X_processed.npy'.
    Returns:
        df (pd.DataFrame): df with new columns:
            'text_no_mojibake' (HTML syntax-processed), 'text_processed'
    '''
    def row_text_preprocess_(row):
        row['text_no_mojibake'], row[text_processed_col] = \
            preprocess_text(row['text'])
        return row
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df = df.apply(row_text_preprocess_, axis=1)

    if data_output_dir not in [None, '']:
        os.makedirs(data_output_dir, exist_ok=True)
        text_preprocessed_np = df[[text_processed_col]].to_numpy()
        np.save(os.path.join(data_output_dir, output_filename),
                text_preprocessed_np)
    return df
