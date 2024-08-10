
import os

import category_encoders as ce
import joblib
import numpy as np
import pandas as pd

MODEL_ASSETS_DIR = 'model_assets'
ENCODER_DIR = os.path.join(MODEL_ASSETS_DIR, 'encoders')
os.makedirs(ENCODER_DIR, exist_ok=True)
FREQ_ENCODER_FILENAME: str = 'freq_encoder.joblib'
TARGET_MEAN_ENCODER_FILENAME: str = 'target_mean_encoder.joblib'

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
TARGET_COL = 'target'
HASHMENTION_COLS = ['hashtags', 'at']
NO_TEXT_COLS = ['id', 'keyword', 'char_count', 'punc_ratio', 'cap_ratio',
                'sentence_count', 'stopword_num', 'hashtag_num', 'at_num',
                'url_num', 'country_mention_num', 'token_count']
FREQ_COLS = ['sentence_count', 'stopword_num', 'hashtag_num',
             'at_num', 'url_num', 'country_mention_num', 'token_count']
TARGET_MEAN_COLS = ['keyword']
COLS_TO_KEEP = ['char_count', 'punc_ratio', 'cap_ratio',
                'sentence_count_freq_encoded', 'stopword_num_freq_encoded',
                'hashtag_num_freq_encoded', 'at_num_freq_encoded',
                'url_num_freq_encoded', 'country_mention_num_freq_encoded',
                'token_count_freq_encoded', 'keyword_mean_encoded']


def fit_encoders(
        X_train: pd.DataFrame, y_train: pd.Series,
        no_text_cols: list = NO_TEXT_COLS,
        hashmention_cols: list = HASHMENTION_COLS,
        freq_cols: list = FREQ_COLS,
        target_mean_cols: list = TARGET_MEAN_COLS,
        output_dir: str = ENCODER_DIR,
        freq_encoder_filename: str = FREQ_ENCODER_FILENAME,
        target_mean_encoder_filename: str = TARGET_MEAN_ENCODER_FILENAME):

    # X_train_hashmention = X_train[hashmention_cols]
    X_train_no_text = X_train[no_text_cols]

    # fit frequency encoder
    # change data type to object before feeding it into the encoder
    freq_encoder = ce.count.CountEncoder()
    freq_encoder.fit(X_train_no_text[freq_cols].astype(object))

    # fit target mean encoder
    # change data type to object before feeding it into the encoder
    target_mean_encoder = ce.target_encoder.TargetEncoder()
    target_mean_encoder.fit(
        X_train_no_text[target_mean_cols].astype(object), y_train)

    if output_dir not in [None, '']:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(freq_encoder, os.path.join(
            output_dir, freq_encoder_filename))
        joblib.dump(target_mean_encoder, os.path.join(
            output_dir, target_mean_encoder_filename))

    return freq_encoder, target_mean_encoder, output_dir


def transform_features(
        X: pd.DataFrame,
        output_dir: str = None,
        output_filename: str = 'X_no_text_encoded.npy',
        no_text_cols: list = NO_TEXT_COLS,
        hashmention_cols: list = HASHMENTION_COLS,
        freq_cols: list = FREQ_COLS,
        target_mean_cols: list = TARGET_MEAN_COLS,
        cols_to_keep: list = COLS_TO_KEEP,
        freq_encoder: object = None, target_mean_encoder: object = None,
        encoder_dir: str = ENCODER_DIR,
        freq_encoder_filename: str = FREQ_ENCODER_FILENAME,
        target_mean_encoder_filename: str = TARGET_MEAN_ENCODER_FILENAME):
    if freq_encoder is None:
        freq_encoder = joblib.load(os.path.join(
            encoder_dir, freq_encoder_filename))
    if target_mean_encoder is None:
        target_mean_encoder = joblib.load(os.path.join(
            encoder_dir, target_mean_encoder_filename))

    # X_train_hashmention = X_train[hashmention_cols]
    X_no_text = X[no_text_cols]
    # change data type to object before feeding it into the encoder
    X_no_text[freq_cols] = X_no_text[freq_cols].astype(object)
    X_no_text[target_mean_cols] = \
        X_no_text[target_mean_cols].astype(object)

    X_no_text_encoded = (pd.concat(
        [X_no_text,
         freq_encoder.transform(X_no_text[freq_cols])
         .rename(columns=dict(zip(
             freq_cols, [col + '_freq_encoded' for col in freq_cols])))],
        axis=1).drop(freq_cols, axis=1))

    X_no_text_encoded = (pd.concat(
        [X_no_text_encoded,
         target_mean_encoder.transform(
             X_no_text_encoded[target_mean_cols])
         .rename(columns=dict(zip(
             target_mean_cols, [col + '_mean_encoded'
                                for col in target_mean_cols])))], axis=1)
                               .drop(target_mean_cols, axis=1))

    X_no_text_encoded = X_no_text_encoded[cols_to_keep].fillna(
        X_no_text_encoded.mean())

    if output_dir not in [None, '']:
        os.makedirs(output_dir, exist_ok=True)
        text_output_np = X_no_text_encoded.to_numpy()
        np.save(os.path.join(output_dir, output_filename),
                text_output_np)

    return X_no_text_encoded
