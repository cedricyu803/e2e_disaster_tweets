
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

MODEL_ASSETS_DIR = 'model_assets'
VECTORISER_DIR = os.path.join(MODEL_ASSETS_DIR, 'vectoriser')
os.makedirs(VECTORISER_DIR, exist_ok=True)
VECTORISER_FILENAME: str = 'vectoriser.joblib'

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
TARGET_COL = 'target'


vectoriser_mapping = {'count_vectoriser': CountVectorizer,
                      'tfidf_vectoriser': TfidfVectorizer}


def fit_vectoriser(
        X_train: pd.DataFrame,
        text_processed_col: str = 'text_processed',
        vectoriser_name: str = 'tfidf_vectoriser',
        vectoriser_args: dict = {'min_df': 3,
                                 'max_df': 0.3,
                                 'stop_words': 'english',
                                 'ngram_range': (1, 3)},
        output_dir: str = VECTORISER_DIR,
        vectoriser_filename: str = VECTORISER_FILENAME,):

    vectoriser = (vectoriser_mapping[vectoriser_name](**vectoriser_args)
                  .fit(X_train[[text_processed_col]]
                       .to_numpy().squeeze().tolist()))

    if output_dir not in [None, '']:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(vectoriser, os.path.join(
            output_dir, vectoriser_filename))

    return vectoriser, output_dir


def transform_text_vec(
        X: pd.DataFrame,
        text_processed_col: str = 'text_processed',
        output_dir: str = None,
        output_filename: str = 'X_text_vect.npy',
        vectoriser: object = None,
        vectoriser_dir: str = VECTORISER_DIR,
        vectoriser_filename: str = VECTORISER_FILENAME,):
    if vectoriser is None:
        vectoriser = joblib.load(os.path.join(
            vectoriser_dir, vectoriser_filename))

    X_train_text_vect = vectoriser.transform(
        X[[text_processed_col]][[text_processed_col]]
        .to_numpy().squeeze().tolist()).toarray()

    if output_dir not in [None, '']:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, output_filename),
                X_train_text_vect)

    return X_train_text_vect
