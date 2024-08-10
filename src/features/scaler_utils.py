
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

MODEL_ASSETS_DIR = 'model_assets'
SCALER_DIR = os.path.join(MODEL_ASSETS_DIR, 'scaler')
os.makedirs(SCALER_DIR, exist_ok=True)
SCALER_FILENAME: str = 'scaler.joblib'

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
TARGET_COL = 'target'


scaler_mapping = {'minmax_scaler': MinMaxScaler, }


scaler_tfidf = MinMaxScaler()


X_train_tfidf_added_scaled = scaler_tfidf.transform(X_train_tfidf_added)
X_valid_tfidf_added_scaled = scaler_tfidf.transform(X_valid_tfidf_added)
X_test_tfidf_added_scaled = scaler_tfidf.transform(X_test_tfidf_added)


def fit_scaler(
        X_train: pd.DataFrame,
        scaler_name: str = 'minmax_scaler',
        output_dir: str = SCALER_DIR,
        scaler_filename: str = SCALER_FILENAME,):

    scaler = (scaler_mapping[scaler_name]().fit(X_train))

    if output_dir not in [None, '']:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(
            output_dir, scaler_filename))

    return scaler, output_dir


def transform_scales(
        X: pd.DataFrame,
        output_dir: str = None,
        output_filename: str = 'X_scaled.npy',
        scaler: object = None,
        scaler_dir: str = SCALER_DIR,
        scaler_filename: str = SCALER_FILENAME,):
    if scaler is None:
        scaler = joblib.load(os.path.join(
            scaler_dir, scaler_filename))

    X_scaled = scaler.transform(X)

    if output_dir not in [None, '']:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, output_filename),
                X_scaled)

    return X_scaled
