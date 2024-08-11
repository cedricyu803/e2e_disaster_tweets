
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

MODEL_ASSETS_DIR = 'model_assets'
SCALER_FILENAME: str = 'scaler.joblib'

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
TARGET_COL = 'target'


scaler_mapping = {'minmax_scaler': MinMaxScaler, }


def fit_scaler(
        X_train: pd.DataFrame,
        model_assets_dir: str = MODEL_ASSETS_DIR,
        scaler_name: str = 'minmax_scaler',
        scaler_filename: str = SCALER_FILENAME,):

    os.makedirs(model_assets_dir, exist_ok=True)
    scaler_dir = os.path.join(model_assets_dir, 'scaler')
    os.makedirs(scaler_dir, exist_ok=True)

    scaler = (scaler_mapping[scaler_name]().fit(X_train))

    if scaler_dir not in [None, '']:
        os.makedirs(scaler_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(
            scaler_dir, scaler_filename))

    return scaler, scaler_dir


def transform_scales(
        X: pd.DataFrame,
        data_output_dir: str = None,
        output_filename: str = 'X_scaled.npy',
        scaler: object = None,
        scaler_dir: str = None,
        scaler_filename: str = SCALER_FILENAME,):
    if scaler is None:
        scaler = joblib.load(os.path.join(
            scaler_dir, scaler_filename))

    X_scaled = scaler.transform(X)

    if data_output_dir not in [None, '']:
        os.makedirs(data_output_dir, exist_ok=True)
        np.save(os.path.join(data_output_dir, output_filename),
                X_scaled)

    return X_scaled
