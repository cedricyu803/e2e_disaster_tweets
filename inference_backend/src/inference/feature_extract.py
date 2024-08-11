# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:00:00 2021
@author: Cedric Yu
This file:
text pre-processing
"""
import os

import numpy as np
import pandas as pd
from src.features.feature_encoding_utils import ENCODER_SUBDIR, transform_features
from src.features.feature_engineering_utils import df_get_features
from src.features.preprocessing_utils import df_preprocess_text
from src.features.scaler_utils import SCALER_SUBDIR, transform_scales
from src.features.vectorisers_utils import VECTORISER_SUBDIR, transform_text_vec

# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
TEXT_COL = 'text'
TARGET_COL = 'target'

MODEL_ASSETS_DIR = 'model_assets'


def run_feature_extraction(data,
                           model_assets_dir: str = MODEL_ASSETS_DIR,
                           data_output_dir: str = None,):

    os.makedirs(model_assets_dir, exist_ok=True)
    if data_output_dir not in [None, '']:
        os.makedirs(data_output_dir, exist_ok=True)

    # load data
    if isinstance(data, str):
        X_test = pd.DataFrame([data], columns=['text'])
        X_test[['id', 'keyword']] = np.nan
    elif isinstance(data, list):
        X_test = pd.DataFrame(data, columns=['text'])
        X_test[['id', 'keyword']] = np.nan
    elif isinstance(data, pd.Series):
        data = data.to_frame()
    assert 'text' in X_test

    # preprocess text
    X_test_text_preprocessed = df_preprocess_text(
        X_test,
        data_output_dir=data_output_dir,
        output_filename='X_test_text_preprocessed.npy')

    # feature engineering
    X_test_no_text = df_get_features(
        X_test_text_preprocessed,
        model_assets_dir=model_assets_dir,
        data_output_dir=data_output_dir,
        output_filename='X_test_no_text.npy')

    # feature encoding
    encoder_dir = os.path.join(model_assets_dir, ENCODER_SUBDIR)
    X_test_no_text_encoded = transform_features(
        X_test_no_text,
        encoder_dir=encoder_dir,
        data_output_dir=data_output_dir,
        output_filename='X_test_no_text_encoded.npy')

    # text vectoriser
    vectoriser_dir = os.path.join(model_assets_dir, VECTORISER_SUBDIR)
    X_test_text_vect = transform_text_vec(
        X_test_text_preprocessed,
        vectoriser_dir=vectoriser_dir,
        data_output_dir=data_output_dir,
        output_filename='X_test_text_vect.npy')

    # stack features togther
    X_test_vect_added = np.hstack(
        [X_test_text_vect, X_test_no_text_encoded])

    # scaler
    scaler_dir = os.path.join(model_assets_dir, SCALER_SUBDIR)
    X_test_vect_added_scaled = transform_scales(
        X_test_vect_added,
        scaler_dir=scaler_dir,
        data_output_dir=data_output_dir,
        output_filename='X_test_vect_added_scaled.npy')

    return X_test_vect_added_scaled
