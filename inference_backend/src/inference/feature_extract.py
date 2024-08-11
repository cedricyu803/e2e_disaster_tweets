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
from sklearn.model_selection import train_test_split
from src.features.feature_encoding_utils import fit_encoders, transform_features
from src.features.feature_engineering_utils import df_get_features
from src.features.preprocessing_utils import df_preprocess_text
from src.features.scaler_utils import fit_scaler, transform_scales
from src.features.vectorisers_utils import fit_vectoriser, transform_text_vec

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
    # load data
    if isinstance(data, str):
        X_test = pd.DataFrame([data], columns = ['text'])
    elif isinstance(data, pd.Series):
        X_test = X_test['text']
    # X_test = pd.read_csv(os.path.join(
    #     data_dir, TEST_FILENAME), header=[0])
    X_train, X_valid, y_train, y_valid = train_test_split(
        train.drop([TARGET_COL], axis=1), train[TARGET_COL],
        random_state=RANDOM_SEED, train_size=train_size)

    X_train = X_train.iloc[:20]
    y_train = y_train.iloc[:20]
    X_valid = X_valid.iloc[:20]
    y_valid = y_valid.iloc[:20]
    # # X_test = X_test.iloc[:20]

    if data_output_dir not in [None, '']:
        os.makedirs(data_output_dir, exist_ok=True)
        np.save(os.path.join(data_output_dir, 'y_train.npy'),
                y_train.to_numpy())
        np.save(os.path.join(data_output_dir, 'y_valid.npy'),
                y_valid.to_numpy())

    # preprocess text
    X_train_text_preprocessed = df_preprocess_text(
        X_train,
        data_output_dir=data_output_dir,
        output_filename='X_train_text_preprocessed.npy')
    X_valid_text_preprocessed = df_preprocess_text(
        X_valid,
        data_output_dir=data_output_dir,
        output_filename='X_valid_text_preprocessed.npy')
    # X_test_text_preprocessed = df_preprocess_text(
    #     X_test,
    #     data_output_dir=data_output_dir,
    #     output_filename='X_test_text_preprocessed.npy')

    # feature engineering
    X_train_no_text = df_get_features(
        X_train_text_preprocessed,
        data_output_dir=data_output_dir,
        output_filename='X_train_no_text.npy')
    X_valid_no_text = df_get_features(
        X_valid_text_preprocessed,
        data_output_dir=data_output_dir,
        output_filename='X_valid_no_text.npy')
    # X_test_no_text = df_get_features(
    #     X_test_text_preprocessed,
    #     data_output_dir=data_output_dir,
    #     output_filename='X_test_no_text.npy')

    # feature encoding
    freq_encoder, target_mean_encoder, encoder_dir = fit_encoders(
        X_train_no_text, y_train,
        model_assets_dir=model_assets_dir)
    X_train_no_text_encoded = transform_features(
        X_train_no_text,
        encoder_dir=encoder_dir,
        data_output_dir=data_output_dir,
        output_filename='X_train_no_text_encoded.npy',
        freq_encoder=freq_encoder, target_mean_encoder=target_mean_encoder)
    X_valid_no_text_encoded = transform_features(
        X_valid_no_text,
        encoder_dir=encoder_dir,
        data_output_dir=data_output_dir,
        output_filename='X_valid_no_text_encoded.npy',
        freq_encoder=freq_encoder, target_mean_encoder=target_mean_encoder)
    # X_test_no_text_encoded = transform_features(
    #     X_test_no_text,
    #     encoder_dir=encoder_dir,
    #     data_output_dir=data_output_dir,
    #     output_filename='X_test_no_text_encoded.npy',
    #     freq_encoder=freq_encoder, target_mean_encoder=target_mean_encoder)

    # text vectoriser
    vectoriser, vectoriser_dir = fit_vectoriser(
        X_train_text_preprocessed,
        model_assets_dir=model_assets_dir,
        vectoriser_name=vectoriser_name,
        vectoriser_args=vectoriser_args)
    X_train_text_vect = transform_text_vec(
        X_train_text_preprocessed,
        vectoriser_dir=vectoriser_dir,
        data_output_dir=data_output_dir,
        output_filename='X_train_text_vect.npy',
        vectoriser=vectoriser)
    X_valid_text_vect = transform_text_vec(
        X_valid_text_preprocessed,
        vectoriser_dir=vectoriser_dir,
        data_output_dir=data_output_dir,
        output_filename='X_valid_text_vect.npy',
        vectoriser=vectoriser)
    # X_test_text_vect = transform_text_vec(
    #     X_test_text_preprocessed,
    #     vectoriser_dir=vectoriser_dir,
    #     data_output_dir=data_output_dir,
    #     output_filename='X_test_text_vect.npy',
    #     vectoriser=vectoriser)

    # stack features togther
    X_train_vect_added = np.hstack(
        [X_train_text_vect, X_train_no_text_encoded])
    X_valid_vect_added = np.hstack(
        [X_valid_text_vect, X_valid_no_text_encoded])
    # X_test_vect_added = np.hstack(
    #     [X_test_text_vect, X_test_no_text_encoded])

    # scaler
    scaler, scaler_dir = fit_scaler(
        X_train_vect_added, scaler_name=scaler_name,
        model_assets_dir=model_assets_dir,)
    X_train_vect_added_scaled = transform_scales(
        X_train_vect_added,
        scaler_dir=scaler_dir,
        data_output_dir=data_output_dir,
        output_filename='X_train_vect_added_scaled.npy',
        scaler=scaler)
    X_valid_vect_added_scaled = transform_scales(
        X_valid_vect_added,
        scaler_dir=scaler_dir,
        data_output_dir=data_output_dir,
        output_filename='X_valid_vect_added_scaled.npy',
        scaler=scaler)
    # X_test_vect_added_scaled = transform_scales(
    #     X_test_vect_added,
    #     scaler_dir=scaler_dir,
    #     data_output_dir=data_output_dir,
    #     output_filename='X_test_vect_added_scaled.npy',
    #     scaler=scaler)

    return (X_train_vect_added_scaled, y_train,
            X_valid_vect_added_scaled, y_valid,
            # X_test_vect_added_scaled
            )
