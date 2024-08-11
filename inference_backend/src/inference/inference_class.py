# -*- coding: utf-8 -*-
import os

import joblib
import numpy as np
import pandas as pd
from src.features.feature_encoding_utils import (
    ENCODER_SUBDIR,
    FREQ_ENCODER_FILENAME,
    TARGET_MEAN_ENCODER_FILENAME,
    transform_features,
)
from src.features.feature_engineering_utils import df_get_features
from src.features.preprocessing_utils import df_preprocess_text
from src.features.scaler_utils import SCALER_FILENAME, SCALER_SUBDIR, transform_scales
from src.features.vectorisers_utils import (
    VECTORISER_FILENAME,
    VECTORISER_SUBDIR,
    transform_text_vec,
)
from src.inference.model_inference import MODEL_FILENAME, MODEL_SUBDIR, model_inference

# default='warn' # ignores warning about dropping columns inplace
pd.options.mode.chained_assignment = None

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
TEXT_COL = 'text'
TARGET_COL = 'target'

MODEL_ASSETS_DIR = 'model_assets'


class Inference():
    '''Class for loading fitted objects/model, and run inference
    '''
    def __init__(self,
                 model_assets_dir: str = MODEL_ASSETS_DIR,
                 data_output_dir: str = None,
                 ):
        '''Loads fitted objects/model

        Args:
            model_assets_dir (str, optional): _description_. Defaults to MODEL_ASSETS_DIR.
            data_output_dir (str, optional): _description_. Defaults to None.
        '''
        self.status = -1
        self.model_assets_dir = model_assets_dir
        self.data_output_dir = data_output_dir
        assert os.path.exists(self.model_assets_dir)
        if self.data_output_dir not in [None, '']:
            os.makedirs(self.data_output_dir, exist_ok=True)

        # load fitted objects
        self.freq_encoder = None
        self.target_mean_encoder = None
        self.vectoriser = None
        self.scaler = None
        self.model = None
        self.load_assets()

    def load_assets(self,):
        '''Loads fitted objects/model
        '''
        if self.freq_encoder is None:
            self.freq_encoder = joblib.load(os.path.join(
                self.model_assets_dir, ENCODER_SUBDIR, FREQ_ENCODER_FILENAME))
        if self.target_mean_encoder is None:
            self.target_mean_encoder = joblib.load(os.path.join(
                self.model_assets_dir, ENCODER_SUBDIR,
                TARGET_MEAN_ENCODER_FILENAME))

        if self.vectoriser is None:
            self.vectoriser = joblib.load(os.path.join(
                self.model_assets_dir, VECTORISER_SUBDIR, VECTORISER_FILENAME))

        if self.scaler is None:
            self.scaler = joblib.load(os.path.join(
                self.model_assets_dir, SCALER_SUBDIR, SCALER_FILENAME))

        if self.model is None:
            self.model = joblib.load(os.path.join(
                self.model_assets_dir, MODEL_SUBDIR, MODEL_FILENAME))

        self.status = 0

    def parse_data(self, data):
        '''Parses input data for inference

        Args:
            data: Either str, list of str, pd.Series, or
                pd.DataFrame with 'text' columns

        Returns:
            X_test (pd.DataFrame):
        '''
        if isinstance(data, str):
            X_test = pd.DataFrame([data], columns=['text'])
            X_test[['id', 'keyword']] = np.nan
        elif isinstance(data, list):
            X_test = pd.DataFrame(data, columns=['text'])
            X_test[['id', 'keyword']] = np.nan
        elif isinstance(data, pd.Series):
            X_test = data.to_frame()
        assert 'text' in X_test
        return X_test

    def extract_features(self, X_test: pd.DataFrame):
        '''Extracts features from parsed input data

        Args:
            X_test (pd.DataFrame): parsed input data

        Returns:
            X_test_vect_added_scaled: features
        '''
        data_output_dir = self.data_output_dir

        # preprocess text
        X_test_text_preprocessed = df_preprocess_text(
            X_test,
            data_output_dir=data_output_dir,
            output_filename='X_test_text_preprocessed.npy')

        # feature engineering
        X_test_no_text = df_get_features(
            X_test_text_preprocessed,
            model_assets_dir=self.model_assets_dir,
            data_output_dir=data_output_dir,
            output_filename='X_test_no_text.npy')

        # feature encoding
        X_test_no_text_encoded = transform_features(
            X_test_no_text,
            freq_encoder=self.freq_encoder,
            target_mean_encoder=self.target_mean_encoder,
            data_output_dir=data_output_dir,
            output_filename='X_test_no_text_encoded.npy')

        # text vectoriser
        X_test_text_vect = transform_text_vec(
            X_test_text_preprocessed,
            vectoriser=self.vectoriser,
            data_output_dir=data_output_dir,
            output_filename='X_test_text_vect.npy')

        # stack features togther
        X_test_vect_added = np.hstack(
            [X_test_text_vect, X_test_no_text_encoded])

        # scaler
        X_test_vect_added_scaled = transform_scales(
            X_test_vect_added,
            scaler=self.scaler,
            data_output_dir=data_output_dir,
            output_filename='X_test_vect_added_scaled.npy')

        return X_test_vect_added_scaled

    def run_inference(self, data):
        '''Runs inference on input data

        Args:
            data: Either str, list of str, pd.Series, or
                pd.DataFrame with 'text' columns

        Returns:
            y_pred (np.ndarray):
                predictions are 0 (not disaster) or 1 (disaster)
        '''
        X_test = self.parse_data(data)
        X_test_vect_added_scaled = self.extract_features(X_test)
        y_pred = model_inference(
            X_test_vect_added_scaled=X_test_vect_added_scaled,
            model=self.model, data_output_dir=self.data_output_dir)
        return y_pred
