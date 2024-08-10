# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:00:00 2021
@author: Cedric Yu
This file:
text pre-processing
"""
import argparse
import os
import warnings

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

DATA_DIR = './data'
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
DATA_OUTPUT_DIR = './data_train'
MODEL_ASSETS_DIR = 'model_assets'
os.makedirs(MODEL_ASSETS_DIR, exist_ok=True)


def get_arguments():

    parser = argparse.ArgumentParser(
        description='Arguments')

    parser.add_argument(
        '--data-dir',
        '-d',
        action='store',
        dest='data_dir',
        help='data dir',
        default=DATA_DIR,
        required=False)

    parser.add_argument(
        '--train-size',
        '-t',
        action='store',
        dest='train_size',
        help='train size',
        type=float,
        default=0.8,
        required=False)

    args = parser.parse_args()

    return args


def run_feature_extraction(data_dir: str = DATA_DIR,
                           train_size: float = 0.8):

    # load data
    train = pd.read_csv(os.path.join(data_dir, TRAIN_FILENAME), header=[0])[
        [TEXT_COL, TARGET_COL]]
    X_test = pd.read_csv(os.path.join(
        data_dir, TEST_FILENAME), header=[0])[[TEXT_COL]]
    X_train, X_valid, y_train, y_valid = train_test_split(
        train.drop([TARGET_COL], axis=1), train[TARGET_COL],
        random_state=RANDOM_SEED, train_size=train_size)

    X_train_text_processed = df_preprocess_text(X_train)
    X_valid_text_processed = df_preprocess_text(X_valid)
    X_test_text_processed = df_preprocess_text(X_test)


if __name__ == "__main__":
    # load arguments
    args = get_arguments()
    data_dir = args.data_dir
    train_size = args.train_size

