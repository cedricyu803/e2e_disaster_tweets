# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:00:00 2021
@author: Cedric Yu
This file:
text pre-processing
"""
import argparse
import os

import yaml

from src.model_train.feature_extract import run_feature_extraction
from src.model_train.model_train import train_model

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

TRAIN_PATHS = [os.path.join(DATA_OUTPUT_DIR, 'X_train_vect_added_scaled.npy'),
               os.path.join(DATA_OUTPUT_DIR, 'y_train.npy')]
VALID_PATHS = [os.path.join(DATA_OUTPUT_DIR, 'X_valid_vect_added_scaled.npy'),
               os.path.join(DATA_OUTPUT_DIR, 'y_valid.npy')]


def get_arguments():

    parser = argparse.ArgumentParser(
        description='Arguments')

    parser.add_argument(
        '--config_path',
        '-c',
        action='store',
        dest='config_path',
        help='config_path',
        default='./train_config.yml',
        required=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # load arguments
    args = get_arguments()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    feature_config = config['features']
    model_config = config['model']
    if 'ngram_range' in feature_config['vectoriser_args']:
        feature_config['vectoriser_args']['ngram_range'] = \
            tuple(feature_config['vectoriser_args']['ngram_range'])

    (X_train_vect_added_scaled, y_train,
     X_valid_vect_added_scaled, y_valid,) = \
        run_feature_extraction(**feature_config)

    train_model, eval_scores = train_model(
        X_train_vect_added_scaled=X_train_vect_added_scaled,
        y_train=y_train,
        X_valid_vect_added_scaled=X_valid_vect_added_scaled,
        y_valid=y_valid,
        **model_config)
