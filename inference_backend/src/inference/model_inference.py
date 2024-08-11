
import os

import joblib
import numpy as np

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
TARGET_COL = 'target'

DATA_OUTPUT_DIR = './data_train'
MODEL_ASSETS_DIR = 'model_assets'
MODEL_SUBDIR = 'model'
MODEL_FILENAME = 'model.joblib'


def model_inference(X_test_vect_added_scaled,
                    model: object = None,
                    model_assets_dir: str = MODEL_ASSETS_DIR,
                    data_output_dir: str = None,
                    output_filename: str = 'pred.npy',
                    **kwargs):

    model_dir = os.path.join(model_assets_dir, MODEL_SUBDIR)

    if model is None:
        model = joblib.load(os.path.join(model_dir, MODEL_FILENAME))

    y_pred = model.predict(X_test_vect_added_scaled)

    if data_output_dir not in [None, '']:
        os.makedirs(data_output_dir, exist_ok=True)
        np.save(os.path.join(data_output_dir, output_filename),
                y_pred)

    return y_pred
