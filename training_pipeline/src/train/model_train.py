import json
import os

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, roc_auc_score
from src.models.model_mapping import model_map

RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
TARGET_COL = 'target'

DATA_OUTPUT_DIR = './data_train'
MODEL_ASSETS_DIR = 'model_assets'
MODEL_SUBDIR = 'model'
MODEL_FILENAME = 'model.joblib'


def train_model(X_train_vect_added_scaled, y_train,
                X_valid_vect_added_scaled, y_valid,
                model_assets_dir: str = MODEL_ASSETS_DIR,
                model_name: str = 'lr',
                model_params: dict = {'random_state': RANDOM_SEED},
                metrics=[f1_score, roc_auc_score],
                model_filename: str = MODEL_FILENAME,
                **kwargs
                ):
    '''Fits model on transformed train set, evaluates model on test set.
    Outputs fitted model and evaluation scores

    Args:
        X_train_vect_added_scaled: transformed train set
        y_train:
        X_valid_vect_added_scaled: transformed test set
        y_valid:
        model_assets_dir (str, optional):
            output dir of fitted model and evaluation metrics.
            Defaults to MODEL_ASSETS_DIR.
        model_name (str, optional): Defaults to 'lr'.
        model_params (dict, optional):
            Defaults to {'random_state': RANDOM_SEED}.
        metrics (list, optional): Defaults to [f1_score, roc_auc_score].
        model_filename (str, optional): Defaults to MODEL_FILENAME.

    Returns:
        model, eval_scores
    '''

    os.makedirs(model_assets_dir, exist_ok=True)
    model_dir = os.path.join(model_assets_dir, MODEL_SUBDIR)
    os.makedirs(model_dir, exist_ok=True)

    model = model_map[model_name](**model_params)
    model.fit(X_train_vect_added_scaled, y_train)
    pred_valid = model.predict(X_valid_vect_added_scaled)

    # dummy classifier
    dummy_model = DummyClassifier()
    dummy_model.fit(X_train_vect_added_scaled, y_train)
    dummy_pred_valid = dummy_model.predict(X_valid_vect_added_scaled)

    eval_scores = {'dummy': {}}
    for metric in metrics:
        eval_scores[metric.__name__] = metric(y_valid, pred_valid)
        eval_scores['dummy'][metric.__name__] = \
            metric(y_valid, dummy_pred_valid)
    print(eval_scores)

    if model_dir not in [None, '']:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(
            model_dir, model_filename))
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(eval_scores, f)

    return model, eval_scores
