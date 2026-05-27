"""CatBoost."""

from catboost import CatBoostClassifier
from config import HYPERPARAMS

def build_model():
    params = HYPERPARAMS["CatBoost"]
    return CatBoostClassifier(**params)