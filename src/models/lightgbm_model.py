"""LightGBM."""

from lightgbm import LGBMClassifier
from config import HYPERPARAMS

def build_model():
    params = HYPERPARAMS["LightGBM"]
    return LGBMClassifier(**params)