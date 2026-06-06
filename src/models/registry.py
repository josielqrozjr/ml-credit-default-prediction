"""Registro central de modelos com interface padronizada."""

from src.models.logistic_regression import build_model as _build_lr
from src.models.knn import build_model as _build_knn
from src.models.ann import build_model as _build_ann
from src.models.random_forest import build_model as _build_rf
from src.models.xgboost_model import build_model as _build_xgb
from src.models.lightgbm_model import build_model as _build_lgb
from src.models.catboost_model import build_model as _build_cat
from src.models.voting_classifier import build_model as _build_voting
from src.models.stacking import build_model as _build_stacking
from src.models.blending import build_model as _build_blending

MODEL_REGISTRY: dict[str, callable] = {
    "Logistic Regression": _build_lr,
    "KNN": _build_knn,
    "ANN (MLP)": _build_ann,
    "Random Forest": _build_rf,
    "XGBoost": _build_xgb,
    "LightGBM": _build_lgb,
    "CatBoost": _build_cat,
    "Voting Classifier": _build_voting,
    "Stacking": _build_stacking,
    "Blending": _build_blending,
}