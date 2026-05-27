"""Blending — Ensemble com holdout explícito (20%) para treinar o meta-learner."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED
from src.models.random_forest import build_model as build_rf
from src.models.xgboost_model import build_model as build_xgb
from src.models.lightgbm_model import build_model as build_lgb

class BlendingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimators, meta_estimator, holdout_size=0.2, random_state=42):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.holdout_size = holdout_size
        self.random_state = random_state

    def fit(self, X, y):
        # Proteção contra formatos (Pandas vs Numpy)
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else y

        # Split estratificado interno do Blending
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_arr, y_arr, 
            test_size=self.holdout_size, 
            stratify=y_arr, 
            random_state=self.random_state
        )

        meta_features_holdout = np.zeros((X_holdout.shape[0], len(self.base_estimators)))

        # Treina a base no subset e gera predições no holdout
        for i, (name, model) in enumerate(self.base_estimators):
            model.fit(X_train, y_train)
            meta_features_holdout[:, i] = model.predict_proba(X_holdout)[:, 1]

        # O Meta-learner aprende a combinar baseado apenas nas predições do holdout
        self.meta_estimator.fit(meta_features_holdout, y_holdout)

        # Retreinar base learners em todo o dataset (X) para não perder dados em produção
        for name, model in self.base_estimators:
            model.fit(X_arr, y_arr)

        return self

    def predict_proba(self, X):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        meta_features = np.zeros((X_arr.shape[0], len(self.base_estimators)))
        
        for i, (name, model) in enumerate(self.base_estimators):
            meta_features[:, i] = model.predict_proba(X_arr)[:, 1]
            
        return self.meta_estimator.predict_proba(meta_features)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def build_model():
    base_estimators = [
        ("rf", build_rf()),
        ("xgb", build_xgb()),
        ("lgb", build_lgb()),
    ]
    meta_estimator = LogisticRegression(
        max_iter=1000, 
        solver="lbfgs", 
        class_weight="balanced",
        random_state=RANDOM_SEED
    )
    
    return BlendingClassifier(
        base_estimators=base_estimators,
        meta_estimator=meta_estimator,
        holdout_size=0.2,
        random_state=RANDOM_SEED,
    )