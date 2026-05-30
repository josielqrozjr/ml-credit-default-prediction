"""Voting Classifier — integração de XGBoost, LightGBM e CatBoost com soft voting (RAPIDS/GPU)."""

import cupy as cp
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.xgboost_model import build_model as build_xgb
from src.models.lightgbm_model import build_model as build_lgb
from src.models.catboost_model import build_model as build_cat

class RAPIDSVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Implementação customizada de Soft Voting nativa em GPU.
    Garante que as predições dos três maiores modelos (Boosting) sejam 
    calculadas e combinadas usando matemática matricial na VRAM.
    """
    def __init__(self, estimators, voting="soft"):
        self.estimators = estimators
        # Mantemos apenas o soft voting, pois o hard voting (0 ou 1) anula 
        # as predições contínuas exigidas pela Métrica AMEX.
        self.voting = voting

    def fit(self, X, y):
        # Treina cada modelo individualmente. 
        # A aceleração de hardware ocorre internamente em cada algoritmo.
        for name, model in self.estimators:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        probs_list = []
        
        for name, model in self.estimators:
            preds = model.predict_proba(X)
            
            # Normaliza a saída caso algum modelo retorne um array 1D
            if len(preds.shape) == 1:
                preds = cp.column_stack((1 - preds, preds))
                
            probs_list.append(preds)
            
        # Empilha as matrizes de predição e tira a média (Soft Voting)
        # axis=0 calcula a média verticalmente entre os modelos para cada linha de cliente
        avg_probs = cp.mean(cp.stack(probs_list), axis=0)
        
        return avg_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        probs_positive = probs[:, 1]
        return (probs_positive >= 0.5).astype(cp.int32)

def build_model():
    estimators = [
        ("xgb", build_xgb()),
        ("lgb", build_lgb()),
        ("cat", build_cat()),
    ]
    
    return RAPIDSVotingClassifier(
        estimators=estimators, 
        voting="soft"
    )