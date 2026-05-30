"""LightGBM (Compatibilidade RAPIDS/GPU)."""

import cupy as cp
from lightgbm import LGBMClassifier
from config import RANDOM_SEED, HYPERPARAMS

class RAPIDSLightGBM:
    """
    Wrapper para garantir que o LightGBM funcione de forma fluida no pipeline RAPIDS.
    Ele assegura que a entrada seja um formato que a API aceite e, mais criticamente,
    intercepta o predict_proba para devolver um array CuPy (mantendo o dado na VRAM).
    """
    def __init__(self, **kwargs):
        self.model = LGBMClassifier(**kwargs)

    def _prepare_data(self, data):
        """Converte DataFrames (cuDF ou Pandas) para matrizes antes de injetar no modelo."""
        if hasattr(data, 'to_cupy'):
            return data.to_cupy()
        elif hasattr(data, 'values'):
            return data.values
        return data

    def fit(self, X, y):
        X_prep = self._prepare_data(X)
        y_prep = self._prepare_data(y)
        
        self.model.fit(X_prep, y_prep)
        return self

    def predict_proba(self, X):
        X_prep = self._prepare_data(X)
        probs = self.model.predict_proba(X_prep)
        
        # O LightGBM retorna NumPy (CPU). Convertendo de volta para CuPy (GPU).
        return cp.asarray(probs, dtype=cp.float32)

    def predict(self, X):
        X_prep = self._prepare_data(X)
        preds = self.model.predict(X_prep)
        return cp.asarray(preds, dtype=cp.int32)
        
    @property
    def classes_(self):
        # Expondo as classes caso meta-modelos como o Stacking/Blending procurem
        return self.model.classes_

def build_model():
    # Usamos .copy() para garantir que não alteramos o dicionário global
    params = HYPERPARAMS.get("LightGBM", {}).copy()
    
    # Assegura que hiperparâmetros de controle não se percam
    params["random_state"] = RANDOM_SEED
    
    return RAPIDSLightGBM(**params)