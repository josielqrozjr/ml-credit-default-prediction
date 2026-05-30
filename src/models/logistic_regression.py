"""Regressão Logística (Versão RAPIDS/GPU)."""

from cuml.linear_model import LogisticRegression
from config import HYPERPARAMS

def build_model():
    # Usamos .copy() para não alterar o dicionário original do config acidentalmente
    params = HYPERPARAMS.get("Logistic Regression", {}).copy()
    
    # -------------------------------------------------------------------------
    # Tratamento de compatibilidade: Scikit-Learn -> cuML
    # -------------------------------------------------------------------------
    # 1. cuML não suporta balanceamento interno (class_weight) na Regressão Logística
    params.pop("class_weight", None)
    
    # 2. n_jobs não existe (a GPU já paralela nativamente)
    params.pop("n_jobs", None)
    
    # 3. random_state não é aceito na inicialização do solver matemático da GPU
    params.pop("random_state", None)
    
    # 4. Tradução do otimizador: lbfgs (Sklearn) -> qn (Quasi-Newton do cuML)
    if params.get("solver") == "lbfgs":
        params["solver"] = "qn"
        
    return LogisticRegression(**params)