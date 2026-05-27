"""
Módulo de Avaliação e Validação Cruzada (OOF)
---------------------------------------------
Atualizado para o benchmark da AMEX. Implementa o StratifiedKFold
para garantir validação cruzada robusta e extrai tanto as métricas 
clássicas (ROC-AUC) quanto a métrica oficial de negócio (AMEX Score).
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
)

# Importando a métrica customizada que criamos na Fase 0
from src.evaluation.amex_metric import amex_metric

logger = logging.getLogger(__name__)

def evaluate_model(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Avaliação estática (Holdout) para predições diretas.
    Ideal para as Provas de Conceito (Fase 1) e validação da Base de Teste Final (20%).
    """
    y_pred_bin = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        "AMEX_Score": amex_metric(y_true, y_pred_proba),
        "ROC_AUC": roc_auc_score(y_true, y_pred_proba),
        "F1_Score": f1_score(y_true, y_pred_bin, zero_division=0),
        "Precision": precision_score(y_true, y_pred_bin, zero_division=0),
        "Recall": recall_score(y_true, y_pred_bin, zero_division=0)
    }
    
    return metrics

def evaluate_model_cv(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> Dict[str, Any]:
    """
    Executa a Validação Cruzada Estratificada (OOF - Out-Of-Fold).
    Garante que a métrica reportada na Fase 2 e 3 seja blindada contra overfitting local.
    
    Retorna as métricas globais e o array de predições OOF (essencial para Stacking).
    """
    logger.info(f"Iniciando StratifiedKFold ({n_splits} splits) para o modelo...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Array vazio para armazenar as predições seguras de toda a base de treino
    oof_predictions = np.zeros(len(X))
    fold_metrics = []
    
    # Conversão de segurança para NumPy arrays (evita erros de indexação com Pandas/Polars)
    X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y_arr = y.to_numpy() if isinstance(y, pd.Series) else y

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        # Separa as dobras
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_val, y_val = X_arr[val_idx], y_arr[val_idx]
        
        # Treina o modelo na fração de treino (80% da dobra interna)
        model.fit(X_train, y_train)
        
        # Gera probabilidades na fração de validação (20% da dobra interna)
        # Proteção para modelos que não possuem predict_proba (ex: SVM linear sem ajuste)
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_val)[:, 1]
        else:
            preds = model.predict(X_val)
            
        # Armazena as predições no local correto do array OOF
        oof_predictions[val_idx] = preds
        
        # Calcula e guarda o AMEX Score do fold atual (para monitorar variância)
        fold_score = amex_metric(y_val, preds)
        fold_metrics.append(fold_score)
        logger.debug(f"Fold {fold}/{n_splits} - AMEX Score: {fold_score:.4f}")

    # =========================================================
    # CÁLCULO DAS MÉTRICAS GLOBAIS (Em toda a base OOF)
    # =========================================================
    # Como as predições OOF nunca viram seus respectivos gabaritos durante o treino,
    # calcular a métrica em cima de todo o array OOF é a medida mais realista possível.
    
    global_metrics = evaluate_model(y_arr, oof_predictions)
    
    # Adicionamos a variância (desvio padrão entre os folds) para avaliar estabilidade
    global_metrics["AMEX_Std"] = np.std(fold_metrics)
    global_metrics["OOF_Predictions"] = oof_predictions
    
    logger.info(f"CV Finalizado | Global AMEX: {global_metrics['AMEX_Score']:.4f} (±{global_metrics['AMEX_Std']:.4f})")
    
    return global_metrics