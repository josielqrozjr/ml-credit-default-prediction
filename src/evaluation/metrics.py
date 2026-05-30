"""
Módulo de Avaliação e Validação Cruzada (OOF)
---------------------------------------------
Atualizado para o benchmark da AMEX. Implementa o StratifiedKFold
para garantir validação cruzada robusta e extrai tanto as métricas 
clássicas (ROC-AUC) quanto a métrica oficial de negócio (AMEX Score).
"""

import cupy as cp
import cudf
import logging
from typing import Dict, Any

from cuml.model_selection import StratifiedKFold
from cuml.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# Importando a métrica customizada que criamos na Fase 0
from src.evaluation.amex_metric import amex_metric

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred_proba, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Avaliação estática (Holdout) para predições diretas na GPU.
    Ideal para as Provas de Conceito (Fase 1) e validação da Base de Teste Final (20%).
    """
    # Garante que os dados estejam em arrays do CuPy na VRAM
    y_true_cp = cp.asarray(y_true).flatten()
    y_pred_cp = cp.asarray(y_pred_proba).flatten()
    
    y_pred_bin = (y_pred_cp >= threshold).astype(cp.int32)
    
    # -----------------------------------------------------------------
    # Cálculo manual ultra-rápido de métricas matriciais na GPU (CuPy)
    # -----------------------------------------------------------------
    tp = float(cp.sum((y_true_cp == 1) & (y_pred_bin == 1)))
    fp = float(cp.sum((y_true_cp == 0) & (y_pred_bin == 1)))
    tn = float(cp.sum((y_true_cp == 0) & (y_pred_bin == 0)))
    fn = float(cp.sum((y_true_cp == 1) & (y_pred_bin == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Matriz de Confusão estruturada para lista (facilitando a exportação visual)
    cm_list = [[tn, fp], [fn, tp]]
    
    # -----------------------------------------------------------------
    # AUPRC (Fallback para CPU)
    # -----------------------------------------------------------------
    # Como o cuML não possui AUPRC nativo, enviamos temporariamente 
    # os vetores 1D para a RAM (asnumpy) apenas para este cálculo isolado.
    auprc_val = average_precision_score(cp.asnumpy(y_true_cp), cp.asnumpy(y_pred_cp))

    metrics = {
        "AMEX_Score": amex_metric(y_true_cp, y_pred_cp),
        "ROC_AUC": float(roc_auc_score(y_true_cp, y_pred_cp)),
        "AUPRC": float(auprc_val),
        "F1_Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Confusion_Matrix": cm_list
    }
    
    return metrics

def evaluate_model_cv(model, X: cudf.DataFrame, y: cudf.Series, n_splits: int = 5, random_state: int = 42) -> Dict[str, Any]:
    """
    Executa a Validação Cruzada Estratificada (OOF - Out-Of-Fold) via RAPIDS.
    Garante que a métrica reportada na Fase 2 e 3 seja blindada contra overfitting local.
    
    Retorna as métricas globais e o array de predições OOF (essencial para Stacking).
    """
    logger.info(f"Iniciando StratifiedKFold ({n_splits} splits) nativo na GPU para o modelo...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Array vazio na GPU para armazenar as predições seguras de toda a base de treino
    oof_predictions = cp.zeros(len(X), dtype=cp.float32)
    fold_metrics = []
    
    # Conversão de segurança para CuPy arrays para evitar erros de indexação
    X_arr = X.to_cupy() if hasattr(X, 'to_cupy') else cp.asarray(X)
    y_arr = y.to_cupy() if hasattr(y, 'to_cupy') else cp.asarray(y)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        # Separa as dobras usando indexação matricial diretamente na VRAM
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_val, y_val = X_arr[val_idx], y_arr[val_idx]
        
        # Treina o modelo na fração de treino
        model.fit(X_train, y_train)
        
        # Gera probabilidades na fração de validação
        if hasattr(model, "predict_proba"):
            # Para cuML e XGBoost/LGBM com interface sklearn
            preds = model.predict_proba(X_val)
            # Alguns modelos cuML retornam 1D se for classificação binária explícita
            preds = preds[:, 1] if len(preds.shape) > 1 else preds
        else:
            preds = model.predict(X_val)
            
        # Armazena as predições no local correto do array OOF
        oof_predictions[val_idx] = cp.asarray(preds).flatten()
        
        # Calcula e guarda o AMEX Score do fold atual
        fold_score = amex_metric(y_val, preds)
        fold_metrics.append(fold_score)
        logger.debug(f"Fold {fold}/{n_splits} - AMEX Score: {fold_score:.4f}")

    # =========================================================
    # CÁLCULO DAS MÉTRICAS GLOBAIS (Em toda a base OOF)
    # =========================================================
    global_metrics = evaluate_model(y_arr, oof_predictions)
    
    # Adicionamos a variância (desvio padrão entre os folds) para avaliar estabilidade
    global_metrics["AMEX_Std"] = float(cp.std(cp.array(fold_metrics)))
    # Mantemos as predições no formato CuPy para serem consumidas pelos meta-modelos depois
    global_metrics["OOF_Predictions"] = oof_predictions
    
    logger.info(f"CV Finalizado | Global AMEX: {global_metrics['AMEX_Score']:.4f} (±{global_metrics['AMEX_Std']:.4f})")
    
    return global_metrics