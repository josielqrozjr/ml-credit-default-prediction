"""
Fase 5: O Teste Final (Produção)
----------------------------------------
Treina os modelos campeões (otimizados na Fase 3) em 100% da base de treino.
Avalia a arquitetura final (Voting Classifier) na base de teste isolada (20%),
garantindo a métrica definitiva e livre de viés do projeto.
"""

import sys
import time
import logging
import gc
import json
import warnings
import numpy as np
import pandas as pd
import polars as pl

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Importações do projeto
from config import (
    RANDOM_SEED, RESULTS_DIR, RESULTS_BEST_MODELS, 
    TRAIN_DATA_PATH, TEST_DATA_PATH, SELECTED_FEATURES_PATH, GPU_AVAILABLE
)
from app.evaluation.metrics import evaluate_model

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(path, selected_features=None):
    """Carrega um dataset (Treino ou Teste) e aplica o filtro de features."""
    df_pd = pl.scan_parquet(path).collect().to_pandas()
    
    cols_to_drop = [col for col in ["customer_ID", "S_2"] if col in df_pd.columns]
    if cols_to_drop:
        df_pd = df_pd.drop(columns=cols_to_drop)
        
    object_cols = df_pd.select_dtypes(include=['object', 'string', 'category']).columns
    if len(object_cols) > 0:
        df_pd = df_pd.drop(columns=object_cols)

    y = df_pd["target"].astype("int8")
    X = df_pd.drop(columns=["target"]).astype("float32")
    del df_pd
    gc.collect()
    
    # Aplica o filtro de colunas se fornecido
    if selected_features is not None:
        valid_cols = [col for col in selected_features if col in X.columns]
        X = X[valid_cols]
        
    return X, y


def get_selected_features():
    """Lê o arquivo .txt com as 400 features da Fase 1."""
    with open(SELECTED_FEATURES_PATH, "r") as f:
        selected_cols = [line.strip() for line in f.readlines()]
    if "target" in selected_cols:
        selected_cols.remove("target")
    return selected_cols


def load_optimized_models():
    """Lê o JSON da Fase 3 e instancia os algoritmos."""
    json_path = RESULTS_BEST_MODELS / "optuna_best_params.json"
    if not json_path.exists():
        raise FileNotFoundError("Arquivo optuna_best_params.json não encontrado.")
        
    with open(json_path, "r") as f:
        best_params = json.load(f)
        
    xgb = XGBClassifier(
        **best_params["XGBoost"]["params"],
        scale_pos_weight=3,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="gpu_hist" if GPU_AVAILABLE else "hist",
        device="cuda" if GPU_AVAILABLE else "cpu"
    )
    
    lgb = LGBMClassifier(
        **best_params["LightGBM"]["params"],
        is_unbalance=True,
        random_state=RANDOM_SEED,
        n_jobs=-1 if GPU_AVAILABLE else 1,
        device="gpu" if GPU_AVAILABLE else "cpu",
        verbose=-1
    )
    
    cat = CatBoostClassifier(
        **best_params["CatBoost"]["params"],
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=0,
        task_type="GPU" if GPU_AVAILABLE else "CPU"
    )
    
    return xgb, lgb, cat


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== INICIANDO FASE 5: TESTE FINAL (PRODUÇÃO) ===")
    
    try:
        features = get_selected_features()
        logger.info("Passo 1: Carregando Base de Treino (80%)...")
        X_train, y_train = load_dataset(TRAIN_DATA_PATH, features)
        
        logger.info("Passo 2: Carregando Base de Teste (20%)...")
        X_test, y_test = load_dataset(TEST_DATA_PATH, features)
        
        xgb_model, lgb_model, cat_model = load_optimized_models()
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)
        
    # Verifica alinhamento de features (Garante a ausência de Leakage)
    assert list(X_train.columns) == list(X_test.columns), "Erro: As colunas de treino e teste não coincidem!"
    
    logger.info(f"Shape de Treino: {X_train.shape} | Shape de Teste: {X_test.shape}")
    
    logger.info("\nTreinando os modelos definitivos com 100% da base de treino...")
    start_time = time.time()
    
    # Treina na base completa de uma vez
    xgb_model.fit(X_train, y_train)
    logger.info(" -> XGBoost treinado.")
    
    lgb_model.fit(X_train, y_train)
    logger.info(" -> LightGBM treinado.")
    
    cat_model.fit(X_train, y_train)
    logger.info(" -> CatBoost treinado.")
    
    logger.info(f"Treinamento concluído em {time.time() - start_time:.1f}s.")
    
    # Libera memória RAM pesada
    del X_train, y_train
    gc.collect()
    
    logger.info("\nGerando predições na Base Isolada de Teste...")
    pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
    pred_lgb = lgb_model.predict_proba(X_test)[:, 1]
    pred_cat = cat_model.predict_proba(X_test)[:, 1]
    
    # O nosso grande campeão:
    pred_voting = (pred_xgb + pred_lgb + pred_cat) / 3
    
    logger.info("\nCalculando as métricas finais...")
    final_metrics = evaluate_model(y_test, pred_voting)
    
    logger.info("=========================================")
    logger.info("RESULTADOS DO TESTE FINAL (PRODUÇÃO)")
    logger.info("=========================================")
    logger.info(f" -> AMEX Score Oficial : {final_metrics['AMEX_Score']:.4f}")
    logger.info(f" -> ROC AUC Global     : {final_metrics['ROC_AUC']:.4f}")
    logger.info(f" -> F1-Score           : {final_metrics['F1_Score']:.4f}")
    logger.info(f" -> Recall (Sensibilidade): {final_metrics['Recall']:.4f}")
    logger.info("=========================================")
    
    # Salva o resultado final para o TCC
    df_final = pd.DataFrame([final_metrics])
    df_final.insert(0, "Modelo", "Voting Classifier (Final)")
    output_path = RESULTS_DIR / "phase5_final_test.csv"
    df_final.to_csv(output_path, index=False)
    logger.info(f"Relatório final salvo em: {output_path}")

if __name__ == "__main__":
    main()