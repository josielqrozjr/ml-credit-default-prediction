"""
Fase 1: Provas de Conceito (Validação Metodológica)
---------------------------------------------------
Este script executa dois experimentos empíricos fundamentais para o TCC:
1. Comprova a eficácia do Feature Selection (Base Completa vs Base Enxuta).
2. Comprova a superioridade do Balanceamento Algorítmico contra Undersampling.
"""

import sys
import time
import logging
import gc
import pandas as pd
import polars as pl
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

from src.config import RANDOM_SEED, RESULTS_DIR, TRAIN_DATA_PATH, SELECTED_FEATURES_PATH, GPU_AVAILABLE, DEVICE
from src.evaluation.metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

def run_dimensionality_poc(X_full: pd.DataFrame, X_reduced: pd.DataFrame, y: pd.Series):
    logger.info("=== Iniciando Experimento 1: Dimensionalidade ===")
    results = []
    
    xgb_params = {"scale_pos_weight": 3, "n_estimators": 200, "max_depth": 6, "random_state": RANDOM_SEED}
    if GPU_AVAILABLE:
        xgb_params["tree_method"] = "hist" # Hist é mais estável que gpu_hist nas novas versões
        xgb_params["device"] = "cuda"

    models = {
        "Logistic Regression (CPU)": LogisticRegression(class_weight="balanced", max_iter=500, random_state=RANDOM_SEED),
        "XGBoost (GPU/CPU)": XGBClassifier(**xgb_params)
    }

    datasets = {"Enxuta (400 features)": X_reduced, "Completa (3265 features)": X_full}

    for db_name, X_data in datasets.items():
        logger.info(f"-> Preparando split para a base: {db_name}")
        X_train, X_val, y_train, y_val = train_test_split(X_data, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
        
        logger.info("-> Split concluído. Limpando lixo da memória...")
        gc.collect()

        for model_name, model in models.items():
            logger.info(f"-> [{model_name}] Iniciando treinamento em {db_name}...")
            start_time = time.time()
            
            # Ponto crítico: Se reiniciar aqui, é OOM no fit ou pico de GPU
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            logger.info(f"-> [{model_name}] Treinamento finalizado em {train_time:.1f}s. Iniciando predição...")
            
            preds = model.predict_proba(X_val)[:, 1]
            metrics = evaluate_model(y_val, preds)
            
            results.append({
                "Experimento": "Dimensionalidade",
                "Modelo": model_name,
                "Base de Dados": db_name,
                "Tempo Treino (s)": round(train_time, 2),
                "AMEX Score": metrics["AMEX_Score"],
                "ROC AUC": metrics["ROC_AUC"],
                "AUPRC": metrics["AUPRC"]
            })
            logger.info(f"-> [{model_name}] AMEX Score calculado: {metrics['AMEX_Score']:.4f}")
            
            logger.info("-> Limpando cache do modelo...")
            gc.collect()

    return pd.DataFrame(results)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"GPU Disponível: {'SIM' if GPU_AVAILABLE else 'NÃO'}")
    if GPU_AVAILABLE:
        import torch
        logger.info(f"Dispositivo: {DEVICE} | GPU: {torch.cuda.get_device_name(0)}")
        # Esvazia cache residual da placa de vídeo antes de começar
        torch.cuda.empty_cache()

    logger.info("Passo 1: Carregando a base usando Polars para eficiência...")
    try:
        # Lendo com Polars (muito mais leve na RAM)
        df_lazy = pl.scan_parquet(TRAIN_DATA_PATH)
        
        logger.info("Passo 2: Convertendo tipos para float32 (Economia de 50% de RAM)...")
        # Força numéricos para float32 para evitar estouro de memória
        df_full_pl = df_lazy.collect()
        
        logger.info("Passo 3: Convertendo para Pandas...")
        df_full = df_full_pl.to_pandas()
        
        # Deleta a versão Polars da memória instantaneamente
        del df_full_pl
        gc.collect()
        
        y = df_full["target"].astype("int8")
        X_full = df_full.drop(columns=["target"]).astype("float32")
        
        # Limpa o DataFrame original inteiro
        del df_full
        gc.collect()
        
        logger.info("Passo 4: Carregando lista de features selecionadas...")
        with open(SELECTED_FEATURES_PATH, "r") as f:
            selected_cols = [line.strip() for line in f.readlines()]
            if "target" in selected_cols:
                selected_cols.remove("target")
            
        X_reduced = X_full[selected_cols]
        logger.info(f"Formato da Base Completa: {X_full.shape} | Base Enxuta: {X_reduced.shape}")
        
    except Exception as e:
        logger.exception(f"Erro fatal no carregamento: {e}")
        sys.exit(1)

    # 1. Roda a Prova de Dimensionalidade
    df_dim = run_dimensionality_poc(X_full, X_reduced, y)
    df_dim.to_csv(RESULTS_DIR / "poc_01_dimensionalidade.csv", index=False)
    
    # IMPORTANTE: Após a prova de dimensionalidade, deletamos a base gigante!
    logger.info("Deletando Base Completa (3265 features) da RAM definitivamente...")
    del X_full
    gc.collect()
    
    # 2. Roda a Prova de Balanceamento (Usando apenas a base reduzida)
    # df_bal = run_balancing_poc(X_reduced, y) # [COMENTADO TEMPORARIAMENTE PARA ISOLAR O ERRO]
    # df_bal.to_csv(RESULTS_DIR / "poc_02_balanceamento.csv", index=False)
    
    logger.info("Fase 1 concluída com sucesso! Resultados exportados para a pasta 'results/'")


if __name__ == "__main__":
    main()