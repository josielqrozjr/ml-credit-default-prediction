"""
Fase 3: Otimização Suprema (Optuna)
-----------------------------------
Aplica Otimização Bayesiana (Optuna) exclusivamente nos 3 modelos campeões da Fase 2.
O objetivo é maximizar a métrica AMEX_Score testando dezenas de combinações de
hiperparâmetros em um espaço de busca pré-definido.
Salva os melhores parâmetros em disco para uso na Fase 4.
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
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Importações do projeto
from config import (
    RANDOM_SEED, RESULTS_DIR, RESULTS_BEST_MODELS, TRAIN_DATA_PATH, 
    SELECTED_FEATURES_PATH, N_SPLITS, OPTUNA_GRIDS, GPU_AVAILABLE
)
from app.evaluation.metrics import evaluate_model

# Suprime warnings do Optuna e do LightGBM para manter o terminal limpo
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

# Configuração de Trials (Ajuste para 5 ou 10 se for testar no Mac. Use 50 ou 100 na GPU)
N_TRIALS = 100 if GPU_AVAILABLE else 50

def load_and_prepare_data():
    """Carrega a base enxuta (400 features)."""
    logger.info("Passo 1: Carregando a base de treino via Polars...")
    df_pd = pl.scan_parquet(TRAIN_DATA_PATH).collect().to_pandas()
    
    cols_to_drop = [col for col in ["customer_ID", "S_2"] if col in df_pd.columns]
    if cols_to_drop:
        df_pd = df_pd.drop(columns=cols_to_drop)
        
    object_cols = df_pd.select_dtypes(include=['object', 'string', 'category']).columns
    if len(object_cols) > 0:
        df_pd = df_pd.drop(columns=object_cols)

    y = df_pd["target"].astype("int8")
    X_full = df_pd.drop(columns=["target"]).astype("float32")
    
    del df_pd
    gc.collect()
    
    with open(SELECTED_FEATURES_PATH, "r") as f:
        selected_cols = [line.strip() for line in f.readlines()]
        if "target" in selected_cols:
            selected_cols.remove("target")
        selected_cols = [col for col in selected_cols if col in X_full.columns]
        
    X_reduced = X_full[selected_cols]
    
    del X_full
    gc.collect()
    
    return X_reduced, y

def get_model_instance(model_name, trial_params):
    """Instancia o algoritmo correspondente já com as proteções de hardware."""
    if model_name == "LightGBM":
        # Força o n_jobs=1 no Mac (se não tiver GPU) para evitar o deadlock do OpenMP
        n_jobs_val = -1 if GPU_AVAILABLE else 1
        device_val = "gpu" if GPU_AVAILABLE else "cpu"
        
        return LGBMClassifier(
            **trial_params,
            is_unbalance=True,
            random_state=RANDOM_SEED,
            n_jobs=n_jobs_val,
            device=device_val,
            verbose=-1
        )
        
    elif model_name == "XGBoost":
        tree_method_val = "gpu_hist" if GPU_AVAILABLE else "hist"
        device_val = "cuda" if GPU_AVAILABLE else "cpu"
        
        return XGBClassifier(
            **trial_params,
            scale_pos_weight=3,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method=tree_method_val,
            device=device_val
        )
        
    elif model_name == "CatBoost":
        task_type_val = "GPU" if GPU_AVAILABLE else "CPU"
        
        return CatBoostClassifier(
            **trial_params,
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED,
            verbose=0,
            task_type=task_type_val
        )
    else:
        raise ValueError(f"Modelo não suportado na Fase 3: {model_name}")

def objective(trial, model_name, X, y, skf):
    """Função Objetivo do Optuna. Treina o K-Fold e retorna o AMEX Score."""
    
    # 1. Busca os limites configurados no config.py
    grid = OPTUNA_GRIDS[model_name]
    trial_params = {}
    
    for param_name, param_config in grid.items():
        param_type = param_config[0]
        if param_type == "int":
            trial_params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
        elif param_type == "float":
            if len(param_config) == 4 and param_config[3] == "log":
                trial_params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
            else:
                trial_params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])

    # --- PROTEÇÃO MATEMÁTICA DO LIGHTGBM ---
    # Garante que num_leaves nunca ultrapasse o limite de 2^max_depth
    if model_name == "LightGBM" and "max_depth" in trial_params and "num_leaves" in trial_params:
        max_allowed_leaves = (2 ** trial_params["max_depth"]) - 1
        # Se o Optuna sugerir 150 folhas, mas a profundidade permitir só 31, cortamos em 31.
        trial_params["num_leaves"] = min(trial_params["num_leaves"], max_allowed_leaves)
    # ---------------------------------------

    # 2. Instancia o modelo com os parâmetros sugeridos
    model = get_model_instance(model_name, trial_params)
    
    oof_preds = np.zeros(len(X))
    
    # 3. K-Fold para garantir robustez (sem leakage)
    for train_idx, val_idx in skf.split(X, y):
        # Proteção de memória do Apple Silicon (recriação do DF)
        X_train = pd.DataFrame(X.iloc[train_idx].values, columns=X.columns)
        y_train = pd.Series(y.iloc[train_idx].values)
        X_val = pd.DataFrame(X.iloc[val_idx].values, columns=X.columns)
        
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)
        preds_positive = preds[:, 1] if len(preds.shape) > 1 else preds
        oof_preds[val_idx] = preds_positive
        
        # Limpa memória a cada fold para não estourar a RAM no Optuna
        del X_train, y_train, X_val
        gc.collect()

    # 4. Avalia OOF e retorna a métrica principal para o Optuna
    metrics = evaluate_model(y, oof_preds)
    return metrics["AMEX_Score"]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_BEST_MODELS.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Aceleração por GPU Detectada: {'SIM' if GPU_AVAILABLE else 'NÃO'}")
    
    try:
        X, y = load_and_prepare_data()
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        sys.exit(1)
        
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    # Os Top 3 consolidados da Fase 2
    top3_models = ["XGBoost", "LightGBM", "CatBoost"]
    
    best_params_all = {}
    
    logger.info(f"=== INICIANDO OTIMIZAÇÃO SUPREMA (OPTUNA) ===")
    logger.info(f"Modelos: {top3_models} | Trials por Modelo: {N_TRIALS}")
    
    for model_name in top3_models:
        logger.info(f"\n-> Iniciando Busca Bayesiana para: {model_name}")
        start_time = time.time()
        
        # Criamos um "Study" do Optuna focado em MAXIMIZAR o score
        study = optuna.create_study(
            direction="maximize", 
            sampler=TPESampler(seed=RANDOM_SEED)
        )
        
        # --- CALLBACK DE SEGURANÇA (NOVO) ---
        # Salva o progresso no JSON toda vez que encontrar um hiperparâmetro melhor
        def save_checkpoint(study, trial):
            best_params_all[model_name] = {
                "best_amex_score": study.best_value,
                "params": study.best_params
            }
            with open(RESULTS_BEST_MODELS / "optuna_best_params.json", "w") as f:
                json.dump(best_params_all, f, indent=4)
        # ------------------------------------

        # Inicia o loop de Trials com o Callback ativado
        study.optimize(
            lambda trial: objective(trial, model_name, X, y, skf), 
            n_trials=N_TRIALS,
            n_jobs=1,
            callbacks=[save_checkpoint] # <-- Callback injetado aqui
        )
        
        total_time = time.time() - start_time
        
        logger.info(f"-> [{model_name}] Otimização Concluída em {total_time:.1f}s")
        logger.info(f"   Melhor AMEX Score Encontrado: {study.best_value:.4f}")
        logger.info(f"   Melhores Hiperparâmetros: {study.best_params}")
        
        # Salva no dicionário global
        best_params_all[model_name] = {
            "best_amex_score": study.best_value,
            "params": study.best_params
        }
        
    # Exporta o JSON com as chaves mestras para a Fase 4
    output_path = RESULTS_BEST_MODELS / "optuna_best_params.json"
    with open(output_path, "w") as f:
        json.dump(best_params_all, f, indent=4)
        
    logger.info("\n=========================================")
    logger.info(f"Fase 3 Concluída! Hiperparâmetros salvos em: {output_path}")
    logger.info("=========================================")

if __name__ == "__main__":
    main()