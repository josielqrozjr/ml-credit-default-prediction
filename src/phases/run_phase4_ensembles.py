"""
Fase 4: Meta-Classificadores (Ensembles)
----------------------------------------
Carrega os hiperparâmetros campeões da Fase 3 e combina os 3 melhores
modelos utilizando técnicas avançadas de agrupamento (Voting, Stacking e Blending).
Avalia se a combinação de modelos consegue superar o desempenho do melhor 
modelo individual (LightGBM).
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

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Importações do projeto
from config import (
    RANDOM_SEED, RESULTS_DIR, RESULTS_BEST_MODELS, TRAIN_DATA_PATH, 
    SELECTED_FEATURES_PATH, N_SPLITS, GPU_AVAILABLE
)
from src.evaluation.metrics import evaluate_model

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)


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


def load_best_models():
    """Lê o JSON da Fase 3 e instancia os algoritmos otimizados."""
    json_path = RESULTS_BEST_MODELS / "optuna_best_params.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {json_path}. Rode a Fase 3 primeiro.")
        
    with open(json_path, "r") as f:
        best_params = json.load(f)
        
    logger.info("Hiperparâmetros da Fase 3 carregados com sucesso!")
    
    # 1. Instancia XGBoost
    xgb_params = best_params["XGBoost"]["params"]
    xgb = XGBClassifier(
        **xgb_params,
        scale_pos_weight=3,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="gpu_hist" if GPU_AVAILABLE else "hist",
        device="cuda" if GPU_AVAILABLE else "cpu"
    )
    
    # 2. Instancia LightGBM
    lgb_params = best_params["LightGBM"]["params"]
    lgb = LGBMClassifier(
        **lgb_params,
        is_unbalance=True,
        random_state=RANDOM_SEED,
        n_jobs=-1 if GPU_AVAILABLE else 1,
        device="gpu" if GPU_AVAILABLE else "cpu",
        verbose=-1
    )
    
    # 3. Instancia CatBoost
    cat_params = best_params["CatBoost"]["params"]
    cat = CatBoostClassifier(
        **cat_params,
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=0,
        task_type="GPU" if GPU_AVAILABLE else "CPU"
    )
    
    return xgb, lgb, cat


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        X, y = load_and_prepare_data()
        xgb_model, lgb_model, cat_model = load_best_models()
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)
        
    logger.info("=== INICIANDO FASE 4: META-CLASSIFICADORES ===")
    
    # =====================================================================
    # ETAPA 1: Geração de Matriz Base (OOF) para Voting e Stacking
    # =====================================================================
    logger.info("Treinando Modelos Base e Extraindo Predições OOF (Isso pode levar alguns minutos)...")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f" -> Processando Fold {fold+1}/{N_SPLITS}...")
        
        # Proteção contra Segmentation Fault no Apple Silicon
        X_train = pd.DataFrame(X.iloc[train_idx].values, columns=X.columns)
        y_train = pd.Series(y.iloc[train_idx].values)
        X_val = pd.DataFrame(X.iloc[val_idx].values, columns=X.columns)
        
        # Treina os 3 especialistas
        xgb_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)
        cat_model.fit(X_train, y_train)
        
        # Coleta as predições limpas
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        
        del X_train, y_train, X_val
        gc.collect()

    logger.info(f"Matriz Base concluída em {time.time() - start_time:.1f}s.")
    
    # Avaliando os indivíduos apenas para logging (devem bater com a Fase 3)
    logger.info(f"   [Indivíduo] XGBoost OOF AMEX: {evaluate_model(y, oof_xgb)['AMEX_Score']:.4f}")
    logger.info(f"   [Indivíduo] LightGBM OOF AMEX: {evaluate_model(y, oof_lgb)['AMEX_Score']:.4f}")
    logger.info(f"   [Indivíduo] CatBoost OOF AMEX: {evaluate_model(y, oof_cat)['AMEX_Score']:.4f}")
    
    # =====================================================================
    # ETAPA 2: Soft Voting Classifier (Média Simples)
    # =====================================================================
    logger.info("\n--- Avaliando Soft Voting Classifier ---")
    # Soft voting nada mais é do que a média das probabilidades
    oof_voting = (oof_xgb + oof_lgb + oof_cat) / 3
    voting_metrics = evaluate_model(y, oof_voting)
    logger.info(f"🏆 AMEX Score (Voting): {voting_metrics['AMEX_Score']:.4f}")

    # =====================================================================
    # ETAPA 3: Stacking Classifier (Meta-Modelo via Regressão Logística)
    # =====================================================================
    logger.info("\n--- Avaliando Stacking Classifier ---")
    # Criamos um DataFrame onde cada coluna é a "opinião" de um modelo
    X_meta = pd.DataFrame({
        'xgb': oof_xgb,
        'lgb': oof_lgb,
        'cat': oof_cat
    })
    
    # O Meta-Modelo será uma Regressão Logística que aprenderá em quem confiar mais
    meta_model = LogisticRegression(random_state=RANDOM_SEED)
    
    # Obtemos as predições do Meta-Modelo usando validação cruzada nas "opiniões"
    oof_stacking = cross_val_predict(meta_model, X_meta, y, cv=skf, method='predict_proba')[:, 1]
    stacking_metrics = evaluate_model(y, oof_stacking)
    logger.info(f"🏆 AMEX Score (Stacking): {stacking_metrics['AMEX_Score']:.4f}")
    
    # =====================================================================
    # ETAPA 4: Blending Classifier (Holdout Fixo 80/20)
    # =====================================================================
    logger.info("\n--- Avaliando Blending Classifier ---")
    # Diferente do Stacking, o Blending divide a base apenas 1 vez (mais rápido, menos robusto)
    X_blend_train, X_blend_val, y_blend_train, y_blend_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    
    # Proteção de memória para o Mac
    X_blend_train = pd.DataFrame(X_blend_train.values, columns=X.columns)
    X_blend_val = pd.DataFrame(X_blend_val.values, columns=X.columns)
    
    # Treina na base 80
    xgb_model.fit(X_blend_train, y_blend_train)
    lgb_model.fit(X_blend_train, y_blend_train)
    cat_model.fit(X_blend_train, y_blend_train)
    
    # Prediz na base 20 para criar as features do meta-modelo
    blend_xgb = xgb_model.predict_proba(X_blend_val)[:, 1]
    blend_lgb = lgb_model.predict_proba(X_blend_val)[:, 1]
    blend_cat = cat_model.predict_proba(X_blend_val)[:, 1]
    
    X_meta_blend = pd.DataFrame({'xgb': blend_xgb, 'lgb': blend_lgb, 'cat': blend_cat})
    
    # Treina e prediz o Meta-Modelo
    meta_model.fit(X_meta_blend, y_blend_val)
    preds_blending = meta_model.predict_proba(X_meta_blend)[:, 1]
    blending_metrics = evaluate_model(y_blend_val, preds_blending)
    
    logger.info(f"🏆 AMEX Score (Blending): {blending_metrics['AMEX_Score']:.4f}")
    logger.warning("Nota: A métrica de Blending reflete apenas 20% da base (Holdout), "
                   "enquanto Voting e Stacking refletem a base inteira (OOF).")

    # =====================================================================
    # ETAPA 5: Salvamento de Resultados
    # =====================================================================
    results = [
        {"Modelo": "Voting Classifier", "AMEX Score": voting_metrics["AMEX_Score"], "ROC AUC": voting_metrics["ROC_AUC"]},
        {"Modelo": "Stacking Classifier", "AMEX Score": stacking_metrics["AMEX_Score"], "ROC AUC": stacking_metrics["ROC_AUC"]},
        {"Modelo": "Blending Classifier", "AMEX Score": blending_metrics["AMEX_Score"], "ROC AUC": blending_metrics["ROC_AUC"]},
    ]
    
    df_results = pd.DataFrame(results).sort_values(by="AMEX Score", ascending=False).reset_index(drop=True)
    output_path = RESULTS_DIR / "phase4_ensembles_ranking.csv"
    df_results.to_csv(output_path, index=False)
    
    logger.info("\n=========================================")
    logger.info(f"Fase 4 Concluída! Ranking dos Ensembles salvo em: {output_path}")
    logger.info("Resultado Final da Arquitetura:")
    for i, row in df_results.iterrows():
        logger.info(f" {i+1}º - {row['Modelo']} | AMEX: {row['AMEX Score']:.4f}")
    logger.info("=========================================")

if __name__ == "__main__":
    main()