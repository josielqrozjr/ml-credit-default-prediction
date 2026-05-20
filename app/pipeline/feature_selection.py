"""
Módulo de Feature Selection (AMEX Contextualized)
-------------------------------------------------
Objetivo: Redução de dimensionalidade inteligente.
Filosofia: Deixar o algoritmo (LightGBM) decidir o que é importante.
Filtros estáticos apenas para lixo absoluto (100% nulo ou constantes).

Para este dataset, a melhor abordagem de balanceamento não é alterar os dados, 
mas sim penalizar o algoritmo. Usaremos o hiperparâmetro scale_pos_weight (ou is_unbalance=True)
dentro do próprio LightGBM no momento do treino final. Ele fará com que a árvore dê um "peso"
matemático maior para o erro cometido em um cliente inadimplente.
"""

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from typing import List

logger = logging.getLogger(__name__)

class SelecionadorFeaturesAMEX:
    def __init__(self, col_target: str = 'target', col_cliente: str = 'customer_ID', seed: int = 42):
        self.col_target = col_target
        self.col_cliente = col_cliente
        self.seed = seed
        
        # Thresholds ultra-permissivos (Filosofia AMEX)
        self.missing_threshold = 0.999  # Só remove se for 100% nulo
        self.variance_threshold = 0.999 # Só remove se o mesmo valor repetir em 99.9% das linhas
        self.correlation_threshold = 0.98 # Permite alta colinearidade, remove apenas clones quase exatos
        self.top_lgbm_features = 400    # Top 400 carrega 99% da informação no dataset AMEX

    def _filtro_estatico(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove apenas colunas que não carregam nenhuma informação matemática."""
        logger.info("Aplicando Filtros Estáticos Ultra-Permissivos...")
        n_linhas = df.height
        colunas_iniciais = df.columns
        colunas_manter = [self.col_cliente, self.col_target]

        for col in colunas_iniciais:
            if col in colunas_manter:
                continue
                
            # Filtro 1: Nulos (só cai se for quase 100% vazio)
            pct_null = df[col].null_count() / n_linhas
            if pct_null > self.missing_threshold:
                continue
                
            # Filtro 2: Quase-constante
            try:
                modo_freq = df[col].value_counts().sort("count", descending=True)[0, "count"]
                if (modo_freq / n_linhas) > self.variance_threshold:
                    continue
            except Exception:
                pass # Tratamento para tipos complexos, se houver
                
            colunas_manter.append(col)

        logger.info(f"Filtro estático: reduzido de {len(colunas_iniciais)} para {len(colunas_manter)} colunas.")
        return df.select(colunas_manter)

    def _filtro_lightgbm(self, df: pd.DataFrame) -> List[str]:
        """Deixa a árvore decidir o que importa, inclusive como tratar nulos."""
        logger.info("Aplicando Filtro de Importância (LightGBM)...")
        
        features = [c for c in df.columns if c not in [self.col_cliente, self.col_target]]
        X = df[features]
        y = df[self.col_target]

        # Configuração para AMEX: Lida com classes desbalanceadas nativamente
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 127,      # Árvores mais profundas para capturar interações raras
            'is_unbalance': True,   # Balanceamento nativo do LightGBM
            'verbose': -1,
            'seed': self.seed,
            'n_jobs': -1
        }

        train_data = lgb.Dataset(X, label=y)
        modelo = lgb.train(lgb_params, train_data, num_boost_round=200)

        importancias = modelo.feature_importance(importance_type='gain')
        
        df_imp = pd.DataFrame({'feature': features, 'importance': importancias})
        df_imp = df_imp.sort_values(by='importance', ascending=False)
        
        # Filtra features com Ganho ZERO absoluto
        df_imp = df_imp[df_imp['importance'] > 0]
        
        features_selecionadas = df_imp.head(self.top_lgbm_features)['feature'].tolist()
        
        logger.info(f"LightGBM manteve {len(features_selecionadas)} features com alto poder preditivo.")
        return [self.col_cliente, self.col_target] + features_selecionadas

    def selecionar(self, df_train: pl.DataFrame) -> pl.DataFrame:
        """Orquestra a seleção AMEX-first."""
        df_clean_pl = self._filtro_estatico(df_train)
        
        logger.info("Convertendo para Pandas para modelagem com LightGBM...")
        df_clean_pd = df_clean_pl.to_pandas()
        
        colunas_finais = self._filtro_lightgbm(df_clean_pd)
        
        logger.info("Salvando lista de features selecionadas no Polars...")
        return df_train.select(colunas_finais)