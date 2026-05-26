"""
Módulo de Split Estratificado (Polars)
--------------------------------------
Objetivo: Separar o dataset em Treino e Validação (80/20) garantindo que
a proporção da classe minoritária (target=1) seja a mesma em ambos os conjuntos.
"""

import polars as pl
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class SplitterEstratificado:
    def __init__(self, col_target: str = 'target', test_size: float = 0.2, seed: int = 42):
        self.col_target = col_target
        self.test_size = test_size
        self.seed = seed

    def separar(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Aplica um split estratificado usando funções de janela (Over) do Polars.
        """
        logger.info(f"Executando split estratificado ({1 - self.test_size:.0%} / {self.test_size:.0%})...")
        
        # A lógica mágica do Polars:
        # 1. Agrupa pelas classes (target 0 e 1).
        # 2. Gera números sequenciais e embaralha (shuffle) dentro de cada classe.
        # 3. Calcula onde fica a "linha de corte" (20%) dentro de cada classe.
        df_com_ranks = df.with_columns(
            pl.int_range(0, pl.len())
            .shuffle(seed=self.seed)
            .over(self.col_target)
            .alias("rand_rank"),
            
            (pl.len().over(self.col_target) * self.test_size).alias("threshold")
        )

        # Tudo abaixo do threshold (20%) vai para Validação/Teste
        df_val = (
            df_com_ranks
            .filter(pl.col("rand_rank") < pl.col("threshold"))
            .drop(["rand_rank", "threshold"])
        )
        
        # Tudo acima do threshold (80%) vai para Treino
        df_train = (
            df_com_ranks
            .filter(pl.col("rand_rank") >= pl.col("threshold"))
            .drop(["rand_rank", "threshold"])
        )

        # Logs de Validação da Estratificação
        train_target_mean = df_train[self.col_target].mean()
        val_target_mean = df_val[self.col_target].mean()
        
        logger.info(f"Shape Treino: {df_train.shape} | Prop. Target: {train_target_mean:.4%}")
        logger.info(f"Shape Valid:  {df_val.shape} | Prop. Target: {val_target_mean:.4%}")

        return df_train, df_val