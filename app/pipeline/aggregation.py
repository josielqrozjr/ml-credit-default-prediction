"""
Módulo de Agregação de Clientes (Polars)
----------------------------------------
Objetivo: Agregar a série temporal em uma linha por cliente.
Diferencial: Lida de forma inteligente com features numéricas, categóricas 
originais e as novas flags de mudança de estado geradas na etapa temporal.
"""

import polars as pl
import logging
from typing import List

logger = logging.getLogger(__name__)

class AgregadorClientePolars:
    def __init__(self, col_cliente: str = 'customer_ID', col_data: str = 'S_2'):
        self.col_cliente = col_cliente
        self.col_data = col_data
        
        # Lista OFICIAL de categóricas da AMEX
        self.categorical_features = [
            'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 
            'D_126', 'D_63', 'D_64', 'D_66', 'D_68'
        ]

    def _obter_expressoes_agregacao(self, colunas_originais: List[str]) -> List[pl.Expr]:
        """
        Constrói a lista de expressões matemáticas (Lazy) com base no tipo da feature.
        """
        expressoes = []
        colunas_ignoradas = [self.col_cliente, self.col_data]
        
        for col in colunas_originais:
            if col in colunas_ignoradas:
                continue
                
            # -------------------------------------------------------------
            # ROTA 1: Agregação de Categóricas Originais
            # -------------------------------------------------------------
            if col in self.categorical_features:
                expressoes.extend([
                    pl.col(col).first().alias(f"{col}_first"),
                    pl.col(col).last().alias(f"{col}_last"),
                    pl.col(col).n_unique().alias(f"{col}_nunique"),
                    pl.col(col).count().alias(f"{col}_count")
                ])
                
            # -------------------------------------------------------------
            # ROTA 2: Flags Binárias de Mudança (Geradas no DuckDB)
            # -------------------------------------------------------------
            elif col.endswith('_changed'):
                expressoes.extend([
                    pl.col(col).sum().alias(f"{col}_total_changes"),
                    pl.col(col).mean().alias(f"{col}_change_frequency"),
                    pl.col(col).last().alias(f"{col}_changed_recently")
                ])
                
            # -------------------------------------------------------------
            # ROTA 3: Agregação Numérica (Originais e _diff1)
            # -------------------------------------------------------------
            else:
                # Estatísticas Base
                expressoes.extend([
                    pl.col(col).mean().alias(f"{col}_mean"),
                    pl.col(col).std().alias(f"{col}_std"),
                    pl.col(col).min().alias(f"{col}_min"),
                    pl.col(col).max().alias(f"{col}_max"),
                    pl.col(col).last().alias(f"{col}_last")
                ])
                
                # --- TREND 1: Total Delta (Variação Absoluta) ---
                # Último valor menos o primeiro valor do cliente
                expressoes.append(
                    (pl.col(col).last() - pl.col(col).first()).alias(f"{col}_total_delta")
                )
                
                # --- TREND 2: Trend Ratio (Razão de Tendência) ---
                # Último valor dividido pela média. Somamos 1e-5 para evitar divisão por zero
                expressoes.append(
                    (pl.col(col).last() / (pl.col(col).mean() + 1e-5)).alias(f"{col}_trend_ratio")
                )
                
                # --- TREND 3: Positive Ratio (Frequência de Aumento) ---
                # Quantos % dos meses o valor foi MAIOR que o mês anterior
                # diff() gera nulos no primeiro mês, preenchemos com 0
                expressoes.append(
                    (pl.col(col).diff().fill_null(0.0) > 0).cast(pl.Float32).mean().alias(f"{col}_pos_ratio")
                )
                
                # --- TREND 4: Proxy de Linear Slope (Variação Média Mensal) ---
                # O Total Delta dividido pela quantidade de meses observados daquele cliente
                expressoes.append(
                    ((pl.col(col).last() - pl.col(col).first()) / pl.col(col).count()).alias(f"{col}_avg_monthly_slope")
                )

        return expressoes

    def transformar(self, df_lazy: pl.LazyFrame) -> pl.LazyFrame:
        """Executa a agregação utilizando a API Lazy do Polars."""
        logger.info("Construindo plano de execução Lazy do Polars para Agregação...")
        
        # Pega as colunas disponíveis no LazyFrame
        colunas_originais = df_lazy.collect_schema().names()
        
        expressoes = self._obter_expressoes_agregacao(colunas_originais)
        
        # O group_by no Polars + agg() aplica todas as expressões paralelamente em C++/Rust
        df_agregado = df_lazy.group_by(self.col_cliente).agg(expressoes)
        
        return df_agregado