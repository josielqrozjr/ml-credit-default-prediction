"""
Módulo de Engenharia Temporal com DuckDB
----------------------------------------
Objetivo: Extrair a tendência sequencial dos dados.
- Para Numéricas: Calcula a diferença matemática do mês atual para o anterior.
- Para Categóricas: Calcula uma flag binária se houve mudança de status no mês.
"""

import duckdb
import logging
from typing import List

logger = logging.getLogger(__name__)

class EngenhariaTemporal:
    def __init__(self, col_cliente: str = 'customer_ID', col_data: str = 'S_2'):
        self.col_cliente = col_cliente
        self.col_data = col_data
        self.categorical_features = [
            'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 
            'D_126', 'D_63', 'D_64', 'D_66', 'D_68'
        ]

    def obter_colunas_numericas(self, colunas_totais: List[str]) -> List[str]:
        """Filtra apenas as colunas numéricas (ignora datas, IDs e categóricas)."""
        colunas_ignoradas = [self.col_cliente, self.col_data] + self.categorical_features
        return [col for col in colunas_totais if col not in colunas_ignoradas]

    def gerar_sql_temporal(self, tabela_origem: str, colunas_totais: List[str]) -> str:
        """
        Gera a query SQL com as Window Functions para ambas as tipagens.
        """
        logger.info("Gerando SQL para features temporais (Window Functions)...")
        
        features_numericas = self.obter_colunas_numericas(colunas_totais)
        # Identifica as categóricas que realmente estão presentes no dataset
        features_categoricas = [col for col in colunas_totais if col in self.categorical_features]
        
        # Seleciona todas as colunas originais (a base da nossa tabela)
        select_statements = ["*"]
        
        # =========================================================
        # 1. TENDÊNCIA NUMÉRICA (Diferença)
        # =========================================================
        for col in features_numericas:
            statement = f"""
            ({col} - LAG({col}) OVER (
                PARTITION BY {self.col_cliente} 
                ORDER BY {self.col_data}
            )) AS {col}_diff1
            """
            select_statements.append(statement)

        # =========================================================
        # 2. TRANSIÇÃO CATEGÓRICA (Mudança de Estado)
        # =========================================================
        for col in features_categoricas:
            # Lógica: Se o mês anterior for nulo (primeiro mês), a flag é 0.
            # Se o valor for diferente do mês anterior, a flag é 1 (mudou).
            # Do contrário, a flag é 0 (manteve o status).
            statement = f"""
            CASE 
                WHEN LAG({col}) OVER (PARTITION BY {self.col_cliente} ORDER BY {self.col_data}) IS NULL THEN 0
                WHEN {col} != LAG({col}) OVER (PARTITION BY {self.col_cliente} ORDER BY {self.col_data}) THEN 1
                ELSE 0
            END AS {col}_changed
            """
            select_statements.append(statement)
            
        sql_select = ",\n".join(select_statements)
        
        # Montamos a query como uma CTE (Common Table Expression)
        query = f"""
        SELECT 
            {sql_select}
        FROM {tabela_origem}
        """
        return query