"""
Módulo de Engenharia Temporal com DuckDB
----------------------------------------
Objetivo: Gerar o SQL dinâmico para calcular a diferença (tendência) 
entre a fatura atual e a anterior usando funções de janela (Window Functions).
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
        Gera a query SQL que calcula a diferença do mês atual para o anterior (LAG).
        """
        logger.info("Gerando SQL para features temporais (Window Functions)...")
        
        features_numericas = self.obter_colunas_numericas(colunas_totais)
        
        # Seleciona todas as colunas originais
        select_statements = ["*"]
        
        # Adiciona o cálculo de diff para cada variável numérica
        for col in features_numericas:
            # Fórmula: Valor Atual - Valor do Mês Anterior (LAG)
            statement = f"""
            ({col} - LAG({col}) OVER (
                PARTITION BY {self.col_cliente} 
                ORDER BY {self.col_data}
            )) AS {col}_diff1
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