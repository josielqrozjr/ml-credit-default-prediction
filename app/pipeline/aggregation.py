"""
Módulo de Agregação de Clientes com DuckDB
------------------------------------------
Objetivo: Gerar o SQL dinâmico para agregar a série temporal (múltiplas linhas)
em uma única linha por cliente, extraindo métricas estatísticas essenciais.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

class AgregadorCliente:
    def __init__(self, col_cliente: str = 'customer_ID', col_data: str = 'S_2'):
        self.col_cliente = col_cliente
        self.col_data = col_data
        self.categorical_features = [
            'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 
            'D_126', 'D_63', 'D_64', 'D_66', 'D_68'
        ]

    def gerar_sql_agregacao(self, nome_cte_temporal: str, colunas_originais: List[str]) -> str:
        """
        Constrói o SELECT de agregação (GROUP BY) baseado nos tipos de coluna.
        """
        logger.info("Gerando SQL para agregação (Group By)...")
        
        select_aggs = [self.col_cliente]
        
        colunas_ignoradas = [self.col_cliente, self.col_data]
        
        # 1. Agregações para colunas originais e categóricas
        for col in colunas_originais:
            if col in colunas_ignoradas:
                continue
                
            if col in self.categorical_features:
                # Categóricas: Último valor conhecido, contagem e valores únicos
                select_aggs.append(f"LAST_VALUE({col} IGNORE NULLS) OVER (PARTITION BY {self.col_cliente} ORDER BY {self.col_data} ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS {col}_last")
                select_aggs.append(f"COUNT({col}) AS {col}_count")
                select_aggs.append(f"COUNT(DISTINCT {col}) AS {col}_nunique")
            else:
                # Numéricas padrão: Média, min, max, desvio padrão e último valor
                select_aggs.append(f"AVG({col}) AS {col}_mean")
                select_aggs.append(f"MIN({col}) AS {col}_min")
                select_aggs.append(f"MAX({col}) AS {col}_max")
                select_aggs.append(f"STDDEV_SAMP({col}) AS {col}_std")
                select_aggs.append(f"LAST_VALUE({col} IGNORE NULLS) OVER (PARTITION BY {self.col_cliente} ORDER BY {self.col_data} ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS {col}_last")

        # 2. Agregações para as colunas de "diff1" (geradas na etapa temporal)
        # Vamos usar apenas colunas numéricas para os diffs
        features_numericas = [c for c in colunas_originais if c not in colunas_ignoradas and c not in self.categorical_features]
        for col in features_numericas:
            col_diff = f"{col}_diff1"
            select_aggs.append(f"AVG({col_diff}) AS {col_diff}_mean")
            # Pegando a última diferença (tendência mais recente)
            select_aggs.append(f"LAST_VALUE({col_diff} IGNORE NULLS) OVER (PARTITION BY {self.col_cliente} ORDER BY {self.col_data} ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS {col_diff}_last")

        sql_select = ",\n".join(select_aggs)
        
        # Note que como usamos Window Functions para pegar o "last_value", 
        # nós agrupamos e pegamos MAX(col_last) apenas para comprimir a tabela.
        # Alternativamente, DuckDB permite funções combinadas.
        
        query = f"""
        SELECT 
            {self.col_cliente},
            -- Envelopamos com MAX() e agrupamos para colapsar as linhas do cliente
            {", ".join([f"MAX({alias.split(' AS ')[1]}) AS {alias.split(' AS ')[1]}" if 'AS' in alias else alias for alias in select_aggs[1:]])}
        FROM (
            SELECT 
                {sql_select}
            FROM {nome_cte_temporal}
        ) sub
        GROUP BY {self.col_cliente}
        """
        return query