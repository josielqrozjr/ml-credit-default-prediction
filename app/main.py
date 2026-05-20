"""
Pipeline de Preparação de Dados com DuckDB
------------------------------------------
Este orquestrador extrai o esquema dos dados, monta a query dinâmica de
engenharia e agregação, e confia ao motor de banco de dados do DuckDB
a execução performática out-of-core.
"""

import os
import duckdb
import logging
from temporal_features import EngenhariaTemporal
from aggregation import AgregadorCliente

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    caminho_input = "../data/raw/parquet/train/data_train.parquet"
    caminho_output_dir = "../data/processed/"
    caminho_output = os.path.join(caminho_output_dir, "train_tabular_final.parquet")
    
    os.makedirs(caminho_output_dir, exist_ok=True)
    
    # Inicia conexão DuckDB em memória
    conn = duckdb.connect(':memory:')
    
    # ---------------------------------------------------------
    # 1. Extração do Esquema (Schema) do Parquet
    # ---------------------------------------------------------
    logger.info(f"Lendo metadados do arquivo parquet: {caminho_input}")
    try:
        # Apenas descrevemos o arquivo para pegar os nomes das colunas (muito rápido)
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{caminho_input}')").df()
        colunas_originais = schema_df['column_name'].tolist()
        logger.info(f"Esquema carregado. {len(colunas_originais)} colunas encontradas.")
    except Exception as e:
        logger.exception(f"Erro ao ler metadados do parquet: {e}")
        return

    # ---------------------------------------------------------
    # 2. Geração da Query SQL
    # ---------------------------------------------------------
    # Tabela virtual lida diretamente pelo DuckDB
    tabela_leitura = f"read_parquet('{caminho_input}')"
    
    engenheiro = EngenhariaTemporal()
    agregador = AgregadorCliente()
    
    # Constrói as partes da query
    sql_temporal = engenheiro.gerar_sql_temporal(tabela_origem=tabela_leitura, colunas_totais=colunas_originais)
    sql_agregacao = agregador.gerar_sql_agregacao(nome_cte_temporal="cte_temporal", colunas_originais=colunas_originais)
    
    # Query Final unificando tudo em CTEs (Common Table Expressions)
    query_final = f"""
    WITH cte_temporal AS (
        {sql_temporal}
    )
    -- O DuckDB vai processar o pipeline abaixo e copiar direto para o novo Parquet
    COPY (
        {sql_agregacao}
    ) TO '{caminho_output}' (FORMAT PARQUET);
    """

    # ---------------------------------------------------------
    # 3. Execução no Motor do DuckDB
    # ---------------------------------------------------------
    logger.info("Iniciando execução vetorizada via DuckDB. Isso pode levar alguns minutos dependendo do disco...")
    try:
        conn.execute(query_final)
        logger.info(f"Sucesso! Dataset processado salvo em: {caminho_output}")
    except Exception as e:
        logger.exception(f"Erro durante a execução do pipeline DuckDB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()