"""
Pipeline de Merge e Split (Polars)
----------------------------------
Orquestra a união do dataset tabular com os labels e divide a base
em Treino (80%) e Validação (20%) de forma estratificada.
"""

import os
import polars as pl
import logging
from pipeline.merge import MescladorLabels
from pipeline.split import SplitterEstratificado

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # ---------------------------------------------------------
    # CONFIGURAÇÃO DE CAMINHOS
    # ---------------------------------------------------------
    # Dataset processado na etapa anterior (features numéricas/agregadas)
    caminho_features = "./data/processed/train_tabular_final.parquet"
    
    # Observe que usamos o asterisco para pegar todas as partições de labels, como você pontuou
    caminho_labels_glob = "./data/raw/parquet/train_labels/data_*.parquet"
    
    # Caminhos de saída
    caminho_output_dir = "./data/processed/split/"
    caminho_train = os.path.join(caminho_output_dir, "train_80.parquet")
    caminho_val = os.path.join(caminho_output_dir, "valid_20.parquet")
    
    os.makedirs(caminho_output_dir, exist_ok=True)

    # =========================================================
    # ETAPA 1: MERGE (LAZY)
    # =========================================================
    logger.info("=== INICIANDO MERGE COM LABELS ===")
    mesclador = MescladorLabels()
    lazy_merged = mesclador.mesclar(features_path=caminho_features, labels_path=caminho_labels_glob)
    
    # Como a próxima etapa (split estratificado) precisa do dataset inteiro na memória
    # para embaralhar as linhas, chamamos o .collect() para disparar o cálculo do Polars
    logger.info("Carregando resultado do merge em memória...")
    df_completo = lazy_merged.collect()
    logger.info(f"Dataset consolidado em memória. Shape total: {df_completo.shape}")

    # =========================================================
    # ETAPA 2: SPLIT ESTRATIFICADO
    # =========================================================
    logger.info("=== INICIANDO SPLIT ESTRATIFICADO ===")
    splitter = SplitterEstratificado(test_size=0.2, seed=42)
    df_train, df_val = splitter.separar(df_completo)

    # =========================================================
    # ETAPA 3: EXPORTAÇÃO
    # =========================================================
    logger.info("Salvando conjuntos em disco...")
    df_train.write_parquet(caminho_train)
    df_val.write_parquet(caminho_val)
    
    logger.info(f"Arquivos salvos em: {caminho_output_dir}")
    logger.info("=== PIPELINE CONCLUÍDO COM SUCESSO ===")

if __name__ == "__main__":
    main()