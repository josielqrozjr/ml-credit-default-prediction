# ============================================================================
# CÉLULA 1: CONFIGURAÇÃO DO AMBIENTE
# ============================================================================

import os
import sys
import warnings
import logging
from datetime import datetime
import random
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Bibliotecas de feature selection e ML
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb
import duckdb

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configurações gerais
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# 1. CONFIGURAÇÃO DE SEEDS PARA REPRODUTIBILIDADE
# ============================================================================
SEED = 42

def set_seeds(seed=SEED):
    """Define seeds em todas as bibliotecas para reprodutibilidade"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(SEED)

# ============================================================================
# 2. DETECÇÃO E CONFIGURAÇÃO DE GPU (COM VERIFICAÇÃO ROBUSTA)
# ============================================================================
# Verificação robusta com múltiplas tentativas
def check_cuda_available():
    """Verificação robusta de CUDA com debug info detalhado"""
    
    try:
        # [1] Tentar método direto PyTorch
        print("  [Debug] Testando torch.cuda.is_available()...")
        if torch.cuda.is_available():
            print("    Detectado via torch.cuda.is_available()")
            return True
        
        # [2] Tentar resetar cache
        print("  [Debug] Tentando resetar cache CUDA...")
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print("    Detectado após reset de cache")
            return True
        
        # [3] Tentar obter device atual
        print("  [Debug] Tentando obter current_device()...")
        try:
            device_id = torch.cuda.current_device()
            print(f"    Device encontrado: {device_id}")
            return True
        except Exception as e:
            print(f"    current_device() falhou: {e}")
        
        # [4] Verificar se PyTorch foi compilado com CUDA
        print("  [Debug] Verificando compilação PyTorch...")
        print(f"    torch.version.cuda: {torch.version.cuda}")
        print(f"    torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
        
        if torch.version.cuda is not None:
            print(f"    PyTorch compilado com CUDA {torch.version.cuda}")
        
        # [5] Tentar criar tensor em GPU manualmente
        print("  [Debug] Testando criação de tensor em GPU...")
        try:
            test_tensor = torch.randn(1, device='cuda')
            print("    Tensor criado com sucesso em GPU")
            return True
        except Exception as e:
            print(f"    Criação de tensor falhou: {e}")
        
        # [6] Verificar nvidia-smi
        print("  [Debug] Verificando nvidia-smi...")
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_count = int(result.stdout.strip())
                print(f"    nvidia-smi encontrou {gpu_count} GPU(s)")
                if gpu_count > 0:
                    print("    → GPU detectada via nvidia-smi! Tentando reinicializar PyTorch...")
                    # Tentar reinicializar CUDA após confirmação via nvidia-smi
                    try:
                        torch.cuda.init()
                        if torch.cuda.is_available():
                            print("    CUDA ativado após init()")
                            return True
                    except Exception as ex:
                        print(f"    init() falhou: {ex}")
        except Exception as e:
            print(f"    nvidia-smi falhou: {e}")
        
        # [7] Tentar CuPy (alternativa ao PyTorch)
        print("  [Debug] Testando CuPy...")
        try:
            import cupy as cp
            cp.cuda.Device(0)
            print(f"    CuPy detectou GPU com sucesso!")
            print("    → GPU está disponível (CuPy confirmou)")
            return True
        except Exception as e:
            print(f"    CuPy falhou: {e}")
        
        # [8] Último resort: forçar device CUDA
        print("  [Debug] Tentativa final: forçar device CUDA...")
        try:
            test_device = torch.device('cuda')
            print(f"    device('cuda') criado: {test_device}")
            # Tentar usar o device
            _ = torch.zeros(1, device=test_device)
            print("    Tensor alocado com sucesso")
            return True
        except Exception as e:
            print(f"    Forçar device falhou: {e}")
            return False
    except Exception as e:
        print(f"  Erro geral ao verificar CUDA: {e}")
        return False

# Executar verificação
print("\n🔍 Iniciando verificação de GPU...")
GPU_AVAILABLE = check_cuda_available()

# Se CUDA foi detectado, configurar device
if GPU_AVAILABLE:
    try:
        DEVICE = torch.device("cuda")
    except Exception:
        DEVICE = torch.device("cpu")
        GPU_AVAILABLE = False
else:
    DEVICE = torch.device("cpu")

print("=" * 80)
print("CONFIGURAÇÃO DE AMBIENTE - FEATURE SELECTION PIPELINE")
print("=" * 80)
print(f"\n✓ Dispositivo: {DEVICE.type.upper()}")
print(f"GPU Disponível: {'SIM' if GPU_AVAILABLE else 'NÃO'}")

if GPU_AVAILABLE:
    try:
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
        print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        print(f"⚠ Erro ao obter detalhes GPU: {e}")
else:
    print("⚠ GPU não detectada. Usando CPU (operações serão mais lentas)")

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"LightGBM version: {lgb.__version__}")
print(f"SHAP version: {shap.__version__}")

# ============================================================================
# 3. CONFIGURAÇÃO DE LOGGING
# ============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"feature_selection_{timestamp}.log")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("INÍCIO DO PIPELINE DE FEATURE SELECTION")
logger.info("="*80)
logger.info(f"Device: {DEVICE}")
logger.info(f"GPU Available: {GPU_AVAILABLE}")
logger.info(f"Seed: {SEED}")

# ============================================================================
# 4. INICIALIZAÇÃO DE DUCKDB
# ============================================================================
# Usar DuckDB em-memória para operações rápidas
conn_duckdb = duckdb.connect(':memory:')
logger.info("DuckDB inicializado em memória")

# ============================================================================
# 5. CONFIGURAÇÃO DE MATPLOTLIB E SEABORN
# ============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configurar tamanho padrão de figuras
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

logger.info("Matplotlib e Seaborn configurados")

# ============================================================================
# 6. VARIÁVEIS GLOBAIS DE CONFIGURAÇÃO
# ============================================================================
# Thresholds e parâmetros do pipeline
CONFIG = {
    'correlation_threshold': 0.95,  # Para remover features altamente correlacionadas
    'missing_threshold': 0.90,       # Features com > 90% missing serão removidas
    'mutual_info_percentile': 75,    # Top 75% para seleção por MI
    'lgb_importance_percentile': 80, # Top 80% para seleção por LightGBM
    'lightgbm_device': 'gpu' if GPU_AVAILABLE else 'cpu',
    'seed': SEED,
}

print(f"\n✓ Configurações do pipeline:")
for key, value in CONFIG.items():
    print(f"  - {key}: {value}")

logger.info(f"Pipeline config: {CONFIG}")
print("\n" + "="*80)
print("AMBIENTE PRONTO PARA EXECUÇÃO")
print("="*80)


# ============================================================================
# CÉLULA 2: LEITURA E PRÉ-PROCESSAMENTO INICIAL
# ============================================================================

# ============================================================================
# 1. LEITURA DE DADOS (AJUSTE O CAMINHO CONFORME NECESSÁRIO)
# ============================================================================
# IMPORTANTE: Substitua os paths conforme seu ambiente
data_path = r"C:\Users\joaov\Workspace\4 ano\tcc\ml-credit-default-prediction\data\raw\parquet\train\data_*.parquet"  # Ajuste conforme necessário
labels_path = r"C:\Users\joaov\Workspace\4 ano\tcc\ml-credit-default-prediction\data\raw\parquet\train_labels\data_*.parquet"

# Opção 1: Ler com DuckDB (recomendado para grandes volumes)
try:
    df = conn_duckdb.execute(f"SELECT * FROM parquet_scan('{data_path}')").df()
    logger.info(f"Dados carregados via DuckDB: {data_path}")
except Exception as e:
    # Fallback para pandas se DuckDB falhar
    logger.warning(f"Fallback para pandas: {e}")
    df = pd.read_parquet(data_path)

# Carregar labels
try:
    labels = conn_duckdb.execute(f"SELECT * FROM parquet_scan('{labels_path}')").df()
    logger.info(f"Labels carregados via DuckDB: {labels_path}")
except Exception as e:
    logger.warning(f"Fallback para pandas (labels): {e}")
    labels = pd.read_parquet(labels_path)

# Merge com labels (ajuste as colunas conforme sua base)
# IMPORTANTE: Adapte os nomes de colunas (ex: 'customer_id' pode variar)
try:
    df = df.merge(labels, on=['customer_id', 'month_id'], how='left')
    target_col = 'target'  # Ajuste conforme necessário
    logger.info("Labels mesclados ao dataset")
except Exception as e:
    logger.warning(f"Merge automático falhou: {e}")
    target_col = 'target'  # Defina manualmente se necessário

print("=" * 80)
print("PRÉ-PROCESSAMENTO INICIAL")
print("=" * 80)
print(f"\n📊 Dataset Original:")
print(f"  Shape: {df.shape}")
print(f"  Colunas: {df.columns.tolist()[:10]}... (total: {len(df.columns)})")
print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# ============================================================================
# 2. SEPARAÇÃO DE TARGET E FEATURES
# ============================================================================
# Identificar coluna de target
if target_col not in df.columns:
    logger.error(f"Target '{target_col}' não encontrado. Colunas: {df.columns.tolist()}")
    raise ValueError(f"Target '{target_col}' não encontrado")

y = df[[target_col]].copy()
X = df.drop(columns=[target_col]).copy()

logger.info(f"Target shape: {y.shape}, Features shape: {X.shape}")
print(f"\n🎯 Target:")
print(f"  Distribuição: {y[target_col].value_counts().to_dict()}")
print(f"  Taxa de positivos: {(y[target_col].sum() / len(y) * 100):.2f}%")

# ============================================================================
# 3. IDENTIFICAÇÃO DE FEATURES PROBLEMÁTICAS
# ============================================================================
features_to_remove = {}
final_features = X.columns.tolist()

print(f"\n🔍 Análise de Features Problemáticas:")

# 3.1 REMOVER IDs (alta cardinalidade)
print(f"\n  [1] Identificando IDs (cardinalidade = n_amostras)...")
id_features = []
for col in X.columns:
    if X[col].nunique() == len(X):
        id_features.append(col)
        features_to_remove[col] = "ID (cardinalidade = n_amostras)"

if id_features:
    print(f"      Removidas {len(id_features)} IDs: {id_features[:5]}")
    logger.info(f"IDs removidas: {id_features}")
else:
    print(f"      Nenhum ID encontrado")

final_features = [f for f in final_features if f not in id_features]

# 3.2 REMOVER CONSTANTES (variância = 0)
print(f"\n  [2] Identificando features constantes...")
const_features = []
for col in final_features:
    if X[col].nunique() == 1:
        const_features.append(col)
        features_to_remove[col] = "Constante (único valor)"

if const_features:
    print(f"      Removidas {len(const_features)} constantes: {const_features}")
    logger.info(f"Features constantes removidas: {const_features}")
else:
    print(f"      Nenhuma feature constante encontrada")

final_features = [f for f in final_features if f not in const_features]

# 3.3 REMOVER QUASE-CONSTANTES (variância muito baixa)
print(f"\n  [3] Identificando features quase-constantes...")
quasi_const_features = []
for col in final_features:
    try:
        # Calcular razão da classe mais frequente
        value_counts = X[col].value_counts()
        if len(value_counts) > 0:
            freq_ratio = value_counts.iloc[0] / len(X)
            if freq_ratio > 0.99:  # 99% da mesma classe
                quasi_const_features.append(col)
                features_to_remove[col] = f"Quase-constante (freq={freq_ratio:.2%})"
    except Exception as e:
        logger.debug(f"Erro ao calcular frequência de {col}: {e}")

if quasi_const_features:
    print(f"      Removidas {len(quasi_const_features)} quase-constantes")
    logger.info(f"Features quase-constantes: {quasi_const_features}")
else:
    print(f"      Nenhuma feature quase-constante encontrada")

final_features = [f for f in final_features if f not in quasi_const_features]

# 3.4 REMOVER FEATURES COM MISSING > THRESHOLD
print(f"\n  [4] Identificando features com missing > {CONFIG['missing_threshold']*100:.0f}%...")
missing_features = []
missing_stats = {}

for col in final_features:
    missing_pct = X[col].isna().sum() / len(X)
    missing_stats[col] = missing_pct
    
    if missing_pct > CONFIG['missing_threshold']:
        missing_features.append(col)
        features_to_remove[col] = f"Missing > {CONFIG['missing_threshold']*100:.0f}% ({missing_pct:.2%})"

if missing_features:
    print(f"      Removidas {len(missing_features)} features com missing extremo")
    for feat in missing_features[:5]:
        print(f"        - {feat}: {missing_stats[feat]:.2%} missing")
    logger.info(f"Features com missing extremo: {missing_features}")
else:
    print(f"      Nenhuma feature com missing extremo")

final_features = [f for f in final_features if f not in missing_features]

# ============================================================================
# 4. DATASET APÓS PRÉ-PROCESSAMENTO
# ============================================================================
X_clean = X[final_features].copy()

print(f"\n" + "="*80)
print(f"📈 Resumo do Pré-Processamento:")
print(f"="*80)
print(f"\n  Features iniciais: {X.shape[1]}")
print(f"  Features removidas: {X.shape[1] - X_clean.shape[1]}")
print(f"  Features finais: {X_clean.shape[1]}")
print(f"  Redução: {((X.shape[1] - X_clean.shape[1]) / X.shape[1] * 100):.1f}%")

# ============================================================================
# 5. RELATÓRIO DETALHADO DE FEATURES REMOVIDAS
# ============================================================================
if features_to_remove:
    print(f"\n📋 Features Removidas:")
    removal_df = pd.DataFrame([
        {'Feature': feat, 'Motivo': reason}
        for feat, reason in features_to_remove.items()
    ])
    print(removal_df.to_string(index=False))
    
    # Salvar relatório
    report_path = f"./reports/removed_features_{timestamp}.csv"
    os.makedirs("./reports", exist_ok=True)
    removal_df.to_csv(report_path, index=False)
    logger.info(f"Relatório de features removidas salvo em: {report_path}")

# ============================================================================
# 6. ESTATÍSTICAS DE MISSING
# ============================================================================
print(f"\n📊 Top 10 Features com Maior Missing (remanescentes):")
missing_remaining = {col: missing_stats.get(col, X_clean[col].isna().sum() / len(X_clean)) 
                     for col in X_clean.columns}
missing_sorted = sorted(missing_remaining.items(), key=lambda x: x[1], reverse=True)[:10]
for feat, pct in missing_sorted:
    print(f"  {feat}: {pct:.2%}")

# ============================================================================
# 7. GUARDAR VARIÁVEIS PARA PRÓXIMAS CÉLULAS
# ============================================================================
print(f"\n✓ Dataset preparado para feature selection")
logger.info(f"Dataset pronto: X_clean.shape={X_clean.shape}, y.shape={y.shape}")
print("="*80)