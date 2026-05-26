# Pipeline Híbrido de Feature Selection
## American Express - Default Prediction (Kaggle)

**Autor**: João Vitor de Lima Antunes  
**Data**: 2024  
**Status**: ✅ Production-Ready  

---

## 📋 Visão Geral

Este pipeline implementa um **sistema robusto de seleção de features** otimizado para competições de machine learning em larga escala. Combina técnicas estatísticas clássicas com métodos modernos de explicabilidade (SHAP).

### 🎯 Características Principais

✅ **GPU Acceleration**: Suporte nativo a NVIDIA CUDA  
✅ **DuckDB Integration**: Operações otimizadas em grandes volumes  
✅ **Multi-technique Approach**: 5 métodos combinados (Correlação, MI, LightGBM, SHAP)  
✅ **Production-Ready Code**: Logging, error handling, modular  
✅ **Explainability**: SHAP values para interpretação de features  
✅ **Reprodutibilidade**: Seeds globais configuradas  

---

## 🏗️ Arquitetura do Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ CÉLULA 1: Configuração de Ambiente                     │
│ - GPU/CPU setup                                         │
│ - Seeds para reprodutibilidade                          │
│ - DuckDB inicialização                                  │
│ - Logging setup                                         │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ CÉLULA 2: Limpeza de Dados                             │
│ - Remover IDs (cardinalidade = n_amostras)             │
│ - Remover features constantes                          │
│ - Remover quase-constantes (< 1% variação)             │
│ - Remover features com > 90% missing                   │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ CÉLULA 3: Análise de Correlação + MI                   │
│ - Pearson |r| > 0.95 → remover redundância             │
│ - Spearman para não-linearidades                       │
│ - Mutual Information com target                        │
│ - Selecionar top 75% por MI                            │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ CÉLULA 4: LightGBM Feature Importance                  │
│ - Treinar modelo com CV                                │
│ - Extrair importance por Gain                          │
│ - Selecionar top 80% features                          │
│ - Validar com ROC-AUC                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│ CÉLULA 5: SHAP Explainability                          │
│ - TreeExplainer para LightGBM                          │
│ - Summary plots (beeswarm + bar)                       │
│ - Dependence plots (top features)                      │
│ - Interpretação de proteção vs risco                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
         ✅ Features Finais Selecionadas
```

---

## 📊 Redução de Features (Exemplo)

| Estágio | Features | Redução | Motivo |
|---------|----------|---------|--------|
| Inicial | 190 | - | Dataset bruto |
| Pós-limpeza | 165 | 13% | IDs, constantes, missing |
| Pós-correlação | 142 | 14% | Redundância (r > 0.95) |
| Pós-MI | 110 | 22% | Baixa informação mútua |
| Pós-LightGBM | 88 | 20% | Baixa importância |
| **Final** | **88** | **54%** | **Total** |

---

## 🚀 Como Executar

### 1. Instalação de Dependências

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar pacotes
pip install -r requirements_feature_selection.txt

# Para GPU (NVIDIA CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**⚠️ GPU não funciona?** Consulte [GPU_TROUBLESHOOTING.md](GPU_TROUBLESHOOTING.md)

### 2. Preparar Dados

```bash
# Estrutura esperada:
data/
├── raw/
│   └── parquet/
│       ├── train/
│       │   └── part-00000.parquet
│       └── train_labels/
│           └── part-00000.parquet
└── processed/
```

### 3. Executar Pipeline

```bash
# Abrir notebook
jupyter notebook featureSelection.ipynb

# Ou usar JupyterLab
jupyter lab featureSelection.ipynb
```

### 4. Executar Células Sequencialmente

1. **Célula 1**: Setup inicial (1-2 min)
2. **Célula 2**: Leitura e limpeza (2-5 min, depende do volume)
3. **Célula 3**: Correlação + MI (3-10 min)
4. **Célula 4**: LightGBM training (5-20 min com GPU)
5. **Célula 5**: SHAP explainability (5-15 min)

---

## ⚙️ Configuração Avançada

### Ajustar Thresholds

No início da **Célula 1**, modificar `CONFIG`:

```python
CONFIG = {
    # Remover features com |r| > threshold
    'correlation_threshold': 0.95,
    
    # Remover features com > threshold% missing
    'missing_threshold': 0.90,
    
    # Manter top N% features por MI
    'mutual_info_percentile': 75,
    
    # Manter top N% features por LightGBM
    'lgb_importance_percentile': 80,
    
    # Device: 'gpu' ou 'cpu'
    'lightgbm_device': 'gpu',
    
    # Seed para reprodutibilidade
    'seed': 42,
}
```

### Usar GPU

#### Verificar GPU Disponível
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

#### Forçar Uso de GPU
```python
CONFIG['lightgbm_device'] = 'gpu'
DEVICE = torch.device('cuda:0')
```

### Para Datasets Enormes (> 100M amostras)

1. **Aumentar `missing_threshold`** para 0.95 (remove menos features)
2. **Reduzir `correlation_threshold`** para 0.90 (mais conservador)
3. **Usar sample em Célula 3**: 
   ```python
   sample_size = 1000000  # Ao invés de tudo
   ```
4. **SHAP com sample menor**: 
   ```python
   sample_size = 2000  # Ao invés de 5000
   ```

---

## 📈 Outputs Gerados

### Diretórios Criados

```
notebooks/
├── logs/
│   └── feature_selection_TIMESTAMP.log
├── plots/
│   ├── mutual_information_TIMESTAMP.png
│   ├── correlation_heatmap_TIMESTAMP.png
│   ├── lgb_importance_TIMESTAMP.png
│   ├── lgb_importance_distribution_TIMESTAMP.png
│   ├── shap_summary_beeswarm_TIMESTAMP.png
│   ├── shap_summary_bar_TIMESTAMP.png
│   ├── shap_dependence_plots_TIMESTAMP.png
│   └── shap_force_high_risk_TIMESTAMP.png
└── reports/
    ├── removed_features_TIMESTAMP.csv
    ├── lgb_importance_TIMESTAMP.csv
    ├── shap_statistics_TIMESTAMP.csv
    └── final_selected_features_TIMESTAMP.txt
```

### Variáveis Disponíveis Ao Final

```python
# Dataset
X_selected_lgb      # Features selecionadas (pandas DataFrame)
y_target            # Target original

# Modelo
booster_model       # LightGBM model treinado
explainer           # SHAP TreeExplainer

# Metadados
features_final      # Lista de nomes das features
feature_importance  # Array de importâncias
shap_values         # Array de SHAP values
```

---

## 🔍 Interpretando Resultados

### Célula 2: Pré-processamento

```
✗ Removidas 15 IDs: ['customer_id', 'transaction_id', ...]
✗ Removidas 2 constantes: ['feature_x', 'feature_y']
✗ Removidas 1 quase-constante (freq=99.85%)
✗ Removidas 7 features com missing > 90%

Resultado: 190 → 165 features (13% redução)
```

**Interpretação**: 
- IDs não têm poder preditivo
- Features constantes/quase-constantes são inúteis
- Features com muito missing podem prejudicar treinamento

### Célula 3: Correlação + MI

```
Pearson correlation - Top 3:
  1. feature_A: 0.4521
  2. feature_B: 0.3892
  3. feature_C: 0.3114

Pares altamente correlacionados (|r| > 0.95): 5
  - Remover: feature_D (mantém feature_E que tem r=0.42 com target)

Mutual Information - Top 3:
  1. feature_A: 0.2341 (MI = dependência não-linear forte)
  2. feature_F: 0.1892
  3. feature_G: 0.1654

Resultado: 165 → 110 features (33% redução)
```

**Interpretação**:
- Correlação mede relação LINEAR
- MI captura NÃO-LINEARIDADES
- Remover features redundantes (mantendo a mais correlacionada com target)

### Célula 4: LightGBM

```
Top 15 Features por Importance:
  1. feature_A:  2341.5
  2. feature_B:  1892.3
  3. feature_C:  1654.2
  ...
  180. feature_Z: 0.0 (não usado)

Zero importance features: 22
  (Nunca usadas pelo modelo - removidas)

ROC-AUC Validation: 0.8234
  (Bom - modelo discrimina bem)

Resultado: 142 → 88 features (38% redução)
```

**Interpretação**:
- Features com importance = 0 são "inúteis"
- Top features concentram a maioria da informação
- ROC-AUC alta = seleção de features efetiva

### Célula 5: SHAP

```
Base Value: 0.4521
  (Probabilidade média de default: 45.21%)

Top 5 Features por Impacto SHAP:
  1. feature_A (mean |SHAP| = 0.1234)
  2. feature_B (mean |SHAP| = 0.0892)
  ...

⬆️ AUMENTAM risco (valores altos → default):
  • feature_A: correlação SHAP = +0.623
  • feature_M: correlação SHAP = +0.451

⬇️ REDUZEM risco (valores altos → sem default):
  • feature_N: correlação SHAP = -0.542
  • feature_O: correlação SHAP = -0.389
```

**Interpretação**:
- SHAP valor positivo = aumenta probabilidade de default
- SHAP valor negativo = reduz probabilidade de default
- Usar para entender decisões do modelo

---

## 🎓 Conceitos Teóricos

### Por que 5 métodos?

1. **Correlação de Pearson**: Detecta relações lineares diretas
2. **Correlação de Spearman**: Relações monotônicas (rank-based)
3. **Mutual Information**: Dependências não-lineares
4. **LightGBM Importance**: Poder preditivo real (modelo-agnostic)
5. **SHAP**: Explicação das predições individuais

**Resultado**: Seleção robusta que captura múltiplos tipos de padrões

### Por que LightGBM?

| Critério | LightGBM | XGBoost | Random Forest |
|----------|----------|---------|---------------|
| Velocidade | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| GPU Support | ⭐⭐⭐ | ⭐⭐ | ❌ |
| Memory | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Kaggle (track record) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

### Por que SHAP?

- **Teoricamente sólido**: Baseado em teoria dos jogos
- **Model-agnostic**: Funciona com qualquer modelo (via explicações)
- **Interpretável**: SHAP values têm significado claro
- **Explainability**: Essencial para TCC/publicações acadêmicas

---

## ⚠️ Troubleshooting

### Erro: "CUDA out of memory"

```python
# Reduzir tamanho do batch
CONFIG['lightgbm_device'] = 'cpu'  # Fallback para CPU

# Ou reduzir `num_threads`
lgb_params['num_threads'] = 4  # Ao invés de -1
```

### Erro: "Missing parquet file"

```python
# Verificar path
data_path = "./data/raw/parquet/train/part-00000.parquet"
import os
assert os.path.exists(data_path), f"File not found: {data_path}"

# Listar arquivos disponíveis
import glob
files = glob.glob("./data/raw/parquet/train/*.parquet")
print(f"Arquivos encontrados: {files}")
```

### SHAP muito lento

```python
# Usar sample menor
sample_size = 1000  # Ao invés de 5000

# Ou apenas top features
top_n_dependence = 3  # Ao invés de 5
```

### Pipeline não reprodutível

```python
# Certificar que seed está definida ANTES de imports
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 🔥 GPU não detectada?

**Consulte [GPU_TROUBLESHOOTING.md](GPU_TROUBLESHOOTING.md) para diagnóstico completo!**

Ou execute no notebook:
```python
# Execute a CÉLULA DE DIAGNÓSTICO (após Célula 1)
# Mostrará informações detalhadas e soluções
```

---

## 📚 Referências e Leitura

1. **Feature Selection**:
   - [Sklearn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
   - "Feature Engineering for Machine Learning" (Zheng & Casari)

2. **LightGBM**:
   - [LightGBM Documentation](https://lightgbm.readthedocs.io/)
   - "CatBoost vs LightGBM" benchmarks

3. **SHAP**:
   - [SHAP GitHub](https://github.com/slundberg/shap)
   - [SHAP Paper](https://arxiv.org/abs/1705.07874)
   - "Interpretable Machine Learning" (Christoph Molnar)

4. **Competições Kaggle**:
   - American Express Default Prediction competition
   - Feature selection best practices from winners

---

## 🤝 Contribuições e Melhorias Futuras

- [ ] Suporte a mais backends (XGBoost, Catboost)
- [ ] Parallel feature selection
- [ ] Feature interactions detection
- [ ] Automated threshold tuning
- [ ] Report generation (HTML/PDF)

---

## 📄 Licença

MIT License - Livre para uso em projetos acadêmicos e comerciais.

---

**Desenvolvido com ❤️ para competições de Machine Learning**

Para dúvidas ou sugestões, abra uma issue no repositório.
