# Predição de Inadimplência com Machine Learning — American Express Default Prediction

## 1. Visão Geral do Projeto

Este repositório contém **todos os artefatos** do projeto de pesquisa em Machine Learning para predição de inadimplência de cartão de crédito, desenvolvido a partir da competição pública [American Express - Default Prediction (Kaggle)](https://www.kaggle.com/competitions/amex-default-prediction/overview).

O pipeline completo abrange desde a ingestão e engenharia de features sobre dados brutos até o treinamento, otimização bayesiana e avaliação final de modelos ensemble, resultando em um **Voting Classifier (LightGBM + XGBoost + CatBoost)** com AMEX Score de **0.7931** no teste final.

---

## 2. Estrutura do Repositório e Descrição dos Artefatos

```text
ml-credit-default-prediction/
├── README.md                         # Este arquivo (descrição geral dos artefatos)
├── config.py                         # Configurações globais, hiperparâmetros e caminhos
├── requirements.txt                  # Dependências do projeto
├── train.ipynb                       # Notebook unificado com pipeline completo
│
├── app/                              # Código-fonte completo do projeto
│   ├── README.md                     # Documentação técnica detalhada do pipeline
│   ├── pipeline/
│   │   ├── convert_parquet.py        # Conversão CSV → Parquet (DuckDB)
│   │   ├── feature_engineering.py    # Engenharia temporal (diff1, changed) via SQL
│   │   ├── aggregation.py            # Agregação por cliente (5.5M linhas → 458K clientes)
│   │   ├── merge_split.py            # Merge com labels + split estratificado 80/20
│   │   └── feature_selection.py      # Seleção de features (3.265 → 400 via LightGBM)
│   ├── evaluation/
│   │   ├── amex_metric.py            # Métrica oficial AMEX (Gini + Top 4%)
│   │   ├── metrics.py                # Avaliação OOF e validação cruzada estratificada
│   │   └── visualization.py          # Geração de gráficos acadêmicos (300 DPI)
│   ├── models/
│   │   ├── registry.py               # Registro central dos 10 modelos
│   │   ├── logistic_regression.py    # Regressão Logística (class_weight=balanced)
│   │   ├── knn.py                    # K-Nearest Neighbors
│   │   ├── ann.py                    # Rede Neural MLP (64→32 neurônios)
│   │   ├── random_forest.py          # Random Forest (n_estimators=150)
│   │   ├── xgboost_model.py          # XGBoost (GPU-accelerated)
│   │   ├── lightgbm_model.py         # LightGBM (is_unbalance=True)
│   │   └── catboost_model.py         # CatBoost (auto_class_weights=Balanced)
│   └── phases/
│       ├── run_phase0_pipeline.py    # Fase 0: Pipeline completo (DuckDB + Polars + Merge + Feature Selection)
│       ├── run_phase1_poc.py         # Fase 1: Provas de Conceito
│       ├── run_phase2_benchmark.py   # Fase 2: Campeonato Aberto (7 modelos)
│       ├── run_phase3_optuna.py      # Fase 3: Otimização Bayesiana (Optuna)
│       ├── run_phase4_ensembles.py   # Fase 4: Meta-Classificadores (Ensembles)
│       └── run_phase5_final_test.py  # Fase 5: Teste Final (Produção)
│
├── data/                             # Dados do projeto
│   ├── raw/parquet/                  # Dados brutos particionados (Parquet)
│   │   ├── train/                    # Partições de treino (data_*.parquet)
│   │   └── train_labels/             # Labels de treino (data_*.parquet)
│   └── processed/
│       ├── merge_split/              # Datasets processados (train_80 + valid_20)
│       └── selection/                # Lista de features selecionadas
│
├── results/                          # Resultados experimentais
│   ├── poc_01_dimensionalidade.csv   # Resultados POC: Dimensionalidade
│   ├── poc_02_balanceamento.csv      # Resultados POC: Balanceamento
│   ├── phase2_benchmark_ranking.csv  # Ranking dos 7 modelos (Fase 2)
│   ├── phase4_ensembles_ranking.csv  # Ranking dos ensembles (Fase 4)
│   ├── phase5_final_test.csv         # Métricas finais do teste (Fase 5)
│   ├── results_phase1.md             # Relatório analítico — Fase 1
│   ├── results_phase2.md             # Relatório analítico — Fase 2
│   ├── results_phase3.md             # Relatório analítico — Fase 3
│   ├── results_phase4.md             # Relatório analítico — Fase 4
│   ├── results_phase5.md             # Relatório analítico — Fase 5
│   ├── best_models/
│   │   └── optuna_best_params.json   # Hiperparâmetros campeões (Optuna, Fase 3)
│   └── plots/
│       ├── 01_phase2_ranking.png     # Gráfico: Ranking AMEX Score (Fase 2)
│       ├── 02_amex_evolution.png     # Gráfico: Evolução do score entre fases
│       └── 03_confusion_matrix.png   # Gráfico: Matriz de confusão final
│
└── catboost_info/                    # Logs de treinamento do CatBoost
    ├── catboost_training.json
    └── learn_error.tsv
```

---

## 3. Descrição Detalhada dos Artefatos Principais

### 3.1 Pipeline de Dados (`app/pipeline/`)

| Artefato | Descrição | Parâmetros / Observações |
|----------|-----------|---------------------------|
| **`app/phases/run_phase0_pipeline.py`** | Orquestrador completo do pipeline de dados. Executa sequencialmente: engenharia temporal (DuckDB), agregação por cliente (Polars), merge com labels, split estratificado e seleção de features. | Caminhos de entrada/saída definidos internamente. Executar a partir da raiz do projeto. |
| **`app/pipeline/feature_engineering.py`** | Gera features temporais via funções de janela SQL (LAG). Para numéricos: `_diff1` (diferença entre meses). Para categóricos: `_changed` (flag de transição). | Utiliza DuckDB in-memory. |
| **`app/pipeline/aggregation.py`** | Agrega séries temporais em uma linha por cliente. Calcula: mean, std, min, max, last, total_delta, trend_ratio, pos_ratio, avg_monthly_slope. | Transforma ~5.5M linhas em ~458K clientes × 3.264 colunas. |
| **`app/pipeline/merge_split.py`** | Realiza inner join com labels e split estratificado 80/20 preservando a proporção de inadimplentes. | `test_size=0.20`, `seed=42`. Saída: `train_80.parquet`, `valid_20.parquet`. |
| **`app/pipeline/feature_selection.py`** | Seleciona as 400 features mais relevantes via importância LightGBM (Gain). Filtra previamente: missing >99.9%, quasi-constantes, correlação >0.98. | `top_lgbm_features=400`, `is_unbalance=True`. |

### 3.2 Configuração Global (`config.py`)

Centraliza **todos** os parâmetros do projeto:
- Caminhos de dados (`TRAIN_DATA_PATH`, `TEST_DATA_PATH`, `SELECTED_FEATURES_PATH`)
- Seed de reprodutibilidade (`RANDOM_SEED = 42`)
- Número de folds (`N_SPLITS = 5`)
- Hiperparâmetros padrão dos 7 modelos individuais (`HYPERPARAMS`)
- Espaços de busca Optuna para o Top 3 (`OPTUNA_GRIDS`)
- Detecção automática de GPU (PyTorch CUDA)

### 3.3 Scripts de Treinamento e Avaliação (`app/phases/`)

| Script | Descrição | Como Executar |
|--------|-----------|---------------|
| **`run_phase0_pipeline.py`** | Pipeline completo de dados: engenharia temporal, agregação, merge, split e seleção de features. | `python -m app.phases.run_phase0_pipeline` |
| **`run_phase1_poc.py`** | Provas de Conceito: (1) valida ganho do Feature Selection (3.265 vs 400 features); (2) compara estratégias de balanceamento. | `python -m app.phases.run_phase1_poc` |
| **`run_phase2_benchmark.py`** | Campeonato Aberto: avalia 7 modelos individuais via StratifiedKFold (5 folds) com métricas OOF. Gera ranking pelo AMEX Score. | `python -m app.phases.run_phase2_benchmark` |
| **`run_phase3_optuna.py`** | Otimização Bayesiana: aplica Optuna (50-100 trials) nos 3 melhores modelos da Fase 2. Salva hiperparâmetros campeões em JSON. | `python -m app.phases.run_phase3_optuna` |
| **`run_phase4_ensembles.py`** | Meta-Classificadores: combina Top 3 otimizados via Soft Voting, Stacking e Blending. Avalia qual arquitetura supera os individuais. | `python -m app.phases.run_phase4_ensembles` |
| **`run_phase5_final_test.py`** | Teste Final: treina Voting Classifier em 100% do treino e avalia na base isolada de 20% (91.783 clientes). Produz métricas finais definitivas. | `python -m app.phases.run_phase5_final_test` |

### 3.4 Módulo de Avaliação (`app/evaluation/`)

| Artefato | Descrição |
|----------|-----------|
| **`amex_metric.py`** | Implementação vetorizada (NumPy) da métrica oficial AMEX = 0.5 × (Gini Normalizado + Taxa de Captura Top 4%). Inclui wrappers para XGBoost e LightGBM. |
| **`metrics.py`** | Função `evaluate_model()` que retorna AMEX Score, ROC AUC, AUPRC, F1, Precision, Recall e Matriz de Confusão. Função `evaluate_model_cv()` para validação cruzada OOF. |
| **`visualization.py`** | Gera gráficos acadêmicos em 300 DPI: ranking de modelos, evolução do AMEX Score e matriz de confusão. |

### 3.5 Resultados e Métricas (`results/`)

| Artefato | Descrição |
|----------|-----------|
| **`poc_01_dimensionalidade.csv`** | Tabela comparativa: XGBoost e LR em base completa (3.265) vs. enxuta (400 features). |
| **`poc_02_balanceamento.csv`** | Tabela comparativa: Sem Balanceamento vs. Undersampling vs. Algorítmico. |
| **`phase2_benchmark_ranking.csv`** | Ranking dos 7 modelos com AMEX Score, ROC AUC, AUPRC, F1 (OOF). |
| **`phase4_ensembles_ranking.csv`** | Ranking: Voting (0.7920) > Stacking (0.7918) > Blending (0.7943*). |
| **`phase5_final_test.csv`** | Métricas oficiais do teste final: AMEX 0.7931, ROC AUC 0.9618, Recall 0.9197. |
| **`results_phase[1-5].md`** | Relatórios analíticos detalhados de cada fase experimental. |
| **`plots/*.png`** | Figuras geradas para o artigo científico. |

### 3.6 Modelo Final (`results/best_models/`)

| Artefato | Descrição |
|----------|-----------|
| **`optuna_best_params.json`** | Hiperparâmetros otimizados para LightGBM (AMEX: 0.7910), XGBoost (0.7900) e CatBoost (0.7893). Utilizados para instanciar o Voting Classifier final. |

### 3.7 Notebook Unificado

| Artefato | Descrição |
|----------|-----------|
| **`train.ipynb`** | Notebook Jupyter que integra e executa todas as fases do projeto em sequência: pipeline de dados, benchmark dos modelos, otimização, ensembles e teste final. Serve como demonstração reproduzível do fluxo completo. |

---

## 4. Preparação do Ambiente e Execução

### 4.1 Requisitos

- Python 3.10 ou superior
- GPU CUDA (opcional, para aceleração de XGBoost/LightGBM/CatBoost)

### 4.2 Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 4.3 Dados da Competição

Os dados brutos da competição devem ser obtidos via [Kaggle](https://www.kaggle.com/competitions/amex-default-prediction/data) e posicionados em `data/raw/parquet/`. Não são incluídos no repositório por questões de licenciamento e tamanho (~50 GB).

### 4.4 Ordem de Execução Completa

```bash
# Todas as fases são executadas a partir da raiz do projeto
python -m app.phases.run_phase0_pipeline   # Pipeline de Dados
python -m app.phases.run_phase1_poc        # Provas de Conceito
python -m app.phases.run_phase2_benchmark  # Campeonato Aberto (7 modelos)
python -m app.phases.run_phase3_optuna     # Otimização Bayesiana (Optuna)
python -m app.phases.run_phase4_ensembles  # Meta-Classificadores
python -m app.phases.run_phase5_final_test # Teste Final
```

Alternativamente, o notebook `train.ipynb` executa todo o fluxo sequencialmente.

---

## 5. Resultados Finais

| Métrica | Resultado |
|---------|-----------|
| **AMEX Score** | **0.7931** |
| **ROC AUC** | 0.9618 |
| **AUPRC** | 0.9004 |
| **F1-Score** | 0.8087 |
| **Recall** | 0.9197 |
| **Precision** | 0.7217 |

Modelo final: **Voting Classifier** (média das probabilidades de LightGBM + XGBoost + CatBoost otimizados via Optuna), avaliado em base de teste isolada com 91.783 clientes.

---

## 6. Reprodutibilidade

- Seed global: `42` (definida em `config.py`)
- Split estratificado: 80% treino / 20% teste (preservação da proporção de classes)
- Validação cruzada: StratifiedKFold com 5 folds
- Todos os resultados em `results/` são reproduzíveis re-executando os scripts na ordem indicada

---

## 7. Referência

American Express. *American Express - Default Prediction* (Kaggle). Disponível em:
<https://www.kaggle.com/competitions/amex-default-prediction/overview>. Acesso em: 10 maio 2026.

## 8. Licença

Este repositório utiliza a licença definida no arquivo `LICENSE`.