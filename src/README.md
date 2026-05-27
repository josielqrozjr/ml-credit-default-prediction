# Diretório de Modelos e Benchmarking: Detecção de Inadimplência

Este diretório contém os códigos e as definições metodológicas para a avaliação dos algoritmos de classificação. O objetivo desta etapa é comparar o desempenho de diferentes modelos matemáticos na previsão de inadimplência (target) utilizando a base de dados tratada nas fases anteriores.

Para evitar alto custo computacional e garantir a validade científica, a avaliação foi dividida em um funil de quatro fases. 

---

## 1. Estrutura de Fases do Benchmark

A execução dos modelos segue uma ordem de validação por etapas. A saída de uma fase justifica as configurações da fase seguinte.

| Fase | Objetivo | Escopo de Execução | Justificativa Metodológica |
| :--- | :--- | :--- | :--- |
| **Fase 1: Provas de Conceito** | Validar empiricamente a utilidade do Feature Selection (FS) e a abordagem de Balanceamento. | Modelos: Regressão Logística e XGBoost.<br>Testes: (Com FS vs. Sem FS) e (Algorítmico vs. Undersampling vs. Sem Balanceamento). | Evita testar todas as combinações em todos os modelos. Comprova as hipóteses iniciais nos dois extremos algorítmicos (um modelo linear e um ensemble de árvores). |
| **Fase 2: Campeonato Aberto** | Estabelecer a linha de base de desempenho para todos os algoritmos escolhidos. | 10 Modelos rodando com hiperparâmetros padrão (default), base enxuta (400 variáveis) e balanceamento algorítmico. | Permite uma comparação nivelada e com baixo custo computacional inicial. Identifica os algoritmos que melhor se adaptam à estrutura de dados da AMEX. |
| **Fase 3: Otimização de Hiperparâmetros** | Encontrar o limite de performance dos melhores algoritmos identificados na Fase 2. | Algoritmo Optuna aplicado apenas no Top 3 modelos da Fase 2. Uso de aceleração por GPU. | O método de otimização bayesiana (Optuna) é mais rápido e eficiente que o GridSearch tradicional. Aplicá-lo apenas no Top 3 economiza recursos de processamento. |
| **Fase 4: Meta-Classificadores** | Combinar as previsões dos modelos otimizados para reduzir variância e erro. | Stacking, Blending e Voting Classifier utilizando os modelos ajustados na Fase 3. | Modelos baseados na combinação de classificadores tendem a apresentar maior robustez contra *overfitting*. |

---

## 2. Seleção e Justificativa dos Algoritmos

O benchmark final é composto por 10 algoritmos, distribuídos em quatro categorias principais. Modelos clássicos como *Decision Tree* e *AdaBoost* foram removidos do escopo. A *Decision Tree* sofre *overfitting* em dados de alta dimensão, enquanto o *AdaBoost* é sensível a *outliers* (comuns em dados financeiros). 

Para substituí-los, foram incluídos o CatBoost e o LightGBM. A tabela abaixo detalha a escolha do portfólio de teste:

| Categoria | Modelo | Justificativa Técnica para Inclusão |
| :--- | :--- | :--- |
| **Modelos Tradicionais (Baseline)** | **Regressão Logística (LR)** | Representante linear paramétrico. Sensível à multicolinearidade. Usado para provar a eficácia da seleção de variáveis. |
| | **K-Nearest Neighbors (KNN)** | Modelo baseado em distância. Útil para capturar agrupamentos locais de perfis de inadimplência. |
| **Redes Neurais** | **ANN (Multi-Layer Perceptron)** | Capacidade de mapear interações não-lineares complexas através de múltiplas camadas ocultas. |
| **Ensembles Homogêneos (Árvores)** | **Random Forest** | Baseado em *Bagging*. Cria variabilidade usando amostras e subconjuntos de variáveis, reduzindo a variância geral em relação a uma árvore simples. |
| | **XGBoost** | Baseado em *Gradient Boosting*. Estrutura robusta contra valores ausentes e alto desempenho de convergência, com suporte nativo a GPU. |
| | **LightGBM** | Algoritmo rápido que constrói árvores por folha (leaf-wise). Incluído por ter sido o algoritmo que extraiu as métricas de importância no Feature Selection. |
| | **CatBoost** | Baseado em árvores simétricas. Lida de forma otimizada com as 22 variáveis categóricas mapeadas no pipeline sem a necessidade de codificação prévia. |
| **Meta-Classificadores** | **Voting Classifier** | Combina as previsões dos modelos base por média simples de probabilidade (*soft voting*), suavizando os erros individuais. |
| | **Stacking** | Treina um modelo final sobre as previsões (usando validação cruzada) dos modelos base. Corrige tendências de erro dos classificadores subjacentes. |
| | **Blending** | Variação do Stacking que utiliza uma partição *holdout* fixa para treinar o meta-classificador. Mais rápido computacionalmente e mitiga o risco de vazamento de dados. |

---

## 3. Diretrizes de Execução e Prevenção de Vazamento de Dados

Algumas decisões técnicas foram adotadas para garantir a integridade dos resultados:

* **Isolamento da Base de Teste:** A base de teste (20%) não é exposta a nenhuma etapa do *Feature Selection* ou balanceamento de dados. 
* **Espelhamento de Dimensão:** O filtro de seleção (400 colunas) é calculado apenas na base de treino. A lista resultante (arquivo `.txt`) é usada como máscara para aplicar o mesmo corte nas colunas da base de teste antes da entrada nos modelos. Isso garante conformidade de formato sem vazamento de informação (*data leakage*).
* **Foco no Balanceamento Algorítmico:** Devido à natureza financeira da base, técnicas de superamostragem (*oversampling*) física, como SMOTE, tendem a criar dados sintéticos irreais. O benchmark priorizará o balanceamento via função de custo (ex: `class_weight` ou `scale_pos_weight`), penalizando matematicamente os erros cometidos na classe minoritária.
* **Aceleração por Hardware:** Modelos compatíveis (como XGBoost e CatBoost) serão executados via placa de vídeo dedicada (GPU) durante a Fase 3 para viabilizar a otimização de dezenas de hiperparâmetros em tempo hábil.