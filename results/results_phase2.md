# Fase 2: Campeonato Aberto (Baseline dos Modelos)

## 1. Objetivo da Fase 2

A Fase 2, denominada "Campeonato Aberto", teve como objetivo estabelecer a linha de base de desempenho empírico (*Baseline*) para 7 algoritmos distintos de *Machine Learning*.

Nesta etapa, os modelos foram instanciados com seus **hiperparâmetros de fábrica (padrão)**, sem otimização avançada. O propósito não era encontrar o limite preditivo final, mas sim atuar como um funil metodológico: testar a afinidade natural das diferentes arquiteturas matemáticas com os dados de crédito da AMEX, filtrar os algoritmos ineficientes e identificar o "Top 3" que avançará para a etapa de otimização pesada na Fase 3.

## 2. Metodologia de Execução

Para garantir rigor científico e mitigar qualquer viés estatístico ou vazamento de dados (*data leakage*), o experimento seguiu as seguintes diretrizes:

* **Entrada de Dados:** Utilizou-se exclusivamente a base de dados "Enxuta" (400 features), validada empiricamente na Fase 1, contendo os tratamentos de imputação adequados para modelos sensíveis a valores nulos (via `Pipeline`).
* **Validação Cruzada (OOF - Out-Of-Fold):** Todos os modelos foram avaliados utilizando `StratifiedKFold` com 5 partições (Folds). Isso garante que a proporção de inadimplentes permaneça estritamente igual em todas as dobras de treino e validação, simulando um ambiente de produção real.
* **Hardware:** O treinamento foi executado inteiramente em arquitetura de CPU (Apple Silicon M1).
* **Métrica Norte:** O ranqueamento oficial foi determinado pela métrica **AMEX Score Global**, calculada sobre as predições consolidadas de validação (*Out-Of-Fold*).

---

## 3. Resultados Oficiais do Campeonato

A tabela abaixo apresenta os resultados consolidados, ordenados do melhor para o pior desempenho segundo o AMEX Score.

| Posição | Modelo | Tempo Total (s) | AMEX Score | ROC AUC | AUPRC | F1-Score | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 🥇 1º | **XGBoost** | 125.41 | **0.7871** | 0.9604 | 0.8952 | 0.8061 | 0.9167 |
| 🥈 2º | **LightGBM** | 290.23 | **0.7870** | 0.9604 | 0.8955 | 0.8059 | 0.9176 |
| 🥉 3º | **CatBoost** | **80.38** | **0.7858** | 0.9600 | 0.8939 | 0.8038 | **0.9222** |
| 4º | **ANN (MLP)** | 172.02 | 0.7747 | 0.9557 | 0.8851 | 0.8021 | 0.7980 |
| 5º | **Random Forest** | 732.88 | 0.7525 | 0.9535 | 0.8742 | 0.8001 | 0.8725 |
| 6º | **Logistic Regression** | 705.07 | 0.7481 | 0.9502 | 0.8666 | 0.7886 | 0.8887 |
| 7º | **KNN** | 420.28 | 0.5969 | 0.8772 | 0.7134 | 0.6614 | 0.5796 |

---

## 4. Análise Crítica de Performance

Os resultados empíricos validam diversas hipóteses metodológicas formuladas durante o planejamento do sistema:

### A Dominância do *Gradient Boosting*

O pódio foi inteiramente ocupado pelas arquiteturas baseadas em árvores de decisão com aprimoramento de gradiente (*Boosting*). XGBoost, LightGBM e CatBoost apresentaram pontuações virtualmente empatadas na casa de **0.787**.

* **XGBoost:** Sagrou-se campeão absoluto da Fase 2, atingindo o maior AMEX Score combinando um equilíbrio excelente entre Tempo e AUPRC.
* **CatBoost:** Destacou-se pela eficiência computacional extrema, finalizando as 5 dobras de treinamento em meros **80 segundos** (menos de 17 segundos por fold), mantendo um desempenho preditivo de elite e o maior *Recall* global do teste (0.9222).

### A Viabilidade das Redes Neurais (ANN)

A *Multi-Layer Perceptron (ANN)* conquistou um honroso 4º lugar (0.7747). Redes neurais tradicionais costumam sofrer severamente com dados tabulares ruidosos, mas o sucesso deste modelo comprova a alta qualidade da técnica de *Feature Selection* (seleção de 400 variáveis) aplicada na Fase 1, que entregou um conjunto de dados denso e matematicamente estável para a otimização dos pesos neuronais.

### O Custo da Complexidade e Dimensionalidade

* **Random Forest e Regressão Logística:** Ambos apresentaram tempos de execução excessivamente altos (acima de 700 segundos). O Random Forest por construir árvores profundas completas, e a Regressão Logística por esgotar as iterações do motor `lbfgs` tentando encontrar convergência matemática em um plano de 400 dimensões sem dados normalizados.
* **KNN:** Como estatisticamente previsto pela Maldição da Dimensionalidade, o algoritmo K-Nearest Neighbors falhou criticamente na base da AMEX (AMEX Score de 0.5969). Em 400 dimensões, o cálculo de distância euclidiana perde o significado prático, impossibilitando a separação efetiva das classes.

## 5. Conclusão e Próximos Passos

A Fase 2 cumpriu seu objetivo de funil analítico. Os modelos KNN, Regressão Logística, Random Forest e Rede Neural estão oficialmente desclassificados para não onerar o custo computacional do projeto em otimizações que não atingirão o estado da arte.

A próxima etapa **(Fase 3: Otimização Suprema)** focará no limite preditivo da base de dados. Utilizando otimização bayesiana (Optuna) e aceleração de hardware (GPU), realizaremos o *Hyperparameter Tuning* exclusivamente nos três campeões: **XGBoost, LightGBM e CatBoost**, visando romper a barreira do AMEX Score de 0.79.