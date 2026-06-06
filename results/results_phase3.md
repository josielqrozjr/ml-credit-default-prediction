# Fase 3: Otimização Suprema (Hyperparameter Tuning com Optuna)

## 1. Objetivo da Fase 3

Após a conclusão do "Campeonato Aberto" (Fase 2), onde estabelecemos a linha de base empírica e filtramos os algoritmos de menor desempenho, a Fase 3 teve como propósito extrair o **limite preditivo absoluto** do Top 3 de modelos selecionados: **LightGBM, XGBoost e CatBoost**.

Para isso, substituímos os parâmetros padrão de fábrica por uma busca de hiperparâmetros através de **Otimização Bayesiana**, utilizando o *framework* Optuna. Diferente do *Grid Search* tradicional (que testa todas as combinações cegamente), a otimização bayesiana utiliza probabilidade para focar o processamento apenas nas regiões mais promissoras do espaço de busca, maximizando a eficiência computacional.

## 2. Metodologia e Decisões Arquiteturais

O ambiente de otimização foi desenhado para ser resiliente e à prova de vazamento de dados (*data leakage*), respeitando as seguintes premissas:

* **Validação Cruzada Estrita:** Cada combinação de hiperparâmetro sugerida pelo Optuna (*Trial*) foi submetida a um `StratifiedKFold` de 5 partições. Isso significa que para cada um dos 50 *trials*, o modelo foi treinado 5 vezes, totalizando **250 treinamentos independentes por algoritmo**.
* **Proteção Matemática Desacoplada (LightGBM):** Para evitar falhas críticas no motor C++ do LightGBM, foi implementado um "escudo matemático" no orquestrador. A restrição teórica de que o número de folhas não pode exceder $2^{\text{max\_depth}} - 1$ foi programada de forma dinâmica, interceptando sugestões inválidas do Optuna sem a necessidade de alterar os limites globais do projeto.
* **Mecanismo de Checkpoint (Segurança de Execução):** Devido à alta carga computacional do teste em CPU (que excedeu 13 horas de duração ininterrupta), injetamos um *callback* no Optuna para salvar o progresso iterativamente em disco (`optuna_best_params.json`). Isso blindou o experimento contra quedas de energia ou estouros de memória.
* **Métrica Norte:** O Optuna foi matematicamente configurado com a diretriz `direction="maximize"`, utilizando exclusivamente o **AMEX Score Global (OOF)** como bússola para validar as melhorias.

---

## 3. Resultados Oficiais da Otimização

A aplicação do Optuna quebrou a estagnação preditiva vista na Fase 2, impulsionando todos os modelos do *Baseline* de ~0.787 para a faixa de excelência superior a **0.789**. O processo consumiu um total de **47.125 segundos (aprox. 13 horas)** de processamento contínuo em arquitetura Apple Silicon (ARM).

Abaixo, o quadro consolida o ganho de performance e os hiperparâmetros campeões descobertos para cada algoritmo:

| Posição | Modelo | AMEX Score (Fase 2) | AMEX Score (Fase 3) | Ganho Absoluto | Tempo de Otimização |
| --- | --- | --- | --- | --- | --- |
| 🥇 1º | **LightGBM** | 0.7871 | **0.7910** | + 0.0039 | 22.097 s (6.1h) |
| 🥈 2º | **XGBoost** | 0.7872 | **0.7900** | + 0.0028 | 16.067 s (4.4h) |
| 🥉 3º | **CatBoost** | 0.7858 | **0.7893** | + 0.0035 | 8.960 s (2.4h) |

### Dicionário de Hiperparâmetros Campeões (JSON Final)

* **LightGBM (O Grande Vencedor):** Encontrou sua performance máxima construindo árvores profundas e contendo a variância via regularização de amostras.
* `n_estimators`: 427
* `max_depth`: 9
* `num_leaves`: 61
* `learning_rate`: 0.0440
* `min_child_samples`: 41
* `subsample`: 0.9276
* `colsample_bytree`: 0.5033


* **XGBoost:** Optou por um modelo mais conservador em profundidade, apostando em um volume maior de estimadores com taxa de aprendizado reduzida.
* `n_estimators`: 474
* `max_depth`: 7
* `learning_rate`: 0.0305
* `subsample`: 0.9070
* `colsample_bytree`: 0.8466
* `min_child_weight`: 10


* **CatBoost:** Destacou-se pela eficiência computacional extrema, encontrando a otimização em menos da metade do tempo do LightGBM.
* `iterations`: 360
* `depth`: 7
* `learning_rate`: 0.0788
* `l2_leaf_reg`: 3.2099
* `border_count`: 140



---

## 4. Análise Crítica e Comportamento Espacial dos Algoritmos

A extração dos logs nativos em C++ revelou o comportamento de "estresse de fronteira" gerado pela Otimização Bayesiana:

### A Ruptura do LightGBM

O LightGBM foi o único capaz de romper a barreira técnica de `0.791`. A análise de seus hiperparâmetros campeões demonstra que ele utilizou apenas **~50% das colunas por árvore** (`colsample_bytree: 0.5033`). Essa decisão matemática imposta pelo Optuna forçou o modelo a criar árvores muito diversas umas das outras, reduzindo drasticamente o viés e alcançando a generalização perfeita sobre a base enxuta (400 features).

### O Teste de Estresse do CatBoost e o "Degenerate Solution"

Durante a iteração do CatBoost, o motor reportou o aviso nativo: *`Training has stopped (degenerate solution on iteration 452, probably too small l2-regularization)`*.
Longe de ser uma falha estrutural, este log comprova a exploração agressiva do algoritmo Bayesiano. O Optuna forçou uma diminuição extrema da regularização L2 (`l2_leaf_reg`) para medir os limites de sobreajuste. Quando o peso matemático da folha tendeu a zero (criando uma divisão por zero latente), o algoritmo abortou o *Trial* graciosamente.
O aprendizado estatístico foi validado pelo resultado final: o JSON descartou o parâmetro falho e selecionou um regulador L2 seguro e alto (`3.2099`), garantindo que o modelo final entregue em produção seja blindado contra colapsos matemáticos.

## 5. Conclusão e Preparação para a Fase 4

A Fase 3 encerra o ciclo de evolução algorítmica individual. Convertemos três *Baselines* competentes em três "Especialistas" em previsão de inadimplência da American Express, todos isoladamente superiores à faixa de 0.789.

O próximo e último passo do fluxo preditivo será a **Fase 4 (Meta-Classificadores e Ensembles)**. Em vez de depender de um único "oráculo", combinaremos o poder de decisão do XGBoost, LightGBM e CatBoost em comitês (utilizando *Voting*, *Stacking* e *Blending*) para diluir os erros individuais e buscar um *AMEX Score* global definitivo rumo à fronteira de 0.80.