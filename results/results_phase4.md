# Fase 4: Meta-Classificadores e Ensembles (A Fronteira Preditiva)

## 1. Objetivo da Fase 4

A Fase 4 teve como propósito fundamental superar o "teto de vidro" preditivo alcançado pelos modelos individuais na Fase 3 (onde o LightGBM atingiu um AMEX Score de **0.7910**).

Para isso, empregamos a teoria de *Ensemble Learning*, que postula que a combinação de múltiplos algoritmos de alta performance — com arquiteturas matemáticas distintas — tende a gerar um preditor global mais preciso e robusto. Ao consolidar as previsões do **XGBoost** (crescimento por nível), **LightGBM** (crescimento por folha) e **CatBoost** (árvores simétricas), o objetivo é fazer com que os acertos de um modelo compensem as falhas pontuais do outro (redução de variância).

## 2. Decisões Arquiteturais e Otimização Computacional

A execução de Meta-Classificadores sobre algoritmos previamente submetidos à Otimização Bayesiana exige extrema cautela para evitar a explosão do custo computacional (*Nested Cross-Validation*). Para viabilizar esta etapa em arquitetura de CPU (Apple Silicon M1), adotamos uma estratégia avançada de engenharia:

* **Desacoplamento da Matriz de Validação (OOF):** Em vez de utilizar as classes nativas completas do *Scikit-Learn* (que reexecutariam o *K-Fold* dos três algoritmos do zero, desperdiçando horas de processamento), o orquestrador gerou uma Matriz Base *Out-Of-Fold* (OOF) única.
* **Reaproveitamento de Memória:** O *K-Fold* Estratificado (5 partições) foi executado apenas uma vez com os hiperparâmetros campeões da Fase 3. Os vetores de probabilidade gerados foram armazenados em memória e reutilizados como *features* para treinar simultaneamente as três arquiteturas de *Ensemble* avaliadas.

## 3. Arquiteturas Avaliadas

Foram implementadas três metodologias distintas para consolidar as opiniões dos modelos base:

1. **Soft Voting Classifier:** A abordagem mais direta. Realiza a média aritmética simples das probabilidades emitidas pelos três modelos. Pressupõe que todos os especialistas têm o mesmo peso de decisão.
2. **Stacking Classifier:** Utiliza uma arquitetura de Meta-Aprendizado. Uma Regressão Logística foi treinada sobre as predições cruzadas (OOF) de 100% da base de treino. O objetivo é que o modelo linear aprenda matematicamente em qual dos três algoritmos confiar mais para perfis específicos de clientes.
3. **Blending Classifier:** Semelhante ao *Stacking*, mas utiliza uma partição de dados fixa (*Holdout* de 20%). Os modelos base são treinados em 80% dos dados e o Meta-Modelo aprende exclusivamente com as previsões geradas sobre a fatia isolada de 20%.

---

## 4. Resultados Oficiais

A tabela abaixo consolida o ranqueamento das arquiteturas de combinação, atestando a quebra da barreira preditiva em relação aos modelos individuais.

| Posição | Arquitetura | AMEX Score | ROC AUC | Base de Validação |
| --- | --- | --- | --- | --- |
| 1º | **Blending Classifier** | 0.7943 | 0.9614 | Holdout Fixo (20%) |
| 2º | **Voting Classifier** | 0.7920 | 0.9614 | Cross-Validation OOF (100%) |
| 3º | **Stacking Classifier** | 0.7918 | 0.9613 | Cross-Validation OOF (100%) |

*(Nota de Referência: O melhor modelo individual da Fase 3 foi o LightGBM, com AMEX Score de 0.7910).*

---

## 5. Análise Crítica e Deliberação do Modelo Campeão

O sucesso empírico dos *Ensembles* foi validado: todas as três metodologias superaram a barreira de 0.7910, confirmando a hipótese de que a diversidade algorítmica agrega valor preditivo ao risco de crédito. Contudo, a seleção do modelo final exige rigor estatístico.

### O Paradoxo do Blending Classifier

Embora o *Blending Classifier* tenha retornado o maior AMEX Score nominal (**0.7943**), este resultado carrega um viés amostral de alta variância. Por definição metodológica, o *Blending* é validado sobre uma fatia menor do conjunto de dados (*Holdout* de 20%). É estatisticamente mais propício a oscilações otimistas (ou pessimistas) devido a características pontuais daquela sub-amostra específica de clientes. No rigor acadêmico, sua métrica não é diretamente comparável às métricas extraídas por validação cruzada.

### A Supremacia do Voting Classifier

O **Voting Classifier (0.7920)** e o **Stacking Classifier (0.7918)** foram validados sobre a totalidade da base de treino (100%) através do *StratifiedKFold*. Suas pontuações refletem um poder de generalização real e inquestionável.

O fato do *Voting Classifier* (média matemática estrita) ter superado o *Stacking* (que possui uma Regressão Logística ponderando pesos) comprova o princípio da Navalha de Ockham: a solução mais simples e direta revelou-se a mais eficaz. A calibração extraída via Optuna na Fase 3 foi tão precisa que a interferência de um Meta-Modelo para repesar as probabilidades tornou-se desnecessária e causou uma levíssima perda por ruído.

## 6. Conclusão da Fase 4

A arquitetura **Voting Classifier (Composição: LightGBM + XGBoost + CatBoost)** é declarada a vencedora absoluta do processo de desenvolvimento e *Tuning* metodológico.

**Próximos Passos:** O projeto avança para a **Fase 5 (Teste Final)**, simulando a implantação do modelo em produção na matriz trancada de 20% do início do projeto.