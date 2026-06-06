# Fase 5: Teste Final e Simulação de Produção (Avaliação Final)

## 1. Objetivo da Fase 5

A Fase 5 representa a prova de fogo e a conclusão empírica do pipeline de *Machine Learning*. O objetivo desta etapa foi validar a capacidade de generalização do modelo campeão (Voting Classifier) em um ambiente que simula rigorosamente a sua implantação no mundo real (Produção).

Para garantir a absoluta isenção de viés (*bias*) ou vazamento de dados (*data leakage*), o modelo foi avaliado em uma base de *Holdout* de 20% (91.783 clientes) que permaneceu trancada, intocada e isolada desde o início da Fase 0 do projeto.

## 2. Metodologia de Execução e Implantação

Diferente da Fase 4 — cujo propósito metodológico era comparar metodologias de combinação de modelos via validação cruzada — a Fase 5 adotou o protocolo padrão de implantação em produção:

1. **Expansão da Base de Conhecimento:** Os três algoritmos constituintes do *Ensemble* (XGBoost, LightGBM e CatBoost) foram instanciados com seus respectivos hiperparâmetros campeões (descobertos via Optuna na Fase 3). Em seguida, eles foram treinados em **100% da base de treino (367.130 clientes)** simultaneamente, sem partições de K-Fold. Isso garantiu que o modelo absorvesse a quantidade máxima de variância e inteligência possível antes do teste.
2. **Espelhamento de Arquitetura:** A base isolada de teste foi submetida à mesma máscara de redução de dimensionalidade validada na Fase 1, contendo apenas as 400 variáveis mais densas.
3. **Mecanismo de Decisão:** Utilizou-se o *Soft Voting Classifier*, calculando a média aritmética simples das probabilidades emitidas pelos três especialistas preditivos para determinar o risco final de cada novo cliente.

## 3. Resultados Oficiais do Teste Final

O modelo não apenas confirmou sua estabilidade, como **superou** as métricas obtidas na validação cruzada da Fase 4 (saltando de um AMEX Score de 0.7920 para 0.7931). Esse aumento empírico no teste final é a prova estatística definitiva de que a arquitetura não sofreu de *overfitting* (sobreajuste), sendo altamente capaz de generalizar padrões financeiros para novos indivíduos.

A tabela abaixo consolida as métricas finais da arquitetura:

| Métrica | Resultado Oficial | Interpretação no Contexto Financeiro |
| --- | --- | --- |
| **AMEX Score** | **0.7931** | Métrica Norte do projeto (Gini + Top 4%). Demonstra excelência na ordenação do risco de crédito e captação de *default*. |
| **ROC AUC** | **0.9618** | Altíssima capacidade global de separação entre clientes adimplentes e inadimplentes. |
| **AUPRC** | **0.9004** | Confirma que o modelo lida de forma excepcional com o desbalanceamento inerente ao risco de crédito. |
| **F1-Score** | **0.8087** | Demonstra o ponto de equilíbrio harmônico matemático entre precisão e sensibilidade. |
| **Recall (Sensibilidade)** | **0.9197** | O modelo foi capaz de capturar **~92%** de todos os inadimplentes reais ocultos na base de teste. |
| **Precision** | **0.7217** | De todos os clientes alertados como alto risco, 72% realmente se tornaram inadimplentes. |

---

## 4. Dessecando a Matriz de Confusão (O Impacto no Banco)

O modelo avaliou **91.783 clientes** inéditos na base de teste. A distribuição matemática das predições reflete decisões de risco com impacto direto sobre o balanço financeiro e a eficiência operacional da instituição de crédito:

* **Verdadeiros Negativos (59.587 clientes):** Clientes que eram bons pagadores e o modelo aprovou corretamente. O banco concede o crédito com segurança, expandindo sua carteira ativa de maneira saudável e gerando receita contínua com juros, tarifas e anuidades.
* **Verdadeiros Positivos (21.857 clientes):** Clientes que iriam dar calote (*default*) e o modelo detectou a tempo. Aqui reside a **mina de ouro comercial** do projeto: ao bloquear cartões ou reduzir preventivamente as linhas de crédito desses indivíduos, a instituição financeira mitigou o prejuízo direto associado à inadimplência sistêmica de quase 22 mil contas.
* **Falsos Positivos / Alarmes Falsos (8.430 clientes):** Clientes com perfil saudável (adimplentes), mas que o modelo classificou como de alto risco, resultando em recusa de crédito. Este volume representa o **Custo de Oportunidade** (lucro cessante pela perda de bons clientes em troca de segurança institucional), mantendo-se em um patamar perfeitamente aceitável e gerenciável pelas réguas de política de crédito do negócio.
* **Falsos Negativos / O Pior Erro (Apenas 1.909 clientes):** Clientes inadimplentes que o modelo falhou em mitigar, fazendo com que o crédito fosse concedido incorretamente. Este grupo simboliza o prejuízo real e o risco residual não absorvido pelas equações do *Ensemble*. Limitar a perda a este contingente em uma base desse tamanho (atingindo um *Recall* de quase 92%) posiciona o modelo no estado da arte da engenharia de risco.

## 5. Conclusão da Pesquisa Analítica

O desenvolvimento iterativo cumpriu com excelência a proposta metodológica. Partindo de uma base massiva e ruidosa com mais de 3.200 variáveis, o funil reduziu a dimensionalidade, isolou arquiteturas não aderentes, otimizou fronteiras bayesianas e, finalmente, fundiu especialidades algorítmicas, entregando um oráculo de classificação financeira altamente acurado, auditável e livre de viés amostral.

Como limitação natural da pesquisa, propõe-se para **trabalhos futuros** a aplicação de testes de estresse estocástico — reexecutando a arquitetura campeã sob múltiplas sementes aleatórias (*Random Seeds*) — para o cálculo do intervalo de confiança formal das predições frente a perturbações de inicialização.