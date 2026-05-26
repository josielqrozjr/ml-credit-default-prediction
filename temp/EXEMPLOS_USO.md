# Exemplos de Uso - Pipeline Feature Selection
## Cenários Práticos e Configurações

---

## 📍 Cenário 1: Dataset Pequeno (< 1M linhas)

**Ideal para**: Prototipação rápida, testes locais

### Configuração
```python
CONFIG = {
    'correlation_threshold': 0.95,
    'missing_threshold': 0.90,
    'mutual_info_percentile': 70,      # Mais agressivo
    'lgb_importance_percentile': 75,   # Mais features
    'lightgbm_device': 'cpu',          # CPU é mais rápido para datasets pequenos
    'seed': 42,
}
```

### Tempo Esperado
- Célula 1: 1 min
- Célula 2: 1 min
- Célula 3: 2 min
- Célula 4: 5 min
- Célula 5: 3 min
- **Total: ~12 minutos**

### Ajustes na Célula 3
```python
# Matriz de correlação em memória (é pequeno)
corr_matrix = X_filled.corr(method='pearson')  # SEM sampling
```

### Ajustes na Célula 5
```python
# SHAP com sample maior
sample_size = 10000  # Ou até len(X_val)
```

---

## 📍 Cenário 2: Dataset Médio (1M - 50M linhas)

**Ideal para**: Competições reais, clusters pequenos

### Configuração
```python
CONFIG = {
    'correlation_threshold': 0.95,
    'missing_threshold': 0.90,
    'mutual_info_percentile': 75,      # Balanceado
    'lgb_importance_percentile': 80,   # Balanceado
    'lightgbm_device': 'gpu',          # GPU ganha velocidade aqui
    'seed': 42,
}
```

### Tempo Esperado (com GPU)
- Célula 1: 1 min
- Célula 2: 5 min
- Célula 3: 10 min
- Célula 4: 15 min
- Célula 5: 8 min
- **Total: ~40 minutos**

### Ajustes Específicos
```python
# Célula 2 - Leitura incremental
# DuckDB lida bem com parquets grandes
df = conn_duckdb.execute(
    f"SELECT * FROM parquet_scan('{data_path}')"
).df()  # Carrega conforme necessário

# Célula 3 - Sampling para correlação
if X_filled.shape[0] > 100000:
    sample_idx = np.random.choice(X_filled.shape[0], 100000, replace=False)
    corr_matrix = X_filled.iloc[sample_idx].corr(method='pearson')

# Célula 4 - LightGBM otimizado
lgb_params['num_leaves'] = 255        # Árvores mais profundas
lgb_params['feature_fraction'] = 0.7  # Mais agressivo
```

---

## 📍 Cenário 3: Dataset Grande (> 50M linhas)

**Ideal para**: Produção, servidores com GPU

### Configuração
```python
CONFIG = {
    'correlation_threshold': 0.92,     # Menos rigoroso
    'missing_threshold': 0.95,         # Remove menos features
    'mutual_info_percentile': 80,      # Mais conservador
    'lgb_importance_percentile': 85,   # Mais features
    'lightgbm_device': 'gpu',          # GPU essencial
    'seed': 42,
}
```

### Tempo Esperado (com GPU A100)
- Célula 1: 1 min
- Célula 2: 15 min (leitura)
- Célula 3: 20 min (MI incremental)
- Célula 4: 30 min (treinamento)
- Célula 5: 10 min (SHAP sample)
- **Total: ~75 minutos**

### Ajustes Críticos
```python
# Célula 2 - Processamento em chunks
chunk_size = 1000000
chunks = []
for start in range(0, len(df), chunk_size):
    chunks.append(process_chunk(df.iloc[start:start+chunk_size]))
X_clean = pd.concat(chunks)

# Célula 3 - MI em sample estratificado
sample_stratified = df.groupby('target', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 100000), random_state=42)
)

# Célula 4 - GPU com max threads
lgb_params.update({
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'num_threads': -1,
    'num_leaves': 255,
    'max_depth': 15,
})

# Célula 5 - Apenas top features para SHAP
sample_size = 5000
top_n_dependence = 3  # Apenas 3 features
```

---

## 🎯 Cenário 4: Kaggle Competition (American Express)

**Exigências**:
- ~190 features originais
- ~5M linhas de treino
- Métrica: AMEX (custom)
- Tempo limite: < 2 horas total

### Configuração Recomendada
```python
CONFIG = {
    'correlation_threshold': 0.95,
    'missing_threshold': 0.88,         # Um pouco mais agressivo
    'mutual_info_percentile': 75,
    'lgb_importance_percentile': 80,
    'lightgbm_device': 'gpu',
    'seed': 42,
}

# LightGBM otimizado para Kaggle
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,              # Regularização
    'lambda_l2': 0.1,
    'verbose': -1,
    'device': 'gpu',
}

# CV agressivo para seleção
num_boost_round = 300
early_stopping_rounds = 30
```

### Pipeline Executado
```bash
# Tempo: 1h 45min (com GPU RTX 3090)
# Resultado esperado: 80-100 features finais
# ROC-AUC: > 0.82

Iniciais: 190
Pós-limpeza: 170
Pós-correlação: 145
Pós-MI: 115
Final: 88 features ✅
```

---

## 🔄 Cenário 5: Produção - Retrainamento Mensal

**Contexto**: Sistema em produção que retraina com novos dados

### Estratégia
```python
# 1. Guardar lista de features anteriores
PRODUCTION_FEATURES = [
    'feature_A', 'feature_B', ..., 'feature_Z'
]

# 2. Executar pipeline em novos dados
# Célula 1-5: Completo

# 3. Comparar seleção
previous_set = set(PRODUCTION_FEATURES)
new_set = set(features_final)

added = new_set - previous_set
removed = previous_set - new_set
stable = previous_set & new_set

print(f"Estáveis: {len(stable)}")
print(f"Adicionadas: {len(added)}")
print(f"Removidas: {len(removed)}")

# 4. Decisão
if len(removed) / len(previous_set) > 0.05:
    print("⚠️ Muita mudança! Investigar feature drift")
    # Usar features antigas (mais conservador)
    features_final = list(previous_set)
else:
    print("✅ Mudanças pequenas. Usar novo conjunto")
    # Usar novo conjunto
    PRODUCTION_FEATURES = features_final
```

### Logging para Monitoramento
```python
logger.info(f"Production Features Changed: {len(removed)} removed, {len(added)} added")
logger.info(f"Feature Stability: {len(stable)/len(previous_set)*100:.1f}%")

# Salvar para alertas
if len(removed) > 10:
    send_alert(f"Feature drift detected: {len(removed)} features removed")
```

---

## 💻 Cenários de Hardware

### 1. CPU Only (Laptop)
```python
CONFIG['lightgbm_device'] = 'cpu'
lgb_params['num_threads'] = 4

# Tempo: 3-5x mais lento
# Máximo recomendado: 1M linhas
```

### 2. GPU NVIDIA (RTX 3090)
```python
CONFIG['lightgbm_device'] = 'gpu'
lgb_params['gpu_device_id'] = 0

# Tempo esperado: normal
# Máximo recomendado: 50M linhas
```

### 3. GPU Servidor (A100)
```python
CONFIG['lightgbm_device'] = 'gpu'
lgb_params.update({
    'gpu_device_id': 0,
    'num_leaves': 512,
    'max_depth': 20,
})

# Máximo recomendado: 100M+ linhas
# Pode ser paralelizado em múltiplas GPUs
```

---

## 🎓 Exemplos de Interpretação

### Exemplo 1: Feature Protection
```
Feature: customer_credit_score
SHAP correlation: -0.62

Interpretação:
- Valores altos de credit_score → SHAP negativo
- Clientes com credit_score alto têm MENOR probabilidade de default
- É um fator de PROTEÇÃO
- Ação: Aumentar weight/importância em modelos de aprovação
```

### Exemplo 2: Feature Risk
```
Feature: debt_to_income_ratio
SHAP correlation: +0.58

Interpretação:
- Valores altos de debt_to_income → SHAP positivo
- Clientes com ratio alto têm MAIOR probabilidade de default
- É um fator de RISCO
- Ação: Rejeitar clientes com ratio > threshold
```

### Exemplo 3: Feature Redundante
```
Pearsn r(feature_A, feature_B): 0.97
target_corr(A): 0.35
target_corr(B): 0.32

Ação: Manter A, remover B
Razão: A tem correlação ligeiramente maior com target
```

---

## 📊 Métricas de Validação

### Antes vs. Depois
```python
# Antes da seleção
from sklearn.metrics import roc_auc_score
y_pred_before = model_with_all_features.predict(X_test)
auc_before = roc_auc_score(y_test, y_pred_before)

# Depois da seleção
y_pred_after = model_with_selected_features.predict(X_test[features_final])
auc_after = roc_auc_score(y_test, y_pred_after)

improvement = ((auc_after - auc_before) / auc_before) * 100
print(f"AUC: {auc_before:.4f} → {auc_after:.4f} ({improvement:+.1f}%)")

# Esperado: redução < 2%, ou até melhoria (menos overfitting)
```

---

## ⚡ Performance Optimization

### Se Pipeline Está Lento

```python
# 1. Aumentar sample size em Célula 3 (menos rigoroso)
if X_filled.shape[0] > 500000:
    sample_idx = np.random.choice(X_filled.shape[0], 50000, replace=False)  # Menor sample
    corr_matrix = X_filled.iloc[sample_idx].corr()

# 2. Reduzir features antes de LightGBM
features_pre_filtered = top_features_by_mi[:200]  # Top 200 apenas
X_filtered = X_selected_mi[features_pre_filtered]

# 3. Menos árvores em LightGBM
num_boost_round = 200  # Ao invés de 500

# 4. Skip SHAP se tempo crítico
# (Vá direto aos features_final)
```

---

## 🔐 Boas Práticas

✅ **Sempre fazer:**
- [ ] Salvar features finais em arquivo
- [ ] Versionar features (com timestamp)
- [ ] Logar todas as decisões
- [ ] Validar em test set separado
- [ ] Comparar com baseline (todas as features)

❌ **Nunca fazer:**
- [ ] Tunar hyperparameters em dados de validação
- [ ] Misturar dados de train/test
- [ ] Usar data leakage features (ex: ID temporal futuro)
- [ ] Esquecer de testar reprodutibilidade

---

**Última atualização**: 2024-05-19
