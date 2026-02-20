"""
FASE 2 — SELEÇÃO DE ATRIBUTOS
objetivo: identificar quais das 32 colunas realmente ajudam a prever fraude
          e quais são irrelevantes, usando 3 métodos combinados:

  Método 1 — Correlação com o Alvo    : quão ligada cada coluna está à fraude
  Método 2 — Importância (Random Forest): peso real que um modelo dá a cada coluna
  Método 3 — Variância Baixa          : colunas que quase não variam = inúteis

  Saída final: ranking unificado + lista de colunas RECOMENDADAS para manter
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. CARREGAR DADOS
# ─────────────────────────────────────────────
CAMINHO_BASE = r'F:\transacoes\base\Base.csv'

print("Carregando Base.csv... (aguarde)")
df = pd.read_csv(CAMINHO_BASE)

# Replicar o pré-processamento do pre_processamento.py
# para garantir que os dados estejam no mesmo estado

# Tratamento de -1 (nulos)
colunas_com_nulos = ['prev_address_months_count', 'bank_months_count', 'current_address_months_count']
for col in colunas_com_nulos:
    mediana = df[df[col] != -1][col].median()
    df[col] = df[col].replace(-1, mediana)

# Encoding das categóricas
le = LabelEncoder()
colunas_texto = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
for col in colunas_texto:
    df[col] = le.fit_transform(df[col].astype(str))

# Usar apenas dados de treino (meses 0-5) para a seleção
# Nunca devemos "espiar" o conjunto de teste na análise
df_treino = df[df['month'] <= 5].copy()

# Remover 'month' — ela separa treino/teste, não é feature real
df_treino = df_treino.drop(columns=['month'])

TARGET = 'fraud_bool'
features = [c for c in df_treino.columns if c != TARGET]

X = df_treino[features]
y = df_treino[TARGET]

print(f"\nDados carregados: {df_treino.shape[0]:,} linhas | {len(features)} features\n")
print(f"Taxa de fraude no treino: {y.mean()*100:.2f}% ({y.sum():,} fraudes)\n")
print("=" * 65)

# ─────────────────────────────────────────────
# MÉTODO 1 — CORRELAÇÃO DE PEARSON COM O ALVO
# ─────────────────────────────────────────────
print("\n[MÉTODO 1] Correlação com fraud_bool")
print("-" * 65)
print("  Interpreta: quanto maior o valor absoluto, mais ligada à fraude.")
print("  Obs: detecta apenas relações LINEARES.\n")

correlacoes = X.corrwith(y).abs().sort_values(ascending=False)
df_corr = correlacoes.reset_index()
df_corr.columns = ['coluna', 'correlacao_abs']
df_corr['rank_corr'] = range(1, len(df_corr) + 1)

print(df_corr.to_string(index=False))

# ─────────────────────────────────────────────
# MÉTODO 2 — IMPORTÂNCIA POR RANDOM FOREST
# ─────────────────────────────────────────────
print("\n\n[MÉTODO 2] Importância por Random Forest")
print("-" * 65)
print("  Interpreta: quanto o modelo usa cada coluna para tomar decisões.")
print("  Captura relações NÃO-LINEARES — mais confiável para fraude.\n")
print("  Treinando Random Forest... (amostra de 100k linhas para agilizar)")

# Amostra estratificada para não demorar horas
from sklearn.utils import resample
idx_frade   = df_treino[df_treino[TARGET] == 1].index
idx_legit   = df_treino[df_treino[TARGET] == 0].index

# Pegar todas as fraudes + amostra dos legítimos (proporção 1:10)
n_fraude  = len(idx_frade)
n_legit   = min(n_fraude * 10, len(idx_legit))
idx_sample = np.concatenate([idx_frade, np.random.choice(idx_legit, n_legit, replace=False)])
np.random.shuffle(idx_sample)

X_sample = X.loc[idx_sample]
y_sample = y.loc[idx_sample]

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',   # compensa desbalanceamento
    random_state=42,
    n_jobs=-1
)
rf.fit(X_sample, y_sample)

importancias = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
df_rf = importancias.reset_index()
df_rf.columns = ['coluna', 'importancia_rf']
df_rf['rank_rf'] = range(1, len(df_rf) + 1)

print(df_rf.to_string(index=False))

# ─────────────────────────────────────────────
# MÉTODO 3 — VARIÂNCIA BAIXA (COLUNAS QUASI-CONSTANTES)
# ─────────────────────────────────────────────
print("\n\n[MÉTODO 3] Filtro de Variância Baixa")
print("-" * 65)
print("  Interpreta: colunas que quase não variam não ensinam nada ao modelo.\n")

# Normalizar para 0-1 antes de medir variância (para comparação justa)
X_norm = (X - X.min()) / (X.max() - X.min() + 1e-9)
selector = VarianceThreshold(threshold=0.01)   # limiar: <1% de variação
selector.fit(X_norm)

colunas_baixa_variancia = [f for f, ok in zip(features, selector.get_support()) if not ok]
colunas_ok_variancia    = [f for f, ok in zip(features, selector.get_support()) if ok]

if colunas_baixa_variancia:
    print(f"  Colunas com variância MUITO BAIXA (candidatas a remover): {colunas_baixa_variancia}")
else:
    print("  Nenhuma coluna com variância criticamente baixa encontrada.")

# ─────────────────────────────────────────────
# RANKING UNIFICADO — COMBINAÇÃO DOS 3 MÉTODOS
# ─────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("RANKING FINAL — COMBINAÇÃO DOS 3 MÉTODOS")
print("=" * 65)

df_final = df_corr.merge(df_rf, on='coluna')

# Score combinado: média dos ranks (menor = melhor)
df_final['score_combinado'] = (df_final['rank_corr'] + df_final['rank_rf']) / 2
df_final = df_final.sort_values('score_combinado')
df_final['rank_final'] = range(1, len(df_final) + 1)

# Marcar colunas com baixa variância
df_final['baixa_variancia'] = df_final['coluna'].isin(colunas_baixa_variancia)

# Definir recomendação
def recomendar(row):
    if row['baixa_variancia']:
        return 'REMOVER (variância baixa)'
    elif row['rank_final'] <= 15:
        return 'MANTER'
    elif row['rank_final'] <= 22:
        return 'AVALIAR'
    else:
        return 'REMOVER (pouco relevante)'

df_final['recomendacao'] = df_final.apply(recomendar, axis=1)

# Formatar para exibição
df_exibir = df_final[['rank_final', 'coluna', 'correlacao_abs', 'importancia_rf', 'recomendacao']].copy()
df_exibir['correlacao_abs']  = df_exibir['correlacao_abs'].round(4)
df_exibir['importancia_rf']  = df_exibir['importancia_rf'].round(4)

print(df_exibir.to_string(index=False))

# ─────────────────────────────────────────────
# RESUMO EXECUTIVO
# ─────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("RESUMO EXECUTIVO")
print("=" * 65)

manter   = df_final[df_final['recomendacao'] == 'MANTER']['coluna'].tolist()
avaliar  = df_final[df_final['recomendacao'] == 'AVALIAR']['coluna'].tolist()
remover  = df_final[df_final['recomendacao'].str.startswith('REMOVER')]['coluna'].tolist()

print(f"\n MANTER  ({len(manter)} colunas) — maior poder preditivo:")
for c in manter:
    print(f"    + {c}")

print(f"\n AVALIAR ({len(avaliar)} colunas) — contribuição moderada:")
for c in avaliar:
    print(f"    ? {c}")

print(f"\n REMOVER ({len(remover)} colunas) — baixo poder preditivo:")
for c in remover:
    print(f"    - {c}")

print("\n" + "=" * 65)
print(f"  Total features originais : {len(features)}")
print(f"  Recomendado para treino  : {len(manter)} (MANTER)")
print(f"  Opcional (ganho marginal): {len(avaliar)} (AVALIAR)")
print(f"  Recomendado remover      : {len(remover)} (REMOVER)")
print("=" * 65)

# ─────────────────────────────────────────────
# SALVAR RESULTADOS
# ─────────────────────────────────────────────
SAIDA = r'f:\transacoes\transacoes\Classificacao-de-Fraudes-Bancarias\resultado_selecao_atributos.csv'
df_final[['rank_final', 'coluna', 'correlacao_abs', 'importancia_rf', 'score_combinado', 'recomendacao']].to_csv(SAIDA, index=False)
print(f"\nResultado salvo em: {SAIDA}")

# Lista pronta para copiar no próximo script
print("\n─── LISTA FINAL PARA USAR NO TREINO ───")
print("colunas_selecionadas =", manter)
