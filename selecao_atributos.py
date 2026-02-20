"""
FASE 2 — SELEÇÃO DE ATRIBUTOS (Módulo)
objetivo: identificar quais das 32 colunas realmente ajudam a prever fraude
          e quais são irrelevantes, usando 3 métodos combinados:

  Método 1 — Correlação de Pearson com o alvo : relações lineares
  Método 2 — Importância por Random Forest    : relações não-lineares (mais confiável)
  Método 3 — Filtro de Variância Baixa        : colunas quasi-constantes = inúteis

Este módulo expõe uma função chamada pelo pipeline (main.py):
  selecionar(df_treino) → (colunas_remover, df_resultado)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

CAMINHO_RESULTADO = r'f:\transacoes\transacoes\Classificacao-de-Fraudes-Bancarias\resultado_selecao_atributos.csv'


def selecionar(df_treino):
    """
    Recebe df_treino já pré-processado (sem 'month').
    Executa 3 métodos de seleção e retorna:
      colunas_remover (list) — colunas a descartar
      df_resultado    (DataFrame) — ranking completo para análise/relatório
    """
    TARGET   = 'fraud_bool'
    features = [c for c in df_treino.columns if c != TARGET]
    X = df_treino[features]
    y = df_treino[TARGET]

    print(f"[SELEÇÃO] Dados recebidos: {df_treino.shape[0]:,} linhas | {len(features)} features")
    print(f"[SELEÇÃO] Taxa de fraude : {y.mean()*100:.2f}% ({y.sum():,} fraudes)")
    print("=" * 65)

    # ── Método 1: Correlação de Pearson ──────────────────────────────
    print("\n[SELEÇÃO - MÉTODO 1] Correlação de Pearson com fraud_bool")
    print("  Detecta relações lineares. Quanto maior o valor, melhor.\n")
    correlacoes = X.corrwith(y).abs().sort_values(ascending=False)
    df_corr = correlacoes.reset_index()
    df_corr.columns = ['coluna', 'correlacao_abs']
    df_corr['rank_corr'] = range(1, len(df_corr) + 1)
    print(df_corr.to_string(index=False))

    # ── Método 2: Importância por Random Forest ───────────────────────
    print("\n[SELEÇÃO - MÉTODO 2] Importância por Random Forest")
    print("  Detecta relações NÃO-LINEARES — mais confiável para fraude.")
    print("  Treinando em amostra estratificada (fraudes + 10x legítimos)...\n")

    idx_fraude = df_treino[df_treino[TARGET] == 1].index
    idx_legit  = df_treino[df_treino[TARGET] == 0].index
    n_legit    = min(len(idx_fraude) * 10, len(idx_legit))
    idx_sample = np.concatenate([idx_fraude, np.random.choice(idx_legit, n_legit, replace=False)])
    np.random.shuffle(idx_sample)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X.loc[idx_sample], y.loc[idx_sample])

    importancias = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    df_rf = importancias.reset_index()
    df_rf.columns = ['coluna', 'importancia_rf']
    df_rf['rank_rf'] = range(1, len(df_rf) + 1)
    print(df_rf.to_string(index=False))

    # ── Método 3: Filtro de Variância Baixa ──────────────────────────
    print("\n[SELEÇÃO - MÉTODO 3] Filtro de Variância Baixa")
    print("  Colunas quasi-constantes não ensinam nada ao modelo.\n")
    X_norm = (X - X.min()) / (X.max() - X.min() + 1e-9)
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_norm)
    colunas_baixa_variancia = [f for f, ok in zip(features, selector.get_support()) if not ok]
    if colunas_baixa_variancia:
        print(f"  Variância baixa detectada: {colunas_baixa_variancia}")
    else:
        print("  Nenhuma coluna com variância criticamente baixa.")

    # ── Ranking unificado ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RANKING FINAL — COMBINAÇÃO DOS 3 MÉTODOS")
    print("=" * 65)

    df_final = df_corr.merge(df_rf, on='coluna')
    df_final['score_combinado'] = (df_final['rank_corr'] + df_final['rank_rf']) / 2
    df_final = df_final.sort_values('score_combinado').reset_index(drop=True)
    df_final['rank_final'] = range(1, len(df_final) + 1)
    df_final['baixa_variancia'] = df_final['coluna'].isin(colunas_baixa_variancia)

    def _recomendar(row):
        if row['baixa_variancia']:
            return 'REMOVER (variância baixa)'
        elif row['rank_final'] <= 15:
            return 'MANTER'
        elif row['rank_final'] <= 22:
            return 'AVALIAR'
        else:
            return 'REMOVER (pouco relevante)'

    df_final['recomendacao'] = df_final.apply(_recomendar, axis=1)

    df_exibir = df_final[['rank_final', 'coluna', 'correlacao_abs', 'importancia_rf', 'recomendacao']].copy()
    df_exibir['correlacao_abs'] = df_exibir['correlacao_abs'].round(4)
    df_exibir['importancia_rf'] = df_exibir['importancia_rf'].round(4)
    print(df_exibir.to_string(index=False))

    # ── Resumo ────────────────────────────────────────────────────────
    manter  = df_final[df_final['recomendacao'] == 'MANTER']['coluna'].tolist()
    avaliar = df_final[df_final['recomendacao'] == 'AVALIAR']['coluna'].tolist()
    remover = df_final[df_final['recomendacao'].str.startswith('REMOVER')]['coluna'].tolist()

    print(f"\n MANTER  ({len(manter)}) : {manter}")
    print(f" AVALIAR ({len(avaliar)}) : {avaliar}")
    print(f" REMOVER ({len(remover)}) : {remover}")
    print("=" * 65)

    # Salvar CSV com ranking completo
    df_final[['rank_final', 'coluna', 'correlacao_abs', 'importancia_rf',
              'score_combinado', 'recomendacao']].to_csv(CAMINHO_RESULTADO, index=False)
    print(f"[SELEÇÃO] Resultado salvo em: {CAMINHO_RESULTADO}")

    # colunas_remover = REMOVER (não inclui AVALIAR — decisão conservadora)
    return remover, df_final


# ── Execução direta (sem pipeline) ──────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, r'f:\transacoes\transacoes\Classificacao-de-Fraudes-Bancarias')
    from pre_processamento import carregar_e_limpar
    print("AVISO: executando seleção isolada. Para o pipeline, use: python main.py\n")
    df, serie_month = carregar_e_limpar()
    df_treino = df[serie_month <= 5].drop(columns=['month'])
    colunas_remover, _ = selecionar(df_treino)
    print(f"\ncolunas_remover = {colunas_remover}")
