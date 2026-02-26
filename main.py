"""
PIPELINE PRINCIPAL — Trabalho Final MD
Detecção de Fraudes Bancárias (BAF NeurIPS 2022)

Executa toda a Fase 2 em sequência, passando dados entre os módulos:

  ETAPA 1 → pre_processamento : carrega e limpa o Base.csv
  ETAPA 2 → selecao_atributos : identifica colunas irrelevantes (3 métodos)
  ETAPA 3 → pre_processamento : remove colunas + divide treino/teste + salva CSVs
  ETAPA 4 → balanceamento      : aplica SMOTE no treino → treino_balanceado.csv

Uso:
  python main.py

Arquivos gerados:
  treino.csv              — treino sem colunas irrelevantes (meses 0-5)
  teste.csv               — teste intacto (meses 6-7), nunca modificado
  treino_balanceado.csv   — treino com SMOTE aplicado, pronto para Fase 3
  resultado_selecao_atributos.csv — ranking completo para o relatório

Fase 3 (treinamento):
  modelo_rf.pkl           — Random Forest treinado
  modelo_xgb.pkl          — XGBoost treinado
  resultado_modelos.csv   — comparativo de métricas dos dois modelos
"""

import time

# ── Importar módulos do pipeline ────────────────────────────────────────────
from pre_processamento import carregar_e_limpar, dividir_e_salvar
from selecao_atributos  import selecionar
from balanceamento       import balancear
from treinamento         import treinar_e_avaliar
from pos_processamento import executar as pos_processar

SEP = "=" * 65

def main():
    inicio_total = time.time()

    print(SEP)
    print("  PIPELINE — FASE 2: PRÉ-PROCESSAMENTO COMPLETO")
    print(SEP)

    # ── ETAPA 1: Carregar e limpar ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  ETAPA 1/4 — Carregar Base.csv e tratar nulos/encoding")
    print(f"{'─'*65}")
    t0 = time.time()
    df, serie_month = carregar_e_limpar()
    print(f"  [OK] Concluído em {time.time()-t0:.1f}s")

    # ── ETAPA 2: Seleção de atributos ───────────────────────────────────────
    # Passa apenas o treino (meses 0-5) sem 'month' para não vazar o teste
    print(f"\n{'─'*65}")
    print("  ETAPA 2/4 — Seleção de Atributos (3 métodos)")
    print(f"{'─'*65}")
    t0 = time.time()
    df_treino_temp = df[serie_month <= 5].drop(columns=['month'])
    colunas_remover, df_ranking = selecionar(df_treino_temp)
    print(f"\n  [OK] Colunas a remover: {colunas_remover}")
    print(f"  [OK] Concluído em {time.time()-t0:.1f}s")

    # ── ETAPA 3: Remover colunas + dividir + salvar ─────────────────────────
    print(f"\n{'─'*65}")
    print("  ETAPA 3/4 — Remover colunas irrelevantes e salvar treino/teste")
    print(f"{'─'*65}")
    t0 = time.time()
    df_treino, df_teste = dividir_e_salvar(df, serie_month, colunas_remover)
    print(f"  [OK] Concluído em {time.time()-t0:.1f}s")

    # ── ETAPA 4: SMOTE ──────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  ETAPA 4/5 — Balanceamento de Classes (SMOTE)")
    print(f"{'─'*65}")
    t0 = time.time()
    df_balanceado = balancear(df_treino)
    print(f"  [OK] Concluído em {time.time()-t0:.1f}s")

    # ── ETAPA 5: Treinamento ────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  ETAPA 5/5 — Treinamento e Avaliação (RF vs. XGBoost)")
    print(f"{'─'*65}")
    t0 = time.time()
    resultados_modelos = treinar_e_avaliar()
    print(f"  [OK] Concluído em {time.time()-t0:.1f}s")

    # ── Resumo final ────────────────────────────────────────────────────────

    # ── ETAPA 6: Pós-processamento ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  ETAPA 6/6 — Pós-processamento (Threshold + Gráficos + SHAP)")
    print(f"{'─'*65}")
    t0 = time.time()
    resultados_fase4 = pos_processar()
    print(f"  [OK] Concluído em {time.time()-t0:.1f}s")

    tempo_total = time.time() - inicio_total
    print(f"\n{SEP}")
    print("  PIPELINE COMPLETO (FASES 2 + 3 + 4) — RESUMO")
    print(SEP)
    print(f"  Colunas removidas         : {len(colunas_remover)} {colunas_remover}")
    print(f"  Colunas no modelo final   : {df_treino.shape[1]}")
    print(f"  Treino original           : {df_treino.shape[0]:,} linhas")
    print(f"  Treino balanceado (SMOTE) : {df_balanceado.shape[0]:,} linhas")
    print(f"  Teste (intacto)           : {df_teste.shape[0]:,} linhas")
    print(f"{'─'*65}")
    print("  MODELOS — Métricas base (threshold 0.50)")
    print(f"{'─'*65}")
    for nome, res in resultados_modelos.items():
        print(f"  {nome:<20}: AUC-ROC={res['auc_roc']:.4f}  AUC-PR={res['auc_pr']:.4f}  F1={res['f1']:.4f}")
    print(f"{'─'*65}")
    print("  PÓS-PROCESSAMENTO — Métricas com threshold ótimo (Fase 4)")
    print(f"{'─'*65}")
    for nome, res in resultados_fase4.items():
        print(f"  {nome:<20}: Threshold={res['threshold']:.2f} | "
              f"AUC-ROC={res['auc_roc']:.4f} | "
              f"AUC-PR={res['auc_pr']:.4f} | "
              f"F1={res['f1']:.4f} | "
              f"Recall={res['recall']:.4f} | "
              f"Precision={res['precision']:.4f}")
        print(f"  {'':20}  TP={res['tp']:,}  FP={res['fp']:,}  "
              f"TN={res['tn']:,}  FN={res['fn']:,}")
    print(f"{'─'*65}")
    melhor_fase4 = max(resultados_fase4, key=lambda n: resultados_fase4[n]['auc_pr'])
    print(f"  Melhor modelo (AUC-PR)    : {melhor_fase4}")
    print(f"  Tempo total               : {tempo_total/60:.1f} min")
    print(SEP)


if __name__ == '__main__':
    main()
