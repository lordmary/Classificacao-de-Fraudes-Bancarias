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
"""

import time

# ── Importar módulos do pipeline ────────────────────────────────────────────
from pre_processamento import carregar_e_limpar, dividir_e_salvar
from selecao_atributos  import selecionar
from balanceamento       import balancear

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
    print("  ETAPA 4/4 — Balanceamento de Classes (SMOTE)")
    print(f"{'─'*65}")
    t0 = time.time()
    df_balanceado = balancear(df_treino)
    print(f"  [OK] Concluído em {time.time()-t0:.1f}s")

    # ── Resumo final ────────────────────────────────────────────────────────
    tempo_total = time.time() - inicio_total
    print(f"\n{SEP}")
    print("  FASE 2 CONCLUÍDA — RESUMO")
    print(SEP)
    print(f"  Colunas removidas         : {len(colunas_remover)} {colunas_remover}")
    print(f"  Colunas no modelo final   : {df_treino.shape[1]}")
    print(f"  Treino original           : {df_treino.shape[0]:,} linhas")
    print(f"  Treino balanceado (SMOTE) : {df_balanceado.shape[0]:,} linhas")
    print(f"  Teste (intacto)           : {df_teste.shape[0]:,} linhas")
    print(f"  Tempo total               : {tempo_total/60:.1f} min")
    print(SEP)
    print("\n  Próximo passo: Fase 3 → treinar Random Forest vs. XGBoost")
    print(f"{SEP}\n")


if __name__ == '__main__':
    main()
