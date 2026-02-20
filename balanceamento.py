"""
FASE 2 — BALANCEAMENTO DE CLASSES (SMOTE) (Módulo)
objetivo: corrigir o desbalanceamento severo do dataset de fraudes.

Problema: a grande maioria das transações é legítima (classe 0).
Se não balancearmos, o modelo aprende a dizer "tudo é legítimo" e
acerta ~99% — mas falha em detectar fraudes (que é o objetivo).

Solução — SMOTE: cria exemplos SINTÉTICOS de fraude interpolando
entre exemplos reais. Aplicado SOMENTE no treino — o teste nunca
é tocado (deve refletir o mundo real desbalanceado).

Este módulo expõe uma função chamada pelo pipeline (main.py):
  balancear(df_treino) → df_balanceado
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

CAMINHO_SAIDA = r'f:\transacoes\transacoes\Classificacao-de-Fraudes-Bancarias\treino_balanceado.csv'


def balancear(df_treino):
    """
    Recebe df_treino pré-processado e com colunas já selecionadas.
    Aplica SMOTE e retorna df_balanceado.
    sampling_strategy=0.3 → fraudes = 30% dos legítimos.
    Não usamos 50/50 pois gera ruído excessivo; 20-30% é o recomendado
    para dados de fraude bancária.
    """
    TARGET   = 'fraud_bool'
    features = [c for c in df_treino.columns if c != TARGET]
    X = df_treino[features]
    y = df_treino[TARGET]

    contagem_antes = Counter(y)
    total_antes    = len(y)
    print(f"[SMOTE] Distribuição ANTES:")
    print(f"  Legítimo (0): {contagem_antes[0]:>7,}  ({contagem_antes[0]/total_antes*100:.2f}%)")
    print(f"  Fraude   (1): {contagem_antes[1]:>7,}  ({contagem_antes[1]/total_antes*100:.2f}%)")
    print(f"  Total       : {total_antes:>7,}")

    print("\n[SMOTE] Aplicando SMOTE (sampling_strategy=0.3)...")
    print("[SMOTE] Isso pode levar alguns minutos com ~800k linhas...")

    smote = SMOTE(
        sampling_strategy=0.3,
        random_state=42,
        k_neighbors=5
    )
    X_bal, y_bal = smote.fit_resample(X, y)

    contagem_depois = Counter(y_bal)
    total_depois    = len(y_bal)
    print(f"\n[SMOTE] Distribuição DEPOIS:")
    print(f"  Legítimo (0): {contagem_depois[0]:>7,}  ({contagem_depois[0]/total_depois*100:.2f}%)")
    print(f"  Fraude   (1): {contagem_depois[1]:>7,}  ({contagem_depois[1]/total_depois*100:.2f}%)")
    print(f"  Total       : {total_depois:>7,}")
    print(f"  Sintéticos criados: {contagem_depois[1] - contagem_antes[1]:,}")

    df_balanceado = pd.DataFrame(X_bal, columns=features)
    df_balanceado[TARGET] = y_bal.values

    df_balanceado.to_csv(CAMINHO_SAIDA, index=False)
    print(f"[SMOTE] Arquivo salvo: treino_balanceado.csv ({df_balanceado.shape[0]:,} linhas x {df_balanceado.shape[1]} colunas)")

    return df_balanceado


# ── Execução direta (sem pipeline) ──────────────────────────────────────────
if __name__ == '__main__':
    print("AVISO: executando balanceamento isolado. Para o pipeline, use: python main.py\n")
    CAMINHO_TREINO = r'f:\transacoes\transacoes\Classificacao-de-Fraudes-Bancarias\treino.csv'
    df_treino = pd.read_csv(CAMINHO_TREINO)
    balancear(df_treino)
