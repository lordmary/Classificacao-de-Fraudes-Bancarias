"""
FASE 2 — PRÉ-PROCESSAMENTO (Módulo)
objetivo: carregar o Base.csv, tratar os valores -1 (nulos),
fazer encoding de categorias e dividir temporalmente em treino/teste.

Este módulo expõe funções chamadas pelo pipeline (main.py):
  carregar_e_limpar(caminho)                        → (df, serie_month)
  dividir_e_salvar(df, serie_month, colunas_remover) → (df_treino, df_teste)

A lista colunas_remover é gerada DINAMICAMENTE pela selecao_atributos.py
e passada pelo main.py — nenhuma coluna está hardcoded aqui.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import os

CAMINHO_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'base\Base.csv')
CAMINHO_SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saidas')


def carregar_e_limpar(caminho=CAMINHO_BASE):
    """
    Etapa 1: carrega CSV, trata nulos e aplica encoding.
    Retorna (df, serie_month):
      df          — base completa limpa (ainda com 'month' dentro)
      serie_month — série preservada para a divisão treino/teste
    """
    print("[PRÉ-PROCESSAMENTO] Lendo Base.csv... (aguarde)")
    df = pd.read_csv(caminho)

    # Preservar 'month' antes de qualquer transformação
    serie_month = df['month'].copy()

    # 1. Tratamento de Valores Faltantes (-1) → imputação pela mediana
    colunas_com_nulos = [
        'prev_address_months_count',
        'bank_months_count',
        'current_address_months_count',
    ]
    for col in colunas_com_nulos:
        mediana = df[df[col] != -1][col].median()
        df[col] = df[col].replace(-1, mediana)
    print("[PRÉ-PROCESSAMENTO] Nulos tratados (imputação por mediana)")

    # 2. Encoding de Variáveis Categóricas (texto → número)
    le = LabelEncoder()
    colunas_texto = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    for col in colunas_texto:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[PRÉ-PROCESSAMENTO] Encoding aplicado em: {colunas_texto}")
    print(f"[PRÉ-PROCESSAMENTO] Base carregada: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

    return df, serie_month


def dividir_e_salvar(df, serie_month, colunas_remover):
    """
    Etapa 2: remove colunas irrelevantes, divide temporalmente e salva os CSVs.
    colunas_remover → lista dinâmica vinda da selecao_atributos via main.py.
    'month' é sempre removida independente da lista (evita overfitting temporal).
    Retorna (df_treino, df_teste).
    """
    # 3. Remoção dinâmica de colunas + 'month' (sempre)
    remover = list(set(list(colunas_remover) + ['month']))
    remover = [c for c in remover if c in df.columns]
    df = df.drop(columns=remover)
    print(f"[PRÉ-PROCESSAMENTO] {len(remover)} colunas removidas → {df.shape[1]} colunas restantes")

    # 4. Divisão Temporal: Treino (meses 0-5) | Teste (meses 6-7)
    df_treino = df[serie_month <= 5].copy()
    df_teste  = df[serie_month > 5].copy()

    # 5. Shuffle apenas no Treino
    df_treino = shuffle(df_treino, random_state=42)

    # 6. Salvar
    df_treino.to_csv(rf'{CAMINHO_SAIDA}\treino.csv', index=False)
    df_teste.to_csv(rf'{CAMINHO_SAIDA}\teste.csv',   index=False)

    print(f"[PRÉ-PROCESSAMENTO] Treino (Meses 0-5) : {df_treino.shape[0]:,} linhas")
    print(f"[PRÉ-PROCESSAMENTO] Teste  (Meses 6-7) : {df_teste.shape[0]:,} linhas")
    print(f"[PRÉ-PROCESSAMENTO] Taxa de fraude (treino): {df_treino['fraud_bool'].mean()*100:.2f}%")
    print(f"[PRÉ-PROCESSAMENTO] Arquivos 'treino.csv' e 'teste.csv' salvos.")

    return df_treino, df_teste


# ── Execução direta (sem pipeline) ──────────────────────────────────────────
if __name__ == '__main__':
    print("AVISO: executando pré-processamento isolado (sem seleção de atributos).")
    print("Para o pipeline completo, execute: python main.py\n")
    df, serie_month = carregar_e_limpar()
    dividir_e_salvar(df, serie_month, colunas_remover=[])