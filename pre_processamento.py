"""
objetivo: carregar o Base.csv, tratar os valores -1 (nulos)
preparar os dados para os algoritmos de classificação.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# 1. Carregar a base bruta
print("Lendo Base.csv... (Aguarde, processando 1 milhão de linhas)")
df = pd.read_csv('Base.csv')

# 2. Tratamento de Valores Faltantes (-1)
# Substituindo -1 pela mediana (Imputação) para manter a consistência estatística
colunas_com_nulos = ['prev_address_months_count', 'bank_months_count', 'current_address_months_count']
for col in colunas_com_nulos:
    # Calculamos a mediana ignorando os -1
    mediana = df[df[col] != -1][col].median()
    df[col] = df[col].replace(-1, mediana)

# 3. Encoding de Variáveis Categóricas
# Transformando colunas de texto em representações numéricas
le = LabelEncoder()
colunas_texto = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
for col in colunas_texto:
    df[col] = le.fit_transform(df[col].astype(str))

# 4. Divisão Temporal (Estratégia de Rigor Científico)
# Treino: Passado (Meses 0 a 5) | Teste: Futuro (Meses 6 e 7)
df_treino = df[df['month'] <= 5]
df_teste = df[df['month'] > 5]

# 5. Shuffle (Embaralhamento) apenas no Treino
# Evita que o modelo aprenda ordens acidentais de inserção no banco
df_treino = shuffle(df_treino, random_state=42)

# 6. Salvando os arquivos finais
print("Salvando arquivos processados...")
df_treino.to_csv('treino.csv', index=False)
df_teste.to_csv('teste.csv', index=False)

print("\n" + "="*30)
print("--- FASE 1 CONCLUÍDA ---")
print(f"Treino (Meses 0-5): {df_treino.shape[0]} linhas")
print(f"Teste  (Meses 6-7): {df_teste.shape[0]} linhas")
print("Arquivos 'treino.csv' e 'teste.csv' prontos")
print("="*30)