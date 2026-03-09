# Classificação de Fraudes Bancárias

Trabalho Final de Mineração de Dados — Detecção de Fraudes Bancárias com Random Forest e XGBoost sobre o dataset **Bank Account Fraud (BAF) — NeurIPS 2022**.

O projeto implementa um pipeline completo de aprendizado de máquina, cobrindo pré-processamento, seleção de atributos, balanceamento de classes (SMOTE), treinamento supervisionado e pós-processamento com ajuste de threshold, gráficos e análise SHAP.

Link para o relatório: https://docs.google.com/document/d/1DD9ghvd8KdMPAQwmdICQVRseYXN095Sz5jzTq66XpSg/edit?usp=sharing

Link para o repositório github: https://github.com/lordmary/Classificacao-de-Fraudes-Bancarias

---

## Sumário

1. [Pré-requisitos](#pré-requisitos)
2. [Instalação das dependências](#instalação-das-dependências)
3. [Obtenção do dataset](#obtenção-do-dataset)
4. [Estrutura do projeto](#estrutura-do-projeto)
5. [Execução](#execução)
6. [Etapas do pipeline](#etapas-do-pipeline)
7. [Modelos e parâmetros](#modelos-e-parâmetros)
8. [Métricas utilizadas](#métricas-utilizadas)
9. [Arquivos gerados](#arquivos-gerados)

---

## Pré-requisitos

- **Python 3.9 ou superior** (recomendado: 3.10+)
- **pip** atualizado

Verifique sua versão de Python no terminal:

```bash
python --version
```

---

## Instalação das dependências

### 1. (Opcional, mas recomendado) Crie um ambiente virtual

```bash
# Criar o ambiente
python -m venv venv

# Ativar — Windows PowerShell
.\venv\Scripts\Activate.ps1

# Ativar — Windows CMD
venv\Scripts\activate.bat

# Ativar — Linux / macOS
source venv/bin/activate
```

### 2. Instale todos os pacotes necessários

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib tqdm matplotlib shap
```

| Pacote | Uso no projeto |
|---|---|
| `pandas` | Leitura e manipulação dos dados |
| `numpy` | Operações numéricas |
| `scikit-learn` | Pré-processamento, seleção de atributos, métricas, Random Forest |
| `imbalanced-learn` | Balanceamento de classes com SMOTE |
| `xgboost` | Classificador XGBoost |
| `joblib` | Serialização dos modelos treinados (`.pkl`) |
| `tqdm` | Barras de progresso durante o treinamento |
| `matplotlib` | Geração de gráficos |
| `shap` | Interpretabilidade dos modelos (opcional) |

> `shap` é opcional. Se não estiver instalado, os gráficos SHAP são ignorados e uma mensagem de aviso é exibida.

---

## Obtenção do dataset

1. Acesse o dataset no Kaggle:
   **[Bank Account Fraud Dataset Suite (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)**

2. Faça o download do arquivo `.zip` e coloque-o dentro da pasta `base/` do repositório:

```
Classificacao-de-Fraudes-Bancarias/
└── base/
    └── archive.zip   ← coloque aqui
```

> **O pipeline descompacta automaticamente.** Na primeira execução, `pre_processamento.py` detecta o `.zip` em `base/`, extrai o `Base.csv` e segue normalmente. Nas execuções seguintes o CSV já existe e a etapa é pulada.
>
> O `Base.csv` extraído está no `.gitignore` e não é versionado.

---

## Estrutura do projeto

```
Classificacao-de-Fraudes-Bancarias/
│
├── base/                    # Coloque o .zip do dataset aqui
│   ├── .gitignore           # Impede que Base.csv seja commitado
│   └── archive.zip          # Dataset original (baixar do Kaggle)
│
├── saidas/                  # Gerado na execução: teste.csv e CSVs de resultado
├── resultados/              # Gerado na execução: modelos .pkl, treino balanceado e gráficos
│
├── main.py                  # Ponto de entrada — executa o pipeline completo
├── pre_processamento.py     # Fase 2 — limpeza, encoding, divisão treino/teste
├── selecao_atributos.py     # Fase 2 — seleção de features (3 métodos)
├── balanceamento.py         # Fase 2 — balanceamento com SMOTE
├── treinamento.py           # Fase 3 — treino e avaliação de RF e XGBoost
└── pos_processamento.py     # Fase 4 — threshold, gráficos e SHAP
```

---

## Execução

Com o ambiente ativado e o `.zip` na pasta `base/`, execute a partir da raiz do repositório:

```bash
python main.py
```

O pipeline completo (Fases 2, 3 e 4) roda em sequência. O tempo total varia de **15 a 40 minutos** dependendo do hardware, principalmente nas etapas de SMOTE e treinamento dos modelos.

---

## Etapas do pipeline

| Etapa | Módulo | Descrição |
|---|---|---|
| 1 | `pre_processamento.py` | Extrai `Base.csv` do `.zip` se necessário, lê o arquivo (1.000.000 linhas), trata valores `-1` (imputação por mediana) e aplica `LabelEncoder` em 5 colunas categóricas |
| 2 | `selecao_atributos.py` | Identifica features irrelevantes via **Correlação de Pearson**, **Importância por Random Forest** e **Filtro de Variância Baixa** |
| 3 | `pre_processamento.py` | Remove as features selecionadas e divide temporalmente: **meses 0–5 → treino** (~795k linhas, 79,5%), **meses 6–7 → teste** (~205k linhas, 20,5%) |
| 4 | `balanceamento.py` | Aplica **SMOTE** com `sampling_strategy=0.3` no treino (fraudes = 30% dos legítimos), nunca toca o conjunto de teste |
| 5 | `treinamento.py` | Treina **Random Forest** (300 árvores) e **XGBoost** (300 estimators), avalia no teste real (desbalanceado) e salva os modelos |
| 6 | `pos_processamento.py` | Ajusta **threshold ótimo** (0.1–0.9, maximizando F1), gera gráficos ROC/PR/matriz de confusão/importância de features e análise **SHAP** |

---

## Modelos e parâmetros

### Random Forest
| Parâmetro | Valor | Justificativa |
|---|---|---|
| `n_estimators` | **300** | Melhor AUC-PR verificado empiricamente via `warm_start` (ver tabela abaixo) |
| `max_depth` | 15 | Limita overfitting mantendo capacidade de capturar padrões complexos |
| `min_samples_leaf` | 10 | Evita folhas com muito poucos exemplos |
| `class_weight` | `balanced` | Compensa o desbalanceamento residual após SMOTE |
| `n_jobs` | -1 | Usa todos os núcleos disponíveis |

**Validação do `n_estimators` via `warm_start`** — árvores adicionadas em passos de 50, parada quando AUC-PR não melhora por 2 passos consecutivos:

| n_estimators | AUC-PR | AUC-ROC |
|:---:|:---:|:---:|
| 50 | 0.0915 | 0.8356 |
| 100 | 0.0919 | 0.8363 |
| 150 | 0.0915 | 0.8371 |
| 200 | 0.0922 | 0.8373 |
| 250 | 0.0926 | 0.8376 |
| **300** | **0.0929** | **0.8378** ← melhor |
| 350 | 0.0928 | 0.8379 |
| 400 | 0.0927 | 0.8381 |

> Parada antecipada após n=400 (sem melhora de AUC-PR por 2 passos). O valor **300** apresentou o maior AUC-PR e foi o escolhido.

### XGBoost
| Parâmetro | Valor | Justificativa |
|---|---|---|
| `n_estimators` | 300 | Boosting sequencial; mais rounds = mais refinamento |
| `max_depth` | 6 | Árvores rasas são padrão em boosting para evitar overfitting |
| `learning_rate` | 0.05 | Taxa conservadora para convergência estável |
| `subsample` | 0.8 | Amostragem de linhas por árvore — reduz variância |
| `colsample_bytree` | 0.8 | Amostragem de features por árvore |
| `scale_pos_weight` | dinâmico | `n_negativos / n_positivos` do treino balanceado |
| `eval_metric` | `aucpr` | Otimiza diretamente a métrica mais relevante para fraude |

---

## Métricas utilizadas

| Métrica | Por que usá-la |
|---|---|
| **AUC-ROC** | Mede a separação geral entre classes, independente do threshold |
| **AUC-PR** | Área sob a curva Precision-Recall — mais informativa quando fraude é rara (~1–2% do total) |
| **F1-Fraude** | Média harmônica entre Precision e Recall da classe minoritária |
| **Recall** | % de fraudes reais detectadas — o custo de perder uma fraude é alto |
| **Precision** | % dos alertas que são fraude de fato — evita excesso de falsos alarmes |
| **FPR** | Taxa de falsos positivos sobre transações legítimas |

---

## Arquivos gerados

| Caminho | Descrição |
|---|---|
| `saidas/teste.csv` | Dados de teste (meses 6–7), nunca modificados |
| `saidas/resultado_selecao_atributos.csv` | Ranking completo das features pelos 3 métodos |
| `saidas/resultado_modelos.csv` | Comparativo de métricas com threshold padrão (0.50) |
| `resultados/treino_balanceado.csv` | Dados de treino após SMOTE |
| `resultados/modelo_rf.pkl` | Modelo Random Forest serializado |
| `resultados/modelo_xgb.pkl` | Modelo XGBoost serializado |
| `resultados/resultado_threshold.csv` | Métricas para cada threshold testado (0.10–0.90) |
| `resultados/grafico_roc.png` | Curva ROC comparativa RF vs XGBoost |
| `resultados/grafico_pr.png` | Curva Precision-Recall comparativa |
| `resultados/grafico_threshold_*.png` | F1/Recall/Precision × threshold por modelo |
| `resultados/grafico_matriz_confusao_*.png` | Matrizes de confusão (absoluta + normalizada) |
| `resultados/grafico_importancia_*.png` | Top 20 features mais importantes por modelo |
| `resultados/grafico_shap_*.png` | SHAP summary plot por modelo (se `shap` instalado) |
