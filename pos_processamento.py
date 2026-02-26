# pos_processamento.py
"""
FASE 4 — PÓS-PROCESSAMENTO E RESULTADOS (Módulo)
objetivo: ajuste de threshold, gráficos, importância de features e SHAP.

Expõe:
  executar() → dict com métricas finais dos dois modelos

Arquivos gerados:
  grafico_roc.png
  grafico_pr.png
  grafico_matriz_confusao_rf.png
  grafico_matriz_confusao_xgb.png
  grafico_importancia_rf.png
  grafico_importancia_xgb.png
  grafico_shap_rf.png
  grafico_shap_xgb.png
  resultado_threshold.csv
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('Agg')  # sem janela — salva direto em arquivo
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, recall_score, precision_score,
    roc_auc_score,
)

try:
    import shap
    SHAP_DISPONIVEL = True
except ImportError:
    SHAP_DISPONIVEL = False
    print("[AVISO] shap não instalado. Execute: pip install shap")

# ── Caminhos ─────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saidas')

RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')

RELATORIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relatorio')
os.makedirs(RELATORIO, exist_ok=True)

CAMINHO_TESTE      = rf'{BASE}\teste.csv'
CAMINHO_RF         = rf'{RESULTADOS}\modelo_rf.pkl'
CAMINHO_XGB        = rf'{RESULTADOS}\modelo_xgb.pkl'
CAMINHO_THRESHOLD  = rf'{RESULTADOS}\resultado_threshold.csv'

TARGET = 'fraud_bool'
SEP    = "=" * 65

def _carregar():
    """Carrega modelos e conjunto de teste."""
    print("[FASE 4] Carregando modelos e teste.csv...")
    df_teste = pd.read_csv(CAMINHO_TESTE)
    features = [c for c in df_teste.columns if c != TARGET]
    X_teste  = df_teste[features]
    y_teste  = df_teste[TARGET]
    rf       = joblib.load(CAMINHO_RF)
    xgb      = joblib.load(CAMINHO_XGB)
    print(f"  Teste : {X_teste.shape[0]:,} linhas | {X_teste.shape[1]} features")
    print(f"  Fraudes no teste: {y_teste.sum():,} ({y_teste.mean()*100:.2f}%)")
    return rf, xgb, X_teste, y_teste, features



def _ajuste_threshold(nome, y_teste, y_prob):
    """
    Testa thresholds de 0.1 a 0.9 e escolhe o que maximiza o F1-Fraude.
    Retorna o melhor threshold e as métricas associadas.
    """
    print(f"\n[THRESHOLD] Buscando threshold ótimo para {nome}...")
    thresholds = np.arange(0.1, 0.91, 0.01)
    resultados = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        resultados.append({
            'threshold' : round(t, 2),
            'f1'        : f1_score(y_teste, y_pred, zero_division=0),
            'recall'    : recall_score(y_teste, y_pred, zero_division=0),
            'precision' : precision_score(y_teste, y_pred, zero_division=0),
        })

    df_t = pd.DataFrame(resultados)
    melhor = df_t.loc[df_t['f1'].idxmax()]

    print(f"  Threshold padrão (0.50) → F1: {df_t[df_t['threshold']==0.50]['f1'].values[0]:.4f}")
    print(f"  Threshold ótimo  ({melhor['threshold']:.2f}) → "
          f"F1: {melhor['f1']:.4f} | "
          f"Recall: {melhor['recall']:.4f} | "
          f"Precision: {melhor['precision']:.4f}")

    # Gráfico threshold vs métricas
    plt.figure(figsize=(10, 5))
    plt.plot(df_t['threshold'], df_t['f1'],        label='F1',        linewidth=2)
    plt.plot(df_t['threshold'], df_t['recall'],    label='Recall',    linewidth=2)
    plt.plot(df_t['threshold'], df_t['precision'], label='Precision', linewidth=2)
    plt.axvline(melhor['threshold'], color='red', linestyle='--',
                label=f"Ótimo = {melhor['threshold']:.2f}")
    plt.title(f'{nome} — Threshold vs Métricas')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    nome_arquivo = nome.lower().replace(' ', '_')
    plt.savefig(rf'{RELATORIO}\grafico_threshold_{nome_arquivo}.png', dpi=150)
    plt.close()
    print(f"  [OK] grafico_threshold_{nome_arquivo}.png salvo")

    return melhor['threshold'], df_t



def _grafico_roc(modelos_prob, y_teste):
    """Curva ROC dos dois modelos no mesmo gráfico."""
    plt.figure(figsize=(8, 6))
    for nome, y_prob in modelos_prob.items():
        fpr, tpr, _ = roc_curve(y_teste, y_prob)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{nome} (AUC={auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Aleatório')
    plt.title('Curva ROC — RF vs XGBoost')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(rf'{RELATORIO}\grafico_roc.png', dpi=150)
    plt.close()
    print("[OK] grafico_roc.png salvo")



def _grafico_pr(modelos_prob, y_teste):
    """Curva Precision-Recall — mais informativa para classes desbalanceadas."""
    plt.figure(figsize=(8, 6))
    for nome, y_prob in modelos_prob.items():
        precision, recall, _ = precision_recall_curve(y_teste, y_prob)
        ap = average_precision_score(y_teste, y_prob)
        plt.plot(recall, precision, linewidth=2, label=f'{nome} (AUC-PR={ap:.4f})')
    baseline = y_teste.mean()
    plt.axhline(baseline, color='gray', linestyle='--',
                label=f'Baseline aleatório ({baseline:.3f})')
    plt.title('Curva Precision-Recall — RF vs XGBoost')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(rf'{RELATORIO}\grafico_pr.png', dpi=150)
    plt.close()
    print("[OK] grafico_pr.png salvo")



def _grafico_matriz(nome, y_teste, y_pred):
    """Matriz de confusão normalizada + absoluta."""
    cm = confusion_matrix(y_teste, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, titulo in zip(
            axes,
            [cm, cm_norm],
            ['d', '.2%'],
            ['Absoluta', 'Normalizada (por linha)']
    ):
        im = ax.imshow(data, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Legítimo', 'Fraude'])
        ax.set_yticklabels(['Legítimo', 'Fraude'])
        ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
        ax.set_title(f'{nome} — Matriz de Confusão {titulo}')
        for i in range(2):
            for j in range(2):
                texto = format(data[i, j], fmt)
                cor   = 'white' if data[i, j] > data.max() / 2 else 'black'
                ax.text(j, i, texto, ha='center', va='center',
                        fontsize=14, color=cor, fontweight='bold')

    plt.tight_layout()
    nome_arquivo = nome.lower().replace(' ', '_')
    plt.savefig(rf'{RELATORIO}\grafico_matriz_confusao_{nome_arquivo}.png', dpi=150)
    plt.close()
    print(f"[OK] grafico_matriz_confusao_{nome_arquivo}.png salvo")



def _grafico_importancia(nome, modelo, features, top_n=20):
    """Gráfico de barras horizontais com as top_n features mais importantes."""
    importancias = pd.Series(modelo.feature_importances_, index=features)
    importancias = importancias.sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(10, 7))
    bars = plt.barh(importancias.index, importancias.values, color='steelblue')
    plt.xlabel('Importância')
    plt.title(f'{nome} — Top {top_n} Features Mais Importantes')
    plt.grid(True, axis='x', alpha=0.3)

    # Rótulo com valor dentro/fora da barra
    for bar, val in zip(bars, importancias.values):
        plt.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    nome_arquivo = nome.lower().replace(' ', '_')
    plt.savefig(rf'{RELATORIO}\grafico_importancia_{nome_arquivo}.png', dpi=150)
    plt.close()
    print(f"[OK] grafico_importancia_{nome_arquivo}.png salvo")



def _grafico_shap(nome, modelo, X_teste, features, n_amostras=2000):
    """SHAP summary plot — mostra impacto de cada feature nas predições.
    Usa amostra para não travar com 200k+ linhas.
    """
    if not SHAP_DISPONIVEL:
        print(f"[SHAP] Pulando {nome} — shap não instalado.")
        return

    print(f"[SHAP] Calculando para {nome} ({n_amostras} amostras)...")
    X_sample = X_teste.sample(n=min(n_amostras, len(X_teste)), random_state=42)

    explainer   = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_sample, approximate=True, check_additivity=False)

    # RF retorna lista [classe0, classe1]; XGBoost retorna array direto
    if isinstance(shap_values, list):
        sv = shap_values[1]   # classe fraude
    else:
        sv = shap_values

    plt.figure()
    shap.summary_plot(sv, X_sample, feature_names=features,
                      show=False, plot_size=(12, 8))
    plt.title(f'SHAP — {nome}')
    plt.tight_layout()
    nome_arquivo = nome.lower().replace(' ', '_')
    plt.savefig(rf'{RELATORIO}\grafico_shap_{nome_arquivo}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] grafico_shap_{nome_arquivo}.png salvo")



def executar():
    """
    Pipeline completo da Fase 4.
    Retorna dict com métricas finais (threshold ótimo) dos dois modelos.
    """
    print(SEP)
    print("  FASE 4 — PÓS-PROCESSAMENTO E RESULTADOS")
    print(SEP)

    rf, xgb_model, X_teste, y_teste, features = _carregar()

    y_prob_rf  = rf.predict_proba(X_teste)[:, 1]
    y_prob_xgb = xgb_model.predict_proba(X_teste)[:, 1]

    modelos = {
        'Random Forest': (rf,        y_prob_rf),
        'XGBoost'      : (xgb_model, y_prob_xgb),
    }

    # ── 1. Curvas ROC e PR (comparativo) ────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  1/5 — Curvas ROC e Precision-Recall")
    print(f"{'─'*65}")
    _grafico_roc({'Random Forest': y_prob_rf, 'XGBoost': y_prob_xgb}, y_teste)
    _grafico_pr( {'Random Forest': y_prob_rf, 'XGBoost': y_prob_xgb}, y_teste)

    # ── 2. Threshold ótimo ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  2/5 — Ajuste de Threshold")
    print(f"{'─'*65}")
    resultados_threshold = []
    thresholds_otimos    = {}
    dfs_threshold        = {}

    for nome, (modelo, y_prob) in modelos.items():
        t_otimo, df_t = _ajuste_threshold(nome, y_teste, y_prob)
        thresholds_otimos[nome] = t_otimo
        dfs_threshold[nome]     = df_t
        resultados_threshold.append(df_t.assign(modelo=nome))

    pd.concat(resultados_threshold).to_csv(CAMINHO_THRESHOLD, index=False)
    print(f"\n[OK] resultado_threshold.csv salvo")

    # ── 3. Matrizes de confusão (threshold ótimo) ────────────────────────────
    print(f"\n{'─'*65}")
    print("  3/5 — Matrizes de Confusão (threshold ótimo)")
    print(f"{'─'*65}")
    resultados_finais = {}

    for nome, (modelo, y_prob) in modelos.items():
        t = thresholds_otimos[nome]
        y_pred = (y_prob >= t).astype(int)
        _grafico_matriz(nome, y_teste, y_pred)

        cm       = confusion_matrix(y_teste, y_pred)
        tn, fp, fn, tp = cm.ravel()
        resultados_finais[nome] = {
            'threshold' : t,
            'auc_roc'   : round(roc_auc_score(y_teste, y_prob),           4),
            'auc_pr'    : round(average_precision_score(y_teste, y_prob),  4),
            'f1'        : round(f1_score(y_teste, y_pred),                 4),
            'recall'    : round(recall_score(y_teste, y_pred),             4),
            'precision' : round(precision_score(y_teste, y_pred),          4),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }

    # ── 4. Importância de Features ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  4/5 — Importância de Features")
    print(f"{'─'*65}")
    for nome, (modelo, _) in modelos.items():
        _grafico_importancia(nome, modelo, features)

    # ── 5. SHAP ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  5/5 — SHAP (Explicabilidade)")
    print(f"{'─'*65}")
    for nome, (modelo, _) in modelos.items():
        _grafico_shap(nome, modelo, X_teste, features)

    # ── Tabela comparativa final ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  COMPARATIVO FINAL — com threshold ótimo por modelo")
    print(SEP)
    df_comp = pd.DataFrame(resultados_finais).T.reset_index()
    df_comp.rename(columns={'index': 'modelo'}, inplace=True)
    print(df_comp[['modelo', 'threshold', 'auc_roc', 'auc_pr',
                   'f1', 'recall', 'precision']].to_string(index=False))

    melhor = df_comp.loc[df_comp['auc_pr'].idxmax(), 'modelo']


    return resultados_finais


# ── Execução direta ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    executar()
