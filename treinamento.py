"""
FASE 3 — TREINAMENTO E AVALIAÇÃO DOS MODELOS (Módulo)
objetivo: treinar e comparar Random Forest vs. XGBoost para detecção de fraudes.

Por que esses dois modelos?
  Random Forest  — conjunto de árvores independentes; robusto, interpreta features bem.
  XGBoost        — boosting sequencial; geralmente mais preciso em dados tabulares
                   e muito usado em competições de fraude bancária.

Métricas utilizadas (focadas em dados desbalanceados):
  AUC-ROC   — separação geral entre classes (independe do threshold)
  AUC-PR    — Área sob Precision-Recall; mais informativa quando fraude é rara
  F1-Fraude — harmônica entre Precision e Recall para a classe minoritária
  Recall    — % de fraudes que o modelo detectou (custo mais alto é perder fraude)
  Precision — % das alertas que realmente são fraude (evitar falsos alarmes)

Fluxo:
  1. Carregar treino_balanceado.csv e teste.csv
  2. Treinar Random Forest
  3. Treinar XGBoost
  4. Avaliar ambos no teste (dados reais, desbalanceados)
  5. Salvar modelos (.pkl) e tabela comparativa (.csv)
  6. Imprimir tabela de comparação final

Este módulo expõe uma função chamada pelo pipeline (main.py):
  treinar_e_avaliar() → dict com resultados de ambos os modelos
"""

import pandas as pd
import numpy as np
import time
import joblib
import contextlib
import warnings
warnings.filterwarnings('ignore')
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_DISPONIVEL = True
except ImportError:
    XGBOOST_DISPONIVEL = False
    print("[AVISO] xgboost não instalado. Execute: pip install xgboost")


# ─────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _tqdm_joblib(tqdm_obj):
    """
    Conecta o tqdm ao backend paralelo do joblib (usado pelo RandomForest).
    A cada lote de árvores concluído, a barra avança automaticamente.
    """
    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_obj.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_obj
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_obj.close()


class _XGBTqdmCallback(xgb.callback.TrainingCallback if XGBOOST_DISPONIVEL else object):
    """Callback que atualiza uma barra tqdm a cada rodada do XGBoost."""
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="  XGBoost", unit="round",
                         bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

BASE              = os.path.dirname(os.path.abspath(__file__))
RESULTADOS        = os.path.join(BASE, 'resultados')
SAIDAS        = os.path.join(BASE, 'saidas')
os.makedirs(RESULTADOS, exist_ok=True)

CAMINHO_TREINO    = os.path.join(RESULTADOS,       'treino_balanceado.csv')
CAMINHO_TESTE     = os.path.join(SAIDAS,       'teste.csv')
CAMINHO_RF_MODEL  = os.path.join(RESULTADOS,       'modelo_rf.pkl')
CAMINHO_XGB_MODEL = os.path.join(RESULTADOS,       'modelo_xgb.pkl')
CAMINHO_RESULTADO = os.path.join(RESULTADOS, 'resultado_modelos.csv')

TARGET = 'fraud_bool'
SEP    = "=" * 65


# ─────────────────────────────────────────────────────────────────────────────
def _carregar_dados():
    """Carrega treino balanceado e teste, separa features/target."""
    print(f"[TREINO] Carregando treino_balanceado.csv...")
    df_treino = pd.read_csv(CAMINHO_TREINO)
    print(f"[TREINO] Carregando teste.csv...")
    df_teste  = pd.read_csv(CAMINHO_TESTE)

    features = [c for c in df_treino.columns if c != TARGET]

    X_treino = df_treino[features]
    y_treino = df_treino[TARGET]
    X_teste  = df_teste[features]
    y_teste  = df_teste[TARGET]

    print(f"\n  Treino (balanceado) : {X_treino.shape[0]:>8,} linhas | {X_treino.shape[1]} features")
    print(f"  Fraudes treino      : {y_treino.sum():>8,} ({y_treino.mean()*100:.1f}%)")
    print(f"  Teste  (real)       : {X_teste.shape[0]:>8,} linhas")
    print(f"  Fraudes teste       : {y_teste.sum():>8,} ({y_teste.mean()*100:.2f}%)")

    return X_treino, y_treino, X_teste, y_teste, features


# ─────────────────────────────────────────────────────────────────────────────
def _avaliar(nome, modelo, X_teste, y_teste, y_prob):
    """Calcula e imprime todas as métricas para um modelo."""
    y_pred = modelo.predict(X_teste)

    auc_roc   = roc_auc_score(y_teste, y_prob)
    auc_pr    = average_precision_score(y_teste, y_prob)
    f1        = f1_score(y_teste, y_pred)
    precision = precision_score(y_teste, y_pred)
    recall    = recall_score(y_teste, y_pred)
    cm        = confusion_matrix(y_teste, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)  # taxa de falsos positivos

    print(f"\n{'─'*65}")
    print(f"  RESULTADO — {nome}")
    print(f"{'─'*65}")
    print(f"  AUC-ROC   : {auc_roc:.4f}   (>0.90 = excelente)")
    print(f"  AUC-PR    : {auc_pr:.4f}   (>0.50 é bom para fraude rara)")
    print(f"  F1-Fraude : {f1:.4f}")
    print(f"  Recall    : {recall:.4f}  ← % de fraudes detectadas")
    print(f"  Precision : {precision:.4f}  ← % dos alertas que são fraude")
    print(f"  FPR       : {fpr:.4f}  ← falsos alarmes sobre legítimos")
    print(f"\n  Matriz de Confusão:")
    print(f"             Previsto Legít.  Previsto Fraude")
    print(f"  Real Legít.     {tn:>8,}        {fp:>8,}")
    print(f"  Real Fraude     {fn:>8,}        {tp:>8,}")
    print(f"\n  Relatório detalhado:")
    print(classification_report(y_teste, y_pred,
                                target_names=['Legítimo', 'Fraude'],
                                digits=4))

    return {
        'modelo'   : nome,
        'auc_roc'  : round(auc_roc,   4),
        'auc_pr'   : round(auc_pr,    4),
        'f1'       : round(f1,        4),
        'recall'   : round(recall,    4),
        'precision': round(precision, 4),
        'fpr'      : round(fpr,       4),
        'tp'       : int(tp),
        'fp'       : int(fp),
        'tn'       : int(tn),
        'fn'       : int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
def treinar_e_avaliar():
    """
    Pipeline completo da Fase 3:
      - Carrega dados
      - Treina RF e XGBoost
      - Avalia no conjunto de teste
      - Salva modelos e resultados
    Retorna dict com métricas dos dois modelos.
    """
    print(SEP)
    print("  FASE 3 — TREINAMENTO E AVALIAÇÃO DOS MODELOS")
    print(SEP)

    # ── 1. Carregar dados ───────────────────────────────────────────────────
    X_treino, y_treino, X_teste, y_teste, features = _carregar_dados()
    resultados = []

    # ── 2. Random Forest ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  MODELO 1/2 — Random Forest")
    print(SEP)
    print("  Parâmetros: 300 árvores, profundidade máx. 15, class_weight=balanced")
    print("  Treinando...")
    t0 = time.time()

    n_est_rf = 300
    rf = RandomForestClassifier(
        n_estimators     = n_est_rf,
        max_depth        = 15,
        min_samples_leaf = 10,
        class_weight     = 'balanced',
        random_state     = 42,
        n_jobs           = -1,
    )
    with _tqdm_joblib(tqdm(total=n_est_rf, desc="  Random Forest", unit="árvore",
                           bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")):
        rf.fit(X_treino, y_treino)
    tempo_rf = time.time() - t0
    print(f"  [OK] Treinado em {tempo_rf:.1f}s")

    y_prob_rf  = rf.predict_proba(X_teste)[:, 1]
    res_rf     = _avaliar("Random Forest", rf, X_teste, y_teste, y_prob_rf)
    res_rf['tempo_treino_s'] = round(tempo_rf, 1)
    resultados.append(res_rf)

    joblib.dump(rf, CAMINHO_RF_MODEL)
    print(f"  [OK] Modelo salvo: modelo_rf.pkl")

    # ── 3. XGBoost ──────────────────────────────────────────────────────────
    if not XGBOOST_DISPONIVEL:
        print("\n[AVISO] XGBoost não disponível. Pulando modelo 2/2.")
    else:
        print(f"\n{SEP}")
        print("  MODELO 2/2 — XGBoost")
        print(SEP)

        # scale_pos_weight equilibra as classes no XGBoost
        # (usa os dados de treino balanceado após SMOTE)
        n_neg = int((y_treino == 0).sum())
        n_pos = int((y_treino == 1).sum())
        spw   = round(n_neg / n_pos, 2)
        print(f"  scale_pos_weight = {n_neg:,} / {n_pos:,} = {spw}")
        print(f"  Parâmetros: 300 estimators, lr=0.05, max_depth=6")
        print(f"  Treinando...")
        t0 = time.time()

        n_est_xgb = 300
        xgb_model = XGBClassifier(
            n_estimators      = n_est_xgb,
            max_depth         = 6,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            scale_pos_weight  = spw,
            eval_metric       = 'aucpr',
            random_state      = 42,
            n_jobs            = -1,
            verbosity         = 0,
            callbacks         = [_XGBTqdmCallback(n_est_xgb)],
        )
        xgb_model.fit(X_treino, y_treino)
        tempo_xgb = time.time() - t0
        print(f"  [OK] Treinado em {tempo_xgb:.1f}s")

        y_prob_xgb = xgb_model.predict_proba(X_teste)[:, 1]
        res_xgb    = _avaliar("XGBoost", xgb_model, X_teste, y_teste, y_prob_xgb)
        res_xgb['tempo_treino_s'] = round(tempo_xgb, 1)
        resultados.append(res_xgb)

        # Remove callbacks antes de salvar (tqdm não é serializável pelo pickle)
        xgb_model.set_params(callbacks=None)
        joblib.dump(xgb_model, CAMINHO_XGB_MODEL)
        print(f"  [OK] Modelo salvo: modelo_xgb.pkl")

    # ── 4. Tabela comparativa ───────────────────────────────────────────────
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(CAMINHO_RESULTADO, index=False)

    print(f"\n{SEP}")
    print("  COMPARATIVO FINAL — Random Forest vs. XGBoost")
    print(SEP)
    print(df_res[[
        'modelo', 'auc_roc', 'auc_pr', 'f1', 'recall', 'precision', 'fpr', 'tempo_treino_s'
    ]].to_string(index=False))
    print(f"\n  Tabela salva: resultado_modelos.csv")

    # ── 5. Determinar vencedor ──────────────────────────────────────────────
    if len(resultados) > 1:
        melhor = df_res.loc[df_res['auc_pr'].idxmax(), 'modelo']
        print(f"\n  Melhor modelo (AUC-PR): {melhor}")
        print(f"  → Recomendado para Fase 4 (ajuste fino / interpretabilidade)")

    print(f"\n{SEP}")
    print("  Fase 3 concluída.")
    print(f"  Próximo passo: Fase 4 → ajuste de threshold + SHAP (explicabilidade)")
    print(SEP)

    return {r['modelo']: r for r in resultados}


# ── Execução direta (sem pipeline) ──────────────────────────────────────────
if __name__ == '__main__':
    treinar_e_avaliar()
