"""Baseline clássico OBRIGATÓRIO — medido ANTES de qualquer chamada ao LLM.

Treina classificadores clássicos para prever a `categoria` (10 classes) e
registra o F1-macro. É a régua contra a qual o ganho do LLM será medido
("o LLM melhorou o quê, exatamente, e a que custo?").

Duas abordagens (técnicas das Aulas 2, 3 e 5):
  (b1) TF-IDF sobre texto pré-processado + Regressão Logística
  (b2) Embeddings multilíngues + Regressão Logística

Como o texto é curto e as classes desbalanceadas, usamos:
  - F1-macro como métrica principal (cada classe pesa igual);
  - split estratificado fixo (src/data.py);
  - class_weight="balanced" na regressão.

Os resultados (y_true, y_pred, métricas) são salvos em data/processed/ para o
evaluate.py montar a tabela comparativa e a análise de erro.
"""
from __future__ import annotations

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from . import config, data, preprocess


def _salvar_resultado(nome: str, df_test, y_pred, f1: float, report: dict) -> None:
    saida = {
        "nome": nome,
        "f1_macro": f1,
        "report": report,
        "ids": df_test["id"].tolist(),
        "textos": df_test["texto"].tolist(),
        "y_true": df_test["categoria"].tolist(),
        "y_pred": list(y_pred),
    }
    caminho = config.DATA_PROC / f"pred_{nome}.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    print(f"  -> salvo em {caminho.name}")


def baseline_tfidf(df_train, df_test):
    """(b1) TF-IDF + Regressão Logística sobre texto pré-processado."""
    print("\n[b1] TF-IDF + LogisticRegression")
    X_train = preprocess.limpar_serie(df_train["texto"])
    X_test = preprocess.limpar_serie(df_test["texto"])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, df_train["categoria"])
    y_pred = clf.predict(Xte)

    f1 = f1_score(df_test["categoria"], y_pred, average="macro")
    report = classification_report(
        df_test["categoria"], y_pred, output_dict=True, zero_division=0
    )
    print(f"  F1-macro = {f1:.4f}")
    _salvar_resultado("tfidf", df_test, y_pred, f1, report)
    return f1


def baseline_embeddings(df_train, df_test):
    """(b2) Embeddings multilíngues + Regressão Logística sobre texto cru."""
    print("\n[b2] Embeddings + LogisticRegression")
    from . import embeddings

    Xtr = embeddings.encode(df_train["texto"].tolist())
    Xte = embeddings.encode(df_test["texto"].tolist())

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, df_train["categoria"])
    y_pred = clf.predict(Xte)

    f1 = f1_score(df_test["categoria"], y_pred, average="macro")
    report = classification_report(
        df_test["categoria"], y_pred, output_dict=True, zero_division=0
    )
    print(f"  F1-macro = {f1:.4f}")
    _salvar_resultado("embeddings", df_test, y_pred, f1, report)
    return f1


def main():
    df = data.carregar()
    df_train, df_test = data.split(df)
    print(f"Treino: {len(df_train)} | Teste: {len(df_test)}")

    f1_tfidf = baseline_tfidf(df_train, df_test)
    f1_emb = baseline_embeddings(df_train, df_test)

    print("\n=== Resumo do baseline (F1-macro) ===")
    print(f"  TF-IDF + LogReg      : {f1_tfidf:.4f}")
    print(f"  Embeddings + LogReg  : {f1_emb:.4f}")


if __name__ == "__main__":
    main()
