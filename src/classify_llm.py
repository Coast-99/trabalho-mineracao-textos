"""Uso do LLM na classificação — os dois caminhos exigidos pelo enunciado.

(L1) CLASSIFICAÇÃO direta via LLM (zero-shot e few-shot): o LLM prevê a
     `categoria_prevista`. Comparamos o F1-macro com o baseline clássico.

(L2) ENRIQUECIMENTO: a MESMA chamada estruturada também extrai
     tipo_problema, sentimento e urgencia. Usamos esses campos como features
     adicionais (one-hot) concatenadas aos embeddings e treinamos um
     classificador clássico, medindo se o F1 melhora vs. o baseline.

Tudo passa pelo cache JSONL (llm_client.processar_corpus), então rodar de novo
não gasta API. Use LIMITE para amostrar durante o desenvolvimento; deixe None
para a medição final sobre todo o conjunto de teste.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder

from . import config, data, llm_client
from .schemas import AnaliseReclamacao

_PROMPTS = {
    "zeroshot": config.PROMPTS_DIR / "classificacao_zeroshot.txt",
    "fewshot": config.PROMPTS_DIR / "classificacao_fewshot.txt",
}


def _carregar_template(qual: str) -> str:
    return _PROMPTS[qual].read_text(encoding="utf-8")


def _itens(df):
    return [{"id": int(r.id), "texto": r.texto} for r in df.itertuples()]


def extrair(df, qual: str = "zeroshot", limite: int | None = None):
    """Roda a extração estruturada do LLM sobre df. Retorna lista de registros."""
    template = _carregar_template(qual)
    itens = _itens(df)
    if limite:
        itens = itens[:limite]

    def montar_prompt(item):
        return template.format(texto=item["texto"])

    return llm_client.processar_corpus(
        itens, montar_prompt, AnaliseReclamacao, cache_nome=f"cache_llm_{qual}.jsonl"
    )


def _salvar_pred(nome, df_test, registros):
    """Salva no mesmo formato do baseline para o evaluate.py consumir."""
    por_id = {r["id"]: r for r in registros}
    sub = df_test[df_test["id"].isin(por_id)].copy()
    y_true = sub["categoria"].tolist()
    y_pred = [por_id[int(i)]["categoria_prevista"] for i in sub["id"]]

    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    saida = {
        "nome": nome,
        "f1_macro": f1,
        "report": report,
        "ids": sub["id"].tolist(),
        "textos": sub["texto"].tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }
    with open(config.DATA_PROC / f"pred_{nome}.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    print(f"  [{nome}] F1-macro = {f1:.4f}  (n={len(y_true)})")
    return f1


def classificar(df_test, qual: str = "zeroshot", limite: int | None = None):
    """(L1) Classificação direta via LLM."""
    print(f"\n[L1] Classificação LLM ({qual})")
    registros = extrair(df_test, qual=qual, limite=limite)
    return _salvar_pred(f"llm_{qual}", df_test, registros)


def _features_llm(registros, encoder=None, fit=False):
    """One-hot de tipo_problema/sentimento/urgencia extraídos pelo LLM."""
    campos = [
        [r["tipo_problema"], r["sentimento"], r["urgencia"]] for r in registros
    ]
    if fit:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return encoder.fit_transform(campos), encoder
    return encoder.transform(campos), encoder


def enriquecer(df_train, df_test, qual: str = "zeroshot", limite: int | None = None):
    """(L2) Embeddings + features extraídas pelo LLM -> classificador clássico."""
    print(f"\n[L2] Enriquecimento (embeddings + features do LLM, {qual})")
    from . import embeddings

    reg_tr = extrair(df_train, qual=qual, limite=limite)
    reg_te = extrair(df_test, qual=qual, limite=limite)

    # alinha df aos itens efetivamente processados (respeita o limite)
    ids_tr = {r["id"] for r in reg_tr}
    ids_te = {r["id"] for r in reg_te}
    tr = df_train[df_train["id"].isin(ids_tr)].reset_index(drop=True)
    te = df_test[df_test["id"].isin(ids_te)].reset_index(drop=True)
    reg_tr = sorted(reg_tr, key=lambda r: list(tr["id"]).index(r["id"]))
    reg_te = sorted(reg_te, key=lambda r: list(te["id"]).index(r["id"]))

    emb_tr = embeddings.encode(tr["texto"].tolist())
    emb_te = embeddings.encode(te["texto"].tolist())
    f_tr, enc = _features_llm(reg_tr, fit=True)
    f_te, _ = _features_llm(reg_te, encoder=enc)

    X_tr = np.hstack([emb_tr, f_tr])
    X_te = np.hstack([emb_te, f_te])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, tr["categoria"])
    y_pred = clf.predict(X_te)

    f1 = f1_score(te["categoria"], y_pred, average="macro")
    report = classification_report(
        te["categoria"], y_pred, output_dict=True, zero_division=0
    )
    saida = {
        "nome": "enriquecido",
        "f1_macro": f1,
        "report": report,
        "ids": te["id"].tolist(),
        "textos": te["texto"].tolist(),
        "y_true": te["categoria"].tolist(),
        "y_pred": list(y_pred),
    }
    with open(config.DATA_PROC / "pred_enriquecido.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    print(f"  [enriquecido] F1-macro = {f1:.4f}")
    return f1


def main(limite: int | None = None):
    df = data.carregar()
    df_train, df_test = data.split(df)
    classificar(df_test, "zeroshot", limite=limite)
    classificar(df_test, "fewshot", limite=limite)
    enriquecer(df_train, df_test, "zeroshot", limite=limite)


if __name__ == "__main__":
    import sys

    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limite=lim)
