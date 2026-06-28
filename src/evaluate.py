"""Agrega os resultados de todas as abordagens numa tabela comparativa e
coleta exemplos de erro para a análise crítica.

Lê todos os `pred_*.json` salvos por baseline.py e classify_llm.py em
data/processed/ e gera:
  - relatorio/analise_resultados.md  (tabela F1-macro + por-classe do melhor)
  - amostra de erros impressa (insumo para relatorio/analise_erros.md)

Funciona mesmo que as abordagens de LLM ainda não tenham sido rodadas (ex.:
sem API key): a tabela mostra só o que existir.
"""
from __future__ import annotations

import json

from sklearn.metrics import f1_score

from . import config

# Ordem e rótulos amigáveis das abordagens.
ROTULOS = {
    "tfidf": "TF-IDF + LogReg (baseline)",
    "embeddings": "Embeddings + LogReg (baseline)",
    "llm_zeroshot": "LLM zero-shot",
    "llm_fewshot": "LLM few-shot",
    "enriquecido": "Embeddings + features do LLM",
}
# Custo (preenchido a partir de custos_llm.md; baseline = R$ 0).
CUSTO_1000 = {
    "tfidf": "R$ 0",
    "embeddings": "R$ 0",
}


def _carregar_preds() -> dict:
    preds = {}
    for arq in sorted(config.DATA_PROC.glob("pred_*.json")):
        nome = arq.stem.replace("pred_", "")
        with open(arq, encoding="utf-8") as f:
            preds[nome] = json.load(f)
    return preds


def tabela_comparativa(preds: dict) -> str:
    linhas = [
        "| Abordagem | F1-macro | n (teste) | Custo / 1000 |",
        "|---|---|---|---|",
    ]
    ordem = [k for k in ROTULOS if k in preds] + [
        k for k in preds if k not in ROTULOS
    ]
    for nome in ordem:
        p = preds[nome]
        rotulo = ROTULOS.get(nome, nome)
        custo = CUSTO_1000.get(nome, "ver custos_llm.md")
        linhas.append(
            f"| {rotulo} | {p['f1_macro']:.4f} | {len(p['y_true'])} | {custo} |"
        )
    return "\n".join(linhas)


def f1_por_classe(pred: dict) -> str:
    rep = pred["report"]
    linhas = ["| Classe | Precision | Recall | F1 | Suporte |", "|---|---|---|---|---|"]
    for classe, m in rep.items():
        if not isinstance(m, dict) or classe in ("accuracy",):
            continue
        linhas.append(
            f"| {classe} | {m['precision']:.2f} | {m['recall']:.2f} | "
            f"{m['f1-score']:.2f} | {int(m['support'])} |"
        )
    return "\n".join(linhas)


def coletar_erros(preds: dict, n: int = 10) -> list[dict]:
    """Coleta exemplos onde o melhor modelo errou (insumo da análise de erro)."""
    erros = []
    for nome, p in preds.items():
        for i, (yt, yp) in enumerate(zip(p["y_true"], p["y_pred"])):
            if yt != yp:
                erros.append(
                    {
                        "abordagem": nome,
                        "id": p["ids"][i],
                        "texto": p["textos"][i],
                        "verdadeiro": yt,
                        "previsto": yp,
                    }
                )
    return erros[: n * len(preds)]


def gerar_relatorio():
    preds = _carregar_preds()
    if not preds:
        print("Nenhum pred_*.json encontrado. Rode baseline.py / classify_llm.py antes.")
        return

    tabela = tabela_comparativa(preds)
    melhor = max(preds.values(), key=lambda p: p["f1_macro"])

    md = [
        "# Análise de Resultados\n",
        "## Tabela comparativa (alvo: `categoria`, 10 classes, F1-macro)\n",
        tabela,
        "\n\n> O baseline clássico é a régua medida **antes** do LLM. ",
        "A pergunta-chave do enunciado — *o LLM melhorou o quê, e a que custo?* — ",
        "é respondida comparando as linhas de LLM com as de baseline acima.\n",
        f"\n## Desempenho por classe — melhor abordagem ({melhor['nome']})\n",
        f1_por_classe(melhor),
        "\n",
    ]
    saida = config.RELATORIO_DIR / "analise_resultados.md"
    saida.write_text("\n".join(md), encoding="utf-8")
    print(f"Relatório salvo em {saida}")
    print("\n" + tabela)


if __name__ == "__main__":
    gerar_relatorio()
