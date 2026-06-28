"""Estimativa de custo do LLM para processar 1000 reclamações.

O enunciado exige prever o custo em três modelos Gemini (mesmo que a execução
use outro LLM). Preços oficiais (ai.google.dev/gemini-api/docs/pricing,
consultado em jun/2026, USD por 1M de tokens):

    modelo                     input    output   free tier
    gemini-2.5-flash           0.30     2.50     sim
    gemini-3-flash-preview     0.50     3.00     sim
    gemini-3.1-pro-preview     2.00    12.00     não

Metodologia:
- tokens de ENTRADA: média real do prompt (template + texto da reclamação) sobre
  o conjunto de teste. Se houver GEMINI_API_KEY, usa client.count_tokens (exato);
  senão, estima por heurística (~4 chars/token, típico de PT).
- tokens de SAÍDA: estimativa fixa do JSON do schema (~110 tokens).
- custo_1000 = 1000 * (in_tokens/1e6 * preco_in + out_tokens/1e6 * preco_out).
"""
from __future__ import annotations

from . import config, data

PRECOS = {  # USD por 1M de tokens (input, output)
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

CHARS_POR_TOKEN = 4.0       # heurística para PT quando não há API
OUTPUT_TOKENS_ESTIMADO = 110  # JSON do schema AnaliseReclamacao
USD_BRL = 5.40               # câmbio aproximado para exibir em R$


def _prompts_de_exemplo(qual: str, n: int = 200):
    from .classify_llm import _carregar_template, _itens

    template = _carregar_template(qual)
    df = data.carregar()
    _, df_test = data.split(df)
    itens = _itens(df_test)[:n]
    return [template.format(texto=it["texto"]) for it in itens]


def _media_tokens_entrada(prompts) -> float:
    # Caminho exato (se houver chave):
    try:
        from . import llm_client

        config.get_api_key()
        amostra = prompts[:30]
        total = sum(llm_client.contar_tokens(p) for p in amostra)
        return total / len(amostra)
    except Exception:
        # Heurística por caracteres:
        return sum(len(p) for p in prompts) / len(prompts) / CHARS_POR_TOKEN


def estimar(qual: str = "fewshot"):
    prompts = _prompts_de_exemplo(qual)
    in_tok = _media_tokens_entrada(prompts)
    out_tok = OUTPUT_TOKENS_ESTIMADO

    linhas = [
        f"Tokens de entrada (média, prompt '{qual}'): {in_tok:.0f}",
        f"Tokens de saída (estimado): {out_tok}",
        "",
        "| Modelo | US$/1M in | US$/1M out | Custo 1.000 itens (US$) | Custo 1.000 itens (R$) |",
        "|---|---|---|---|---|",
    ]
    resultados = {}
    for modelo, (pin, pout) in PRECOS.items():
        custo = 1000 * (in_tok / 1e6 * pin + out_tok / 1e6 * pout)
        resultados[modelo] = custo
        linhas.append(
            f"| {modelo} | {pin:.2f} | {pout:.2f} | {custo:.4f} | "
            f"{custo * USD_BRL:.4f} |"
        )
    return "\n".join(linhas), resultados, in_tok, out_tok


if __name__ == "__main__":
    tabela, _, _, _ = estimar("fewshot")
    print(tabela)
