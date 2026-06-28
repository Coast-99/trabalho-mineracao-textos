# Custos previstos com o LLM

Estimativa do custo para processar **1.000 reclamações** (uma chamada estruturada
por reclamação), conforme exigido no enunciado, nos três modelos Gemini indicados.
Preços oficiais (ai.google.dev/gemini-api/docs/pricing, consultado em jun/2026),
em USD por 1 milhão de tokens.

## Metodologia
- **Tokens de entrada:** média real do prompt few-shot (template + texto da
  reclamação) sobre o conjunto de teste = **~402 tokens**. Estimados por
  heurística (~4 chars/token, típico para PT) quando não há API; com a variável
  `GEMINI_API_KEY` definida, `src/custos.py` usa `client.count_tokens` para a
  contagem exata.
- **Tokens de saída:** ~110 tokens (JSON do schema `AnaliseReclamacao`).
- **Câmbio:** US$ 1 = R$ 5,40 (referência).
- **Fórmula:** `custo_1000 = 1000 × (in/1e6 × preço_in + out/1e6 × preço_out)`.

## Resultado — custo para 1.000 reclamações

| Modelo | US$/1M in | US$/1M out | Custo 1.000 itens (US$) | Custo 1.000 itens (R$) |
|---|---|---|---|---|
| gemini-2.5-flash | 0.30 | 2.50 | 0.3957 | 2.14 |
| gemini-3-flash-preview | 0.50 | 3.00 | 0.5312 | 2.87 |
| gemini-3.1-pro-preview | 2.00 | 12.00 | 2.1246 | 11.47 |

> **Observações**
> - `gemini-2.5-flash` e `gemini-3-flash-preview` têm *free tier* no Google AI
>   Studio — para ~3 mil reclamações o custo real tende a R$ 0 dentro da cota.
> - O *batch API* dá 50% de desconto e o *context caching* até 90% no input.
> - O prompt **zero-shot** é mais barato (sem os exemplos); usamos o **few-shot**
>   aqui por ser o cenário mais caro (limite superior de custo).
> - Reproduzível com `python -m src.custos`.
