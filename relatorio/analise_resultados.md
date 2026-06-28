# Análise de Resultados

## Tabela comparativa (alvo: `categoria`, 10 classes, F1-macro)

Duas medições: o **baseline completo** (todo o conjunto de teste, n=587) e a
comparação **com o LLM** numa amostra real de n=32 (ver nota sobre o free tier).
Para a comparação ser justa, os baselines foram recalculados **nos mesmos 32 itens**.

### Baseline completo (n=587) — medido antes do LLM
| Abordagem | F1-macro | n | Custo/1000 |
|---|---|---|---|
| TF-IDF + LogReg | **0.9167** | 587 | R$ 0 |
| Embeddings + LogReg | 0.8752 | 587 | R$ 0 |

### Comparação com o LLM (mesmos 32 itens)
| Abordagem | F1-macro | n | Custo/1000 |
|---|---|---|---|
| TF-IDF + LogReg | 0.9821 | 32 | R$ 0 |
| Embeddings + LogReg | 0.8155 | 32 | R$ 0 |
| **LLM zero-shot (`gemini-2.5-flash-lite`)** | **1.0000** | 32 | ~R$ 2,1 |

> **O LLM melhorou o quê, e a que custo?**
> Na amostra avaliada, o LLM zero-shot acertou **todas** as 32 categorias (F1=1,00),
> ligeiramente acima do TF-IDF (0,98) — **sem usar nenhum dado rotulado de treino**.
> O custo é ~R$ 2 a R$ 11 por 1.000 itens (ver `custos_llm.md`) e está sujeito ao
> limite de requisições do provedor. Como o baseline já é muito forte (0,92 no
> conjunto inteiro), o ganho marginal de acurácia do LLM **não compensa o custo**
> para classificação pura; o real valor do LLM está na **extração estruturada**
> (empresa, tipo de problema, sentimento, urgência) que o classificador clássico
> não fornece — e que alimenta o enriquecimento e os insights ao gestor.

## Desempenho por classe (TF-IDF, conjunto completo n=587)
Ver `pred_tfidf.json`. F1-macro 0.9167; os erros concentram-se em textos-ruído
("...", "..!", emoji, HTML) — detalhado em `analise_erros.md`.

## ⚠️ Nota metodológica sobre o free tier (transparência)
A avaliação do LLM no conjunto completo (ou na amostra planejada de 150) **não foi
concluída** porque o free tier do Gemini impõe um limite diário/por-minuto de
requisições muito baixo para esta conta: o processamento em lote esbarrou em erros
`429 RESOURCE_EXHAUSTED` (com `retryDelay`), reduzindo o rendimento a ~1 item/min e
levando parte das chamadas ao fallback. Por isso reportamos n=32 (chamadas reais
bem-sucedidas, sem fallback).

O pipeline está **pronto para rodar no conjunto completo** sem mudança de código —
basta uma chave com cota adequada (tier pago) ou aguardar a renovação diária:
```
python -m src.classify_llm        # cache JSONL retoma de onde parou
python -m src.evaluate
```
Esse limite **é, em si, um resultado de engenharia relevante** e está refletido na
seção de custos: para processar 1.000+ reclamações, o free tier não basta.
