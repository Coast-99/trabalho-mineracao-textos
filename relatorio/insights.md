# Três insights acionáveis para o gestor

Insights derivados do corpus de 2.935 reclamações de consumidor (números reais,
reproduzíveis pelo notebook de demo e pelos módulos em `src/`).

## Insight 1 — Saneamento, Educação e Transporte Aéreo são os gargalos de resolução
**Dado:** a taxa de reclamações *não resolvidas / sem resposta* varia muito por
setor: **Saneamento 50,0%**, **Educação 47,5%**, **Transporte Aéreo 46,1%** —
contra Saúde-Planos (34,2%) e Imobiliário (34,4%).
**Ação:** priorizar auditoria de SLA e cobrança de resposta junto às empresas
desses três setores; são onde cada real investido em mediação rende mais.

## Insight 2 — O volume concentra-se no Sudeste, mas a régua de qualidade deve ser por setor
**Dado:** 42% das reclamações vêm do **Sudeste** (1.238), seguido do Nordeste
(730). Telecomunicações sozinha responde por 27% de todo o corpus.
**Ação:** dimensionar a equipe de atendimento por região (capacidade no Sudeste),
mas definir metas de resolução **por setor**, já que o problema de resolução
(Insight 1) não acompanha o volume.

## Insight 3 — ~9% das manifestações são "ruído" e travam a automação
**Dado:** 8,7% dos textos têm ≤15 caracteres (ex.: "...", "ruim", emojis) e
concentram quase todo o erro do classificador automático (que acerta 92% no
geral, mas falha justamente nesses casos sem conteúdo).
**Ação:** adicionar no formulário do canal uma validação mínima de conteúdo
(tamanho/qualidade do texto) e uma **rota de baixa confiança** que envia o caso
para análise humana ou para o LLM. Isso eleva tanto a automação quanto a
qualidade dos dados de entrada — com custo de LLM previsto de **~R$ 2 a R$ 11
por mil reclamações** (ver `custos_llm.md`), aplicável só à fatia ambígua.

---
> Como reproduzir: ver seção de EDA em `notebooks/01_demo.ipynb` e
> `python -m src.evaluate` / `python -m src.rag`.
