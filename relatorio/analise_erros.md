# Análise de erro estruturada

Análise das falhas observadas na tarefa de classificar a `categoria` das
reclamações. Os erros do **baseline clássico** (TF-IDF + LogReg, F1-macro 0.92)
são reais e foram extraídos de `data/processed/pred_tfidf.json`. As falhas
específicas do **LLM** (alucinação) serão preenchidas após a execução com
`GEMINI_API_KEY` — o procedimento e as categorias já estão definidos abaixo.

## Visão geral
- O baseline erra **55 de 587** casos de teste (~9,4%).
- A taxa de ruído do corpus (textos com ≤15 caracteres, ex.: "...", "ruim")
  é de **8,7%** — ou seja, **quase todo o erro do baseline é explicado por ruído
  irredutível**, não por falha do modelo. Esse é o teto prático de melhoria.
- Confusões mais comuns (verdadeiro → previsto): Telecom → Comércio Eletrônico (7),
  Telecom → Saúde-Planos (5), Bancos → Comércio Eletrônico (4).

## Falhas categorizadas

| # | Exemplo (texto) | Verdadeiro | Previsto | Categoria do erro | Hipótese de causa | Ação proposta |
|---|---|---|---|---|---|---|
| 1 | `"..."` | Telecomunicações | Comércio Eletrônico | Ruído (texto vazio de conteúdo) | Sem sinal textual; modelo "chuta" a classe mais provável | Filtrar/encaminhar textos <15 chars para triagem humana; não pontuar no F1 |
| 2 | `"ruim"` | Saneamento | Telecomunicações | Ruído (1 palavra genérica) | Palavra sem termo de domínio | Exigir mínimo de tokens informativos; pedir complemento ao usuário |
| 3 | `"péssimo"` / `"horrível"` | Comércio Elet. / Bancos | Telecomunicações | Ruído (só sentimento) | Termo só carrega polaridade, não setor | Combinar com metadados (empresa) ou classificar como "indeterminado" |
| 4 | `"😤😩"` / `"fraude! 🤬"` | Bancos e Cartões | Comércio Eletrônico | Ruído (emoji/quase sem texto) | TF-IDF ignora emoji; resta token fraco | Normalizar emoji→texto no pré-processamento; rota de baixa confiança |
| 5 | `"<p>Pedi cancelamento e nada foi feito.</p><br/>..."` | Imobiliário | Saúde-Planos | Ruído (HTML não limpo) | Tags HTML viram tokens e diluem o sinal | Remover HTML no pré-processamento (BeautifulSoup/regex) |
| 6 | `"Worst customer service ever. I want a r..."` | Telecomunicações | Energia Elétrica | Ambiguidade (idioma) | Texto em inglês; stopwords/stemmer PT não tratam | Detectar idioma; usar embedding multilíngue ou rota LLM |
| 7 | `"Solicitei o estorno do valor R$ 8.498,50..."` | Telecomunicações | Saúde-Planos | Ambiguidade do dado | Texto fala de estorno (transversal), sem citar o setor | Usar `empresa_mencionada` extraída pelo LLM como feature |
| 8 | `"Pedi cancelamento e nada foi feito. Reembolso pendente"` | Imobiliário | Saúde-Planos | Erro de classificação | Vocabulário genérico (cancelamento/reembolso) comum a vários setores | Few-shot/LLM com pistas de contexto; feature `tipo_problema` |

## Erros do LLM (observados em 32 chamadas reais, `gemini-2.5-flash-lite`)
Na amostra processada, o LLM acertou as 32 categorias (0 erro de classificação).
Mas a inspeção da extração estruturada revelou um padrão importante:

| # | Caso real | Categoria do erro | Hipótese de causa | Ação proposta |
|---|---|---|---|---|
| 9 | **id=169** — texto = `"..!"` (puro ruído). O LLM retornou, com **confiança 0,9**: empresa=`"Claro"`, tipo=`cobranca_indevida`, resumo=`"cliente relata cobrança indevida em fatura de celular referente a serviço não solicitado"`. **Nada disso está no texto.** | **Alucinação do LLM** | Diante de entrada vazia/ruído, o modelo preenche o schema com um caso "plausível" em vez de admitir ausência de informação; a `confianca` não reflete isso | (a) Validar `empresa_mencionada` contra o texto (feito — ver abaixo); (b) pré-filtrar textos <15 chars antes de chamar o LLM; (c) permitir campos nulos e instruir "se não houver informação, retorne null" |
| 10 | Validação de `empresa_mencionada`: das 27 empresas extraídas, **26 realmente aparecem no texto** (Kabum, Amazon, Vivo, Enel, Itaú...) e **1 não** (a do caso #9). Taxa de alucinação ≈ 3,7% na extração de entidade. | (medição) | — | Manter o *check* automático empresa-no-texto como guarda de qualidade |

**Contraste instrutivo:** nos ~9% de textos-ruído, o **baseline erra** (não tem
sinal) e o **LLM alucina** (fabrica sinal). Falham no mesmo ponto, de formas
opostas — reforçando que a ação de maior impacto é **higienizar a entrada** e ter
uma rota separada para textos sem conteúdo, antes de qualquer modelo.

> Reprodução: `python -m src.classify_llm` (preenche o cache) — o *check* de
> alucinação está no script de avaliação. Itens com `ok=false` (fallback por falha
> de validação/cota) ficam registrados na coluna `erro` do cache JSONL.

## Conclusão crítica
O baseline já captura quase todo o sinal disponível; o erro remanescente é
dominado por **ruído de entrada** (textos vazios/curtos, emoji, HTML, idioma).
Portanto, o maior ganho não está em um classificador mais sofisticado, mas em
**higienização de entrada** e em uma **rota de baixa confiança** (triagem humana
ou LLM) para os ~9% de casos ambíguos — exatamente onde o LLM tende a agregar.
