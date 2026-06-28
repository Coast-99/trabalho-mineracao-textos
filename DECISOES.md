# Diário de decisões (DECISOES.md)

Registro curto de cada escolha de implementação e o porquê. Atualizar conforme o
projeto evolui (especialmente após observar erros).

## Dados e pré-processamento
- **Corpus:** `reclamacoes_consumidor.csv` (2.935 reclamações válidas após limpeza).
  Limpo, rotulado e do tamanho certo para iterar rápido com baixo custo de LLM.
- **Encoding = UTF-8** (não latin-1). Decidido após inspecionar os *bytes* do
  arquivo (`é` = `C3 A9`). Ler como latin-1 corrompe ç/õ/ã. Usamos
  `encoding_errors="replace"` para os pouquíssimos bytes inválidos.
- **Categoria canonicalizada por slug** (sem acento/pontuação): a classe
  "Saúde — Planos" tinha um traço unicode inconsistente; o slug evita perder linhas.
- **Texto vazio removido** (~65 linhas): não há o que classificar.
- **Split estratificado, SEED=42, test_size=0.20**, reutilizado por todas as
  abordagens — comparação justa entre baseline e LLM.
- **Duas versões de texto:** cru (para embeddings e LLM) e limpo (para TF-IDF:
  minúsculas, sem acento, stopwords PT do NLTK, `RSLPStemmer`).

## Representação
- **Embedding = `paraphrase-multilingual-MiniLM-L12-v2`**: multilíngue (bom p/ PT),
  leve, rápido e já usado em aula. Vetores cacheados em `.npy`.
- **Similaridade:** cosseno (padrão do sentence-transformers e do índice do RAG).
- **TF-IDF** com n-gramas (1,2), `min_df=2`, `max_features=20000`.

## Classificação (Caminho B)
- **Alvo = `categoria`** (10 classes) → "rotear a reclamação para o setor certo".
- **Métrica = F1-macro** (classes desbalanceadas; cada classe pesa igual) +
  `class_weight="balanced"` na Regressão Logística.
- **Baseline antes do LLM:** TF-IDF+LogReg = **0.9167**; Embeddings+LogReg = **0.8752**.
  TF-IDF vence porque o texto é curto e cheio de palavras-chave de domínio.

## LLM com saída estruturada (núcleo)
- **Schema Pydantic `AnaliseReclamacao`** com `Literal` para vocabulário fechado
  (`categoria_prevista`, `tipo_problema`, `sentimento`, `urgencia`). Campos extras
  (`empresa_mencionada`, `resumo`, `confianca`) servem ao enriquecimento e à
  inspeção. `confianca` (0–1) ajuda a achar casos duvidosos.
- **Modelo = `gemini-3.1-flash-lite`** via `google-genai`, `temperature=0`,
  `response_mime_type="application/json"`, `response_schema=AnaliseReclamacao`.
- **Tratamento de falha de validação:** `try/except` (ValidationError + erros de
  API) com 1 retry e *backoff*; se persistir, devolve `FALLBACK` marcado com
  `ok=false` — o pipeline nunca quebra no meio do lote.
- **Cache JSONL por `id`:** reexecutar não re-chama a API (economiza cota/custo).
- **Dois usos do LLM:** (L1) classificação direta zero-shot/few-shot;
  (L2) enriquecimento — `tipo_problema/sentimento/urgencia` viram features one-hot
  concatenadas aos embeddings.

## RAG (bônus)
- **`llama-index` + HuggingFaceEmbedding** (mesmo modelo de embedding); índice
  persistido em `./storage`.
- **Métrica = relevância@5** (proxy: categoria do trecho recuperado bate com o
  setor esperado da pergunta). Resultado: **1.00** nas 5 perguntas de gestor.

## Custos
- Estimados para os 3 modelos Gemini exigidos (ver `custos_llm.md`); ~402 tokens
  de entrada (few-shot) + ~110 de saída por reclamação.

## O que mudou após observar erros
- **Alucinação confirmada:** ao rodar o LLM, o item id=169 (texto = `"..!"`)
  recebeu uma reclamação totalmente fabricada (empresa "Claro", cobrança indevida)
  com confiança 0,9. Decisão: (1) adotar um *check* automático de
  `empresa_mencionada` contra o texto como guarda de qualidade; (2) pré-filtrar
  textos com <15 caracteres antes de chamar o LLM (rota de ruído).
- **Limite de cota (free tier):** o processamento em lote esbarrou em `429
  RESOURCE_EXHAUSTED`. Decisões de engenharia tomadas no cliente: honrar o
  `retryDelay` do erro, rate limiter global, cache que só persiste sucessos
  (reexecução retoma de onde parou) e fallback marcado com `ok=false`. Conclusão
  para o gestor: classificar 1.000+ itens exige tier pago (ver `custos_llm.md`).
- A partir do baseline: **higienizar entrada** (remover HTML, tratar emoji,
  detectar idioma) — ~9% dos erros vêm de ruído, não do modelo.

## Modelo de LLM
- Começamos com `gemini-3.1-flash-lite` (do notebook da aula); trocamos para
  `gemini-2.5-flash-lite` porque a cota diária do free tier é **por modelo** e a do
  3.1 esgotou. O schema Pydantic é idêntico — só muda a string do modelo.
