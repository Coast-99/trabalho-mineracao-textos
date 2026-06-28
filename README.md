# Mineração de Textos — Classificação de Reclamações de Consumidor

## Integrantes

- Letícia Araújo Costa
- Samara Pereira Quintanilha
- Ana Clara Lomeu Rosa
- Isabel Azar de Holanda

Trabalho final da disciplina de Mineração de Textos (Caminho B — classificação
supervisionada). Pipeline que vai do dado bruto a insights acionáveis para um
gestor de ouvidoria, integrando o **stack clássico (embeddings + ML)** com o
**moderno (LLM com saída estruturada validada por Pydantic)**.

## Problema e relevância
Uma central de atendimento recebe milhares de reclamações em texto livre. Roteá-las
ao **setor correto** (Telecomunicações, Bancos, Energia, etc.) é repetitivo e caro.
Tratamos isso como **classificação supervisionada em 10 categorias** e medimos se um
LLM agrega valor sobre um baseline clássico — e a que custo.

- **Corpus:** 2.935 reclamações rotuladas (`data/raw/reclamacoes_consumidor.csv`).
- **Alvo:** `categoria` (10 classes). **Métrica:** F1-macro (classes desbalanceadas).

## Arquitetura do pipeline
```
dados (utf-8) ─► pré-proc (NLTK, stemming)          ─► TF-IDF ─┐
              └► texto cru ─► embeddings (MiniLM ML) ───────────┼─► LogReg ─► F1 (baseline)
                                                                │
              texto cru ─► LLM (Gemini + Pydantic) ─► {categoria, tipo_problema,
                                                       sentimento, urgência, ...}
                              │                          │
                              ├─ (L1) classificação zero/few-shot ─► F1
                              └─ (L2) features ⊕ embeddings ─► LogReg ─► F1
                                                                │
              corpus ─► índice vetorial (llama-index) ─► RAG (relevância@k)  [bônus]
```

## Resultados (F1-macro)
**Baseline completo (n=587, antes do LLM):**
| Abordagem | F1-macro | Custo/1000 |
|---|---|---|
| TF-IDF + LogReg | **0.9167** | R$ 0 |
| Embeddings + LogReg | 0.8752 | R$ 0 |

**Comparação com o LLM (amostra real n=32, mesmos itens):**
| Abordagem | F1-macro | Custo/1000 |
|---|---|---|
| TF-IDF + LogReg | 0.9821 | R$ 0 |
| LLM zero-shot (`gemini-2.5-flash-lite`) | **1.0000** | ~R$ 2,1 |

- **RAG (bônus):** relevância@5 = **1.00** nas 5 perguntas de gestor.
- O baseline já acerta 92%; ~9% de erro restante é **ruído de entrada** (textos
  vazios, emoji, HTML). Nesses mesmos casos o LLM **alucina** (ver caso id=169 em
  `relatorio/analise_erros.md`) — falham no mesmo ponto, de formas opostas.
- A avaliação do LLM ficou em n=32 por **limite de cota do free tier**; o pipeline
  roda no conjunto completo sem mudar código (ver nota em `relatorio/analise_resultados.md`).

## Estrutura
```
src/            código (data, preprocess, embeddings, baseline, schemas,
                llm_client, classify_llm, evaluate, rag, custos)
prompts/        prompts versionados (zero-shot, few-shot)
data/raw/       CSV original    data/processed/  embeddings, caches, predições
relatorio/      análise de resultados, erros, insights, custos
notebooks/      01_demo.ipynb (demo limpa da Aula 8)
DECISOES.md     diário de decisões
```

## Como rodar
```bash
# 1. Dependências
pip install -r requirements.txt

# 2. Chave do LLM (Google AI Studio — free tier serve)
cp .env.example .env        # edite e cole sua GEMINI_API_KEY

# 3. Baseline clássico (não precisa de chave)
python -m src.baseline

# 4. LLM: classificação + enriquecimento (use um limite p/ testar barato)
python -m src.classify_llm 50      # 50 itens; sem argumento = conjunto todo

# 5. Tabela comparativa
python -m src.evaluate

# 6. RAG bônus (recuperação não precisa de chave)
python -m src.rag

# 7. Custos previstos (Gemini)
python -m src.custos
```
> O cache em JSONL (`data/processed/cache_llm_*.jsonl`) evita re-chamar a API ao
> reexecutar. Para a demo da Aula 8, abra `notebooks/01_demo.ipynb`.

## Componentes do enunciado — onde estão
- **Embeddings:** `src/embeddings.py` · **ML clássico:** `src/baseline.py`
- **LLM + Pydantic (núcleo):** `src/schemas.py`, `src/llm_client.py`, `src/classify_llm.py`
- **Baseline antes do LLM:** `src/baseline.py` (F1 registrado)
- **RAG (bônus):** `src/rag.py` · **Custos:** `src/custos.py` + `relatorio/custos_llm.md`
- **Análise de erro / insights / decisões:** `relatorio/` + `DECISOES.md`
