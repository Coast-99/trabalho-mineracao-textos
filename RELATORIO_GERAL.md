# Relatório Geral — Mineração de Textos sobre Reclamações de Consumidor

> Trabalho final · Caminho B (classificação supervisionada) · Especialização em
> Data Science e IA — Senac DF. Este documento conta **o que foi feito, por quê,
> as dificuldades enfrentadas e os resultados obtidos**. Para rodar o projeto, ver
> o `README.md`; para a justificativa item a item, o `DECISOES.md`.

---

## 1. O problema e por que ele importa

Uma central de atendimento/ouvidoria recebe milhares de reclamações em **texto
livre**. Encaminhar cada uma ao **setor correto** (Telecomunicações, Bancos,
Energia, etc.) é um trabalho repetitivo, lento e caro quando feito manualmente.

Tratamos isso como um problema de **classificação supervisionada em 10 categorias**
e respondemos a uma pergunta de negócio concreta: *vale a pena usar um LLM para
isso, ou um modelo clássico já resolve — e a que custo?*

O objetivo central do trabalho (e o mais avaliado) não é só "acertar a categoria",
e sim **usar o LLM como um componente de software confiável**: com saída
estruturada, validada por um contrato (Pydantic), tratamento de falhas e custo
mensurável — e não como um chatbot de texto livre.

---

## 2. O corpus

- **Fonte:** `reclamacoes_consumidor.csv` — reclamações de consumidores rotuladas.
- **Tamanho:** 3.000 linhas → **2.935 válidas** após limpeza (removidos textos vazios).
- **Colunas:** `id, data_reclamacao, estado, regiao, categoria, status, texto`.
- **Alvo (`categoria`), 10 classes — bastante desbalanceadas:**

| Categoria | Qtd | | Categoria | Qtd |
|---|---|---|---|---|
| Telecomunicações | 800 | | Transporte Aéreo | 152 |
| Bancos e Cartões | 570 | | Educação | 118 |
| Comércio Eletrônico | 522 | | Seguros | 92 |
| Energia Elétrica | 314 | | Imobiliário | 90 |
| Saúde - Planos | 219 | | Saneamento | 58 |

- **Texto curto** (média ~111 caracteres) e, em parte, **ruidoso** (ex.: "...",
  "péssimo", emojis) — o que se mostrou decisivo nos resultados.

Por causa do desbalanceamento, adotamos **F1-macro** (cada classe pesa igual) e
**split estratificado** fixo (`random_state=42`, 80/20), reutilizado por todas as
abordagens para uma comparação justa.

---

## 3. O que foi feito (arquitetura do pipeline) e por quê

```
CSV (utf-8) ─► pré-processamento (NLTK, stemming) ─► TF-IDF ───────────┐
            └► texto cru ─► embeddings (MiniLM multilíngue) ───────────┼─► LogReg ─► F1  (BASELINE)
                                                                       │
            texto cru ─► LLM (Gemini + schema Pydantic) ─► {categoria, tipo_problema,
                            │                                sentimento, urgência, empresa, ...}
                            ├─ (L1) classificação zero-shot ─────────────► F1
                            └─ (L2) campos extraídos viram features ─► LogReg ─► F1
                                                                       │
            corpus ─► índice vetorial (llama-index) ─► RAG (relevância@k)   [BÔNUS]
```

| Etapa | O que foi feito | Por quê |
|---|---|---|
| **Pré-processamento** | minúsculas, remoção de acentos/pontuação, *stopwords* PT (NLTK), `RSLPStemmer` | Reduz vocabulário e ruído para o TF-IDF (técnica da Aula 2) |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) | Representação semântica multilíngue, leve e rápida (Aula 3) |
| **Baseline clássico** | TF-IDF e embeddings + Regressão Logística (`class_weight="balanced"`) | É a régua **medida antes do LLM** — para o ganho do LLM ser mensurável |
| **LLM + Pydantic (núcleo)** | `gemini-2.5-flash-lite`, `response_schema=AnaliseReclamacao`, validação + retry + cache | Saída estruturada confiável; o schema é o "contrato" do sistema |
| **Enriquecimento** | campos extraídos (sentimento, urgência, tipo) viram *features* extras | Testar se o LLM ajuda o classificador clássico |
| **RAG (bônus)** | índice com llama-index + busca semântica + métrica de relevância | Recuperar trechos relevantes para perguntas de gestor (Aula 7) |
| **Custos** | estimativa em 3 modelos Gemini para 1.000 itens | Responder "a que custo?" com número |

### O contrato Pydantic (coração do trabalho)
O schema `AnaliseReclamacao` força o LLM a devolver **sempre** o mesmo formato, com
**vocabulário controlado** (`Literal`) onde faz sentido:

| Campo | Tipo | Para quê |
|---|---|---|
| `categoria_prevista` | Literal[10 categorias] | classificação |
| `tipo_problema` | Literal[8 tipos] | feature de enriquecimento |
| `sentimento` | Literal[4 níveis] | priorização / feature |
| `urgencia` | Literal[3 níveis] | priorização / feature |
| `empresa_mencionada` | str \| None | entidade extraída |
| `resumo` | str (máx. 300) | inspeção humana |
| `confianca` | float 0–1 | achar casos duvidosos |

Se o LLM devolver algo fora do contrato, o cliente captura o erro de validação,
**re-tenta** e, em último caso, usa um **fallback** marcado (`ok=false`) — o lote
nunca quebra no meio.

---

## 4. Dificuldades enfrentadas (e como resolvemos)

Esta seção é honesta sobre os percalços reais — vários renderam decisões de
engenharia que melhoraram o projeto.

### 4.1 Encoding traiçoeiro do CSV
A primeira leitura "parecia" funcionar como `latin-1`, mas corrompia silenciosamente
os acentos (`ç`, `õ`, `ã` viravam `a`), o que **quebrava o agrupamento por
categoria** (só 3 das 10 sobreviviam). **Diagnóstico:** inspecionamos os *bytes* do
arquivo e vimos `é = C3 A9` → o arquivo é **UTF-8**. **Solução:** ler como UTF-8 com
`errors="replace"`. Além disso, a categoria "Saúde — Planos" tinha um traço unicode
inconsistente; resolvemos canonicalizando as categorias por um *slug* (sem
acento/pontuação) em vez de depender da string exata.

### 4.2 O limite do free tier do Gemini (a maior dificuldade)
Foi o gargalo principal e virou um **achado de engenharia** relevante:

- Ao processar em lote, a API começou a recusar com **`429 RESOURCE_EXHAUSTED`**.
- Tentar **paralelizar** para acelerar **piorou**: estourava o limite por minuto e
  ~50% das chamadas caíam no *fallback* (estragando as métricas).
- O erro vinha com um campo **`retryDelay`** (ex.: "espere 43s") — sinal de que é um
  limite **por minuto** (recuperável), não só diário.
- A primeira chave (formato `AQ.A...`) tinha cota especialmente restrita; trocamos
  por uma chave padrão (`AIza...`), que ajudou, mas o teto diário ainda limitou o
  volume processável no prazo.

**O que aprendemos e implementamos no cliente (`src/llm_client.py`):**
1. **Honrar o `retryDelay`** que o servidor sugere, em vez de um backoff cego;
2. **Rate limiter global** para marcar o passo logo abaixo do limite;
3. **Cache JSONL que só guarda sucessos** → reexecutar **retoma de onde parou** e
   re-tenta os que falharam;
4. **Fallback marcado** (`ok=false`) para nunca quebrar o lote.

**Consequência prática:** a avaliação do LLM ficou em **n=32** chamadas reais
bem-sucedidas (sem fallback). O pipeline roda no conjunto completo sem mudar código,
bastando cota adequada (tier pago) — e isso conecta diretamente com a seção de
custos: **o free tier não escala para 1.000+ itens.**

### 4.3 Texto curto e ruidoso
~9% dos textos têm ≤15 caracteres ("...", "ruim", emojis). Isso degrada qualquer
modelo e exigiu uma decisão de produto: **higienizar a entrada** e ter uma **rota de
baixa confiança** (triagem humana/LLM) antes de classificar.

---

## 5. Resultados (com exemplos reais)

### 5.1 Baseline completo (n=587, antes do LLM)
| Abordagem | F1-macro |
|---|---|
| **TF-IDF + Regressão Logística** | **0.9167** |
| Embeddings + Regressão Logística | 0.8752 |

O TF-IDF vence porque o texto é curto e cheio de **palavras-chave de domínio**
("internet", "fatura", "voo") que casam direto com o setor.

### 5.2 Comparação com o LLM (mesma amostra real, n=32)
| Abordagem | F1-macro |
|---|---|
| Embeddings + LogReg | 0.8155 |
| TF-IDF + LogReg | 0.9821 |
| **LLM zero-shot (`gemini-2.5-flash-lite`)** | **1.0000** |

Na amostra avaliada, o LLM acertou **todas** as categorias **sem usar dado de treino**
— ligeiramente acima do TF-IDF. Mas como o baseline já é muito forte (0,92 no
conjunto inteiro), **o ganho de acurácia não justifica o custo** para classificação
pura. O valor real do LLM aparece na **extração estruturada**, abaixo.

### 5.3 Exemplo de saída estruturada (o que o LLM agrega)
Reclamação: *"Minha internet da Vivo cai toda hora e ninguém resolve, já abri 3
protocolos."* →
```json
{
  "categoria_prevista": "Telecomunicações",
  "tipo_problema": "qualidade_servico",
  "sentimento": "negativo",
  "urgencia": "media",
  "empresa_mencionada": "Vivo",
  "resumo": "Cliente relata instabilidade recorrente da internet sem resolução após múltiplos protocolos.",
  "confianca": 1.0
}
```
O classificador clássico só devolveria "Telecomunicações"; o LLM entrega **empresa,
tipo, sentimento e urgência** — insumos diretos para priorização e insights.

### 5.4 Exemplo de FALHA real — alucinação do LLM (caso id=169)
A reclamação id=169 tem como texto apenas **`"..!"`** (puro ruído). Mesmo assim, o
LLM retornou **com confiança 0,9**:
```json
{
  "empresa_mencionada": "Claro",
  "tipo_problema": "cobranca_indevida",
  "resumo": "Cliente relata cobrança indevida em fatura de celular referente a serviço não solicitado",
  "confianca": 0.9
}
```
**Nada disso está no texto** — o modelo *fabricou* uma reclamação plausível.
Contraste instrutivo: nos ~9% de textos-ruído, **o baseline erra** (falta sinal) e
**o LLM alucina** (inventa sinal). Falham no mesmo ponto, de formas opostas.

**Salvaguarda criada:** um *check* automático compara `empresa_mencionada` com o
texto. Resultado: das 27 empresas extraídas, **26 realmente aparecem** e **1 não**
(o caso acima) → taxa de alucinação ≈ **3,7%** na extração de entidade.

### 5.5 RAG (bônus) — relevância@5
Indexamos as 2.935 reclamações e medimos a relevância dos trechos recuperados para
5 perguntas de gestor:

| Pergunta (resumo) | Setor esperado | relevância@5 |
|---|---|---|
| problemas de internet/telefonia | Telecomunicações | 1.00 |
| cobrança indevida no cartão | Bancos e Cartões | 1.00 |
| entregas de compras online | Comércio Eletrônico | 1.00 |
| conta de luz | Energia Elétrica | 1.00 |
| atraso/cancelamento de voos | Transporte Aéreo | 1.00 |
| **Média** | | **1.00** |

### 5.6 Custos previstos com o LLM (1.000 reclamações)
Preços oficiais (jun/2026), prompt few-shot (~402 tokens entrada + ~110 saída):

| Modelo | Custo / 1.000 itens |
|---|---|
| gemini-2.5-flash | ~R$ 2,14 |
| gemini-3-flash-preview | ~R$ 2,87 |
| gemini-3.1-pro-preview | ~R$ 11,47 |

---

## 6. Três insights acionáveis para o gestor
1. **Gargalos de resolução por setor:** Saneamento (50% não resolvidas), Educação
   (47,5%) e Transporte Aéreo (46,1%) são os piores — priorizar auditoria de SLA aí.
2. **Volume × qualidade:** 42% das reclamações vêm do Sudeste, mas o problema de
   resolução **não** acompanha o volume → metas de resolução devem ser **por setor**.
3. **Ruído trava a automação:** ~9% das manifestações são vazias/curtas e concentram
   o erro automático → validar conteúdo mínimo na entrada e criar uma rota de baixa
   confiança (humano/LLM), aplicando o LLM só na fatia ambígua (~R$ 2–11/1000).

---

## 7. Conclusão
O baseline clássico já entrega 92% de F1-macro a custo zero; o LLM iguala/supera de
leve **sem dados de treino**, mas seu valor decisivo é a **extração estruturada
confiável** (via contrato Pydantic), não a classificação pura. As maiores
dificuldades — encoding e, sobretudo, os limites do free tier — viraram decisões de
engenharia (leitura correta dos dados, cliente LLM robusto a `429`, cache
retomável). E o achado mais útil para a operação é que **dinheiro e modelo não
resolvem ruído de entrada**: a higienização dos dados e uma rota de triagem são o
maior alavancador de qualidade.
