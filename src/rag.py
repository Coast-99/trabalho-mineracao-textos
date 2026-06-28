"""RAG sobre o corpus de reclamações (componente BÔNUS — técnica da Aula 7).

Pipeline: indexa as reclamações com embeddings (llama-index +
HuggingFaceEmbedding), recupera os trechos mais relevantes para uma pergunta
de gestor e (se houver GEMINI_API_KEY) gera uma resposta ancorada.

Avaliação de UMA métrica de recuperação (exigência do bônus):
- relevância@k: para perguntas cuja resposta pertence a um setor conhecido,
  medimos a fração dos k documentos recuperados cuja `categoria` confere.
  É um proxy objetivo; a tabela em relatorio/ também permite anotação manual.

A indexação usa o MESMO modelo de embedding do resto do projeto e é persistida
em ./storage para não reprocessar.
"""
from __future__ import annotations

from . import config, data

TOP_K = 5

# Perguntas de gestor com o setor esperado (para a métrica de relevância@k).
PERGUNTAS_AVALIACAO = [
    ("Quais os principais problemas de internet e telefonia?", "Telecomunicações"),
    ("Há reclamações sobre cobrança indevida no cartão?", "Bancos e Cartões"),
    ("O que os clientes relatam sobre entregas de compras online?", "Comércio Eletrônico"),
    ("Quais queixas existem sobre conta de luz?", "Energia Elétrica"),
    ("Há problemas com atraso ou cancelamento de voos?", "Transporte Aéreo"),
]


def _configurar_embeddings():
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
    Settings.llm = None  # recuperação não precisa de LLM


def construir_indice(limite: int | None = None):
    """Cria (ou carrega do disco) o índice vetorial do corpus."""
    from llama_index.core import (
        Document,
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )

    _configurar_embeddings()
    storage = config.BASE_DIR / "storage"

    if storage.exists():
        ctx = StorageContext.from_defaults(persist_dir=str(storage))
        return load_index_from_storage(ctx)

    df = data.carregar()
    if limite:
        df = df.head(limite)
    docs = [
        Document(
            text=r.texto,
            metadata={"id": int(r.id), "categoria": r.categoria, "estado": r.estado},
        )
        for r in df.itertuples()
    ]
    print(f"Indexando {len(docs)} reclamações...")
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=str(storage))
    return index


def recuperar(index, pergunta: str, k: int = TOP_K):
    retriever = index.as_retriever(similarity_top_k=k)
    nos = retriever.retrieve(pergunta)
    return [
        {
            "score": n.score,
            "categoria": n.metadata.get("categoria"),
            "texto": n.node.text,
        }
        for n in nos
    ]


def avaliar_relevancia(index, k: int = TOP_K):
    """relevância@k por pergunta + média geral."""
    print(f"\n=== Avaliação de relevância@{k} ===")
    linhas = ["| Pergunta | Setor esperado | relevância@k |", "|---|---|---|"]
    total = 0.0
    for pergunta, setor in PERGUNTAS_AVALIACAO:
        docs = recuperar(index, pergunta, k)
        acertos = sum(1 for d in docs if d["categoria"] == setor)
        rel = acertos / k
        total += rel
        linhas.append(f"| {pergunta} | {setor} | {rel:.2f} |")
    media = total / len(PERGUNTAS_AVALIACAO)
    linhas.append(f"| **Média** | | **{media:.2f}** |")
    tabela = "\n".join(linhas)
    print(tabela)
    return tabela, media


def responder(index, pergunta: str, k: int = TOP_K) -> str:
    """Geração ancorada (requer GEMINI_API_KEY)."""
    docs = recuperar(index, pergunta, k)
    contexto = "\n\n".join(f"- ({d['categoria']}) {d['texto']}" for d in docs)
    prompt = (
        "Você é um analista de ouvidoria. Responda à pergunta do gestor usando "
        "APENAS as reclamações recuperadas abaixo. Se não houver base, diga que "
        "não há evidências.\n\n"
        f"Pergunta: {pergunta}\n\nReclamações recuperadas:\n{contexto}\n\nResposta:"
    )
    from google.genai import types

    from . import llm_client

    resp = llm_client.get_client().models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return resp.text


def main():
    index = construir_indice()
    tabela, media = avaliar_relevancia(index)
    saida = config.RELATORIO_DIR / "rag_relevancia.md"
    saida.write_text(
        f"# RAG — Avaliação de relevância@{TOP_K}\n\n{tabela}\n", encoding="utf-8"
    )
    print(f"\nSalvo em {saida}")

    # Demonstração de recuperação (sem precisar de chave):
    exemplo = PERGUNTAS_AVALIACAO[0][0]
    print(f"\nExemplo de recuperação para: {exemplo!r}")
    for d in recuperar(index, exemplo, k=3):
        print(f"  [{d['categoria']}] ({d['score']:.3f}) {d['texto'][:90]}")


if __name__ == "__main__":
    main()
