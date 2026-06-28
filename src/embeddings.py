"""Embeddings de sentenças com sentence-transformers (técnica da Aula 3).

Usamos `paraphrase-multilingual-MiniLM-L12-v2`: multilíngue (bom para PT),
leve e rápido. Os vetores são cacheados em disco (.npy) para não reprocessar.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import config

_model = None


def _get_model():
    """Carrega o modelo uma única vez (lazy)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _cache_path(textos) -> "config.Path":
    """Caminho de cache derivado de um hash do conteúdo + nome do modelo."""
    h = hashlib.md5()
    h.update(config.EMBEDDING_MODEL.encode())
    for t in textos:
        h.update(str(t).encode("utf-8", "ignore"))
    return config.DATA_PROC / f"emb_{h.hexdigest()[:12]}.npy"


def encode(textos, *, usar_cache: bool = True) -> np.ndarray:
    """Gera (ou recupera do cache) os embeddings de uma lista de textos."""
    textos = list(textos)
    cache = _cache_path(textos)
    if usar_cache and cache.exists():
        return np.load(cache)

    modelo = _get_model()
    vetores = modelo.encode(
        textos, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    if usar_cache:
        np.save(cache, vetores)
    return vetores


if __name__ == "__main__":
    from . import data

    df = data.carregar()
    amostra = df["texto"].head(50).tolist()
    vec = encode(amostra)
    print("Shape dos embeddings:", vec.shape)
