"""Pré-processamento de texto em português (técnicas da Aula 2).

Fornece uma limpeza leve usada pelo baseline TF-IDF. NÃO aplicamos esta
limpeza ao texto enviado para embeddings nem para o LLM — esses modelos
trabalham melhor com o texto natural (com acento, pontuação e caixa).

Pipeline (configurável):
  minúsculas -> remove acentos -> remove pontuação/dígitos -> tokeniza
  -> remove stopwords PT -> (opcional) RSLPStemmer
"""
from __future__ import annotations

import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Baixa recursos do NLTK na primeira execução (idempotente).
for _pkg in ("stopwords", "rslp"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOPWORDS = set(stopwords.words("portuguese"))
_STEMMER = RSLPStemmer()
_TOKEN_RE = re.compile(r"[a-zà-ÿ]+", re.IGNORECASE)


def remove_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def limpar(texto: str, *, stem: bool = True, remover_acentos: bool = True) -> str:
    """Limpa um texto e devolve uma string de tokens separados por espaço."""
    texto = str(texto).lower()
    if remover_acentos:
        texto = remove_acentos(texto)
    tokens = _TOKEN_RE.findall(texto)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def limpar_serie(textos, **kwargs):
    """Aplica `limpar` a um iterável de textos."""
    return [limpar(t, **kwargs) for t in textos]


if __name__ == "__main__":
    from . import data

    df = data.carregar()
    exemplos = df["texto"].head(3).tolist()
    for t in exemplos:
        print("ORIGINAL:", t[:120])
        print("LIMPO   :", limpar(t))
        print("-" * 60)
