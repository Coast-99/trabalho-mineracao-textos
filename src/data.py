"""Carregamento, limpeza e split estratificado do corpus de reclamações.

Decisões de dados (ver DECISOES.md):
- O CSV está em `utf-8` (confirmado nos bytes: é = C3 A9); lê-lo como latin-1
  corrompe os caracteres multibyte (ç, õ, ã viram "a"). Usamos errors="replace"
  para os pouquíssimos bytes estranhos.
- A categoria "Saúde — Planos" tem um traço unicode; canonicalizamos por slug
  então canonicalizamos toda categoria por um "slug" (sem acento/pontuação) e
  remapeamos para o nome limpo de `config.CATEGORIAS`.
- Removemos reclamações com `texto` vazio (≈65 linhas) — não há o que classificar.
- Split estratificado por categoria, com SEED fixo, reutilizado por TODAS as
  abordagens (baseline, LLM) para comparação justa.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def _slug(texto: str) -> str:
    """Normaliza para comparação: minúsculas, sem acento, só letras/números."""
    nfkd = unicodedata.normalize("NFKD", str(texto))
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", sem_acento.lower())


# slug -> nome canônico limpo
_CANON = {_slug(c): c for c in config.CATEGORIAS}


def _canonicaliza_categoria(valor: str) -> str | None:
    """Mapeia uma categoria possivelmente "suja" para o nome canônico."""
    return _CANON.get(_slug(valor))


def carregar() -> pd.DataFrame:
    """Carrega o CSV bruto, limpa e devolve um DataFrame pronto para uso."""
    df = pd.read_csv(
        config.DATA_RAW, encoding=config.CSV_ENCODING, encoding_errors="replace"
    )

    # normaliza espaços do texto
    df["texto"] = (
        df["texto"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    # remove textos vazios
    df = df[df["texto"].str.len() > 0].copy()

    # canonicaliza a categoria (corrige o traço bagunçado de "Saúde - Planos")
    df["categoria"] = df["categoria"].map(_canonicaliza_categoria)
    df = df[df["categoria"].notna()].reset_index(drop=True)

    return df


def split(df: pd.DataFrame):
    """Split estratificado treino/teste. Retorna (df_train, df_test)."""
    df_train, df_test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=df["categoria"],
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


if __name__ == "__main__":
    df = carregar()
    print(f"Total de reclamações válidas: {len(df)}")
    print("\nDistribuição de categorias:")
    print(df["categoria"].value_counts())
    tr, te = split(df)
    print(f"\nTreino: {len(tr)} | Teste: {len(te)}")
