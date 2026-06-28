"""Configurações centrais do projeto: caminhos, constantes e rótulos.

Tudo que é "decisão de projeto" (modelo de embedding, modelo do LLM, seed,
proporção de teste, vocabulário de categorias) mora aqui para ficar fácil de
auditar e justificar no DECISOES.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Console do Windows costuma usar cp1252 e quebra ao imprimir acentos.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- Caminhos -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw" / "reclamacoes_consumidor.csv"
DATA_PROC = BASE_DIR / "data" / "processed"
PROMPTS_DIR = BASE_DIR / "prompts"
RELATORIO_DIR = BASE_DIR / "relatorio"

DATA_PROC.mkdir(parents=True, exist_ok=True)

# Carrega GEMINI_API_KEY do .env, se existir.
load_dotenv(BASE_DIR / ".env")

# --- Reprodutibilidade ----------------------------------------------------
SEED = 42
TEST_SIZE = 0.20

# --- Encoding do CSV (confirmado na investigação dos bytes: é=C3 A9) ------
# O arquivo é UTF-8; alguns poucos bytes estranhos são tratados com replace.
CSV_ENCODING = "utf-8"

# --- Modelos --------------------------------------------------------------
# Embedding multilíngue usado em aula (rápido e bom para PT).
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# LLM de execução (mesma stack do notebook da aula). Pode trocar por outro
# modelo Gemini disponível na sua conta. Usamos o 2.5-flash-lite porque a cota
# diária do free tier é por modelo (a do 3.1-flash-lite esgotou nos testes).
LLM_MODEL = "gemini-2.5-flash-lite"

# --- Vocabulário controlado de categorias (alvo da classificação) ---------
# É exatamente o conjunto presente no CSV. Serve tanto para validar os dados
# quanto como Literal do schema Pydantic (vocabulário fechado para o LLM).
CATEGORIAS = (
    "Telecomunicações",
    "Bancos e Cartões",
    "Comércio Eletrônico",
    "Energia Elétrica",
    "Saúde - Planos",
    "Transporte Aéreo",
    "Educação",
    "Seguros",
    "Imobiliário",
    "Saneamento",
)


def get_api_key() -> str:
    """Retorna a GEMINI_API_KEY ou levanta erro claro se não configurada."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key == "sua_chave_aqui":
        raise RuntimeError(
            "GEMINI_API_KEY não configurada. Copie .env.example para .env e "
            "preencha com sua chave do Google AI Studio."
        )
    return key
