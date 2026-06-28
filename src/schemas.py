"""Contrato Pydantic — o coração avaliado do trabalho.

`AnaliseReclamacao` é o formato exato que exigimos do LLM. Usar `Literal`
transforma campos abertos em **vocabulário controlado**: o LLM tem que
escolher um valor da lista, o que torna a saída comparável e validável.

Justificativa dos campos (ver DECISOES.md):
- `categoria_prevista`: alvo da classificação via LLM; Literal = as 10 classes
  reais do corpus, idêntico a config.CATEGORIAS (validado abaixo).
- `tipo_problema`: natureza do problema (transversal às categorias) — vira
  feature de enriquecimento para o classificador clássico.
- `sentimento` / `urgencia`: sinais de prioridade úteis ao gestor e como
  features extras.
- `empresa_mencionada`: entidade extraída do texto (pode ser None).
- `resumo`: 1 frase neutra, limitada, para inspeção humana.
- `confianca`: autoavaliação do LLM (0–1) — usada para achar casos duvidosos.
"""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, Field

from . import config

# Vocabulário fechado de categorias (igual ao do corpus).
CategoriaLit = Literal[
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
]

# Garante que o schema não saia de sincronia com o config.
assert set(get_args(CategoriaLit)) == set(config.CATEGORIAS), (
    "CategoriaLit difere de config.CATEGORIAS"
)

SentimentoLit = Literal["muito_negativo", "negativo", "neutro", "positivo"]
UrgenciaLit = Literal["baixa", "media", "alta"]
TipoProblemaLit = Literal[
    "cobranca_indevida",
    "atendimento",
    "qualidade_servico",
    "cancelamento_contrato",
    "fraude_golpe",
    "entrega_instalacao",
    "acesso_login",
    "reembolso_estorno",
    "outro",
]


class AnaliseReclamacao(BaseModel):
    """Saída estruturada da análise de UMA reclamação."""

    categoria_prevista: CategoriaLit = Field(
        description="Setor ao qual a reclamação pertence, dentre as categorias permitidas."
    )
    tipo_problema: TipoProblemaLit = Field(
        description="Natureza principal do problema relatado."
    )
    sentimento: SentimentoLit = Field(
        description="Tom emocional predominante do texto."
    )
    urgencia: UrgenciaLit = Field(
        description="Urgência de tratamento, considerando impacto e gravidade relatados."
    )
    empresa_mencionada: str | None = Field(
        default=None,
        description="Nome da empresa citada no texto, se houver; caso contrário, null.",
    )
    resumo: str = Field(
        description="Resumo neutro do problema em uma frase.", max_length=300
    )
    confianca: float = Field(
        ge=0.0, le=1.0, description="Confiança (0 a 1) na categoria_prevista."
    )


# Valor de fallback usado quando a validação falha em definitivo (ver llm_client).
FALLBACK = {
    "categoria_prevista": "Telecomunicações",  # classe majoritária (ver nota no código)
    "tipo_problema": "outro",
    "sentimento": "negativo",
    "urgencia": "media",
    "empresa_mencionada": None,
    "resumo": "",
    "confianca": 0.0,
}
