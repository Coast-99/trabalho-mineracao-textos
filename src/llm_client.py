"""Cliente Gemini com saída estruturada validada por Pydantic.

Reusa o padrão ensinado em `Notebook_extracao_dados_com_llm.ipynb`:
  client.models.generate_content(
      model=..., contents=prompt,
      config=types.GenerateContentConfig(
          temperature=0, response_mime_type="application/json",
          response_schema=ModeloPydantic))
  ModeloPydantic.model_validate_json(resp.text)

Acréscimos exigidos pelo enunciado (LLM confiável como software):
- TRATAMENTO DE FALHA DE VALIDAÇÃO: try/except em ValidationError + erros de
  API; 1 retry; se persistir, devolve fallback marcado com `_ok=False`.
- CACHE em JSONL por `id`: reexecutar não re-chama a API para itens já feitos.
- contagem de tokens (`contar_tokens`) para a estimativa de custo.
"""
from __future__ import annotations

import json
import re
import threading
import time

from pydantic import BaseModel, ValidationError

from . import config


def _retry_delay(msg: str) -> float | None:
    """Extrai o retryDelay (segundos) que o erro 429 da API sugere, se houver."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)s", msg)
    return float(m.group(1)) if m else None


# --- Rate limiter global (calibrado para o free tier ~15 req/min) ----------
# Garante um intervalo mínimo entre o INÍCIO de chamadas, mesmo com várias
# threads. Marcar o passo logo abaixo do limite evita a cascata de 429 que
# derruba a qualidade (itens caindo no fallback).
_rl_lock = threading.Lock()
_rl_proxima = [0.0]


def _throttle(intervalo: float) -> None:
    if intervalo <= 0:
        return
    with _rl_lock:
        agora = time.monotonic()
        espera = _rl_proxima[0] - agora
        if espera > 0:
            time.sleep(espera)
        _rl_proxima[0] = max(agora, _rl_proxima[0]) + intervalo

_client = None


def get_client():
    """Cria o client Gemini uma vez (lazy)."""
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=config.get_api_key())
    return _client


def _chamar(prompt: str, schema: type[BaseModel]):
    """Uma chamada crua ao Gemini, devolve o objeto validado."""
    from google.genai import types

    resp = get_client().models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return schema.model_validate_json(resp.text)


def analisar(prompt: str, schema: type[BaseModel], *, tentativas: int = 10, intervalo: float = 0.0):
    """Chama o LLM com retry. Retorna (dict, ok: bool, erro: str|None).

    Em sucesso: (dados_validados, True, None).
    Em falha após as tentativas: (FALLBACK, False, mensagem_de_erro).
    `intervalo` aplica o rate limiter global antes de cada chamada; o retry
    honra o retryDelay do 429 e faz backoff exponencial nos demais erros.
    """
    from .schemas import FALLBACK

    ultimo_erro = None
    for tentativa in range(1, tentativas + 1):
        espera = min(2 ** tentativa, 30)  # backoff exponencial padrão
        try:
            _throttle(intervalo)
            obj = _chamar(prompt, schema)
            return obj.model_dump(), True, None
        except ValidationError as e:
            ultimo_erro = f"ValidationError: {str(e)[:200]}"
        except Exception as e:  # erros de API / rede / parsing
            msg = str(e)
            ultimo_erro = f"{type(e).__name__}: {msg[:200]}"
            # Em 429, honra o retryDelay sugerido pela API (+ folga).
            rd = _retry_delay(msg)
            if rd is not None:
                espera = rd + 2
        if tentativa < tentativas:
            time.sleep(espera)
    return dict(FALLBACK), False, ultimo_erro


def processar_corpus(
    itens, montar_prompt, schema, cache_nome: str, *, workers: int = 3,
    intervalo: float = 4.5,
):
    """Processa uma lista de itens com cache incremental em JSONL.

    Chamadas são paralelizadas (ThreadPool) porque o gargalo é a latência da API,
    não CPU. O cache é compartilhado entre threads com um lock; só sucessos são
    persistidos, então reexecutar retenta os itens que falharam.

    `itens`: lista de dicts com pelo menos a chave 'id'.
    `montar_prompt(item) -> str`. Retorna a lista de registros {id, ok, erro, ...}.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = config.DATA_PROC / cache_nome
    feitos = {}
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            for linha in f:
                if linha.strip():
                    r = json.loads(linha)
                    feitos[r["id"]] = r

    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        def tqdm(x, **k):
            return x

    pendentes = [it for it in itens if it["id"] not in feitos]
    resultados = [feitos[it["id"]] for it in itens if it["id"] in feitos]
    lock = threading.Lock()

    def worker(item):
        dados, ok, erro = analisar(montar_prompt(item), schema, intervalo=intervalo)
        registro = {"id": item["id"], "ok": ok, "erro": erro, **dados}
        if ok:
            with lock:
                with open(cache, "a", encoding="utf-8") as f:
                    f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return registro

    if pendentes:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futuros = [ex.submit(worker, it) for it in pendentes]
            for fut in tqdm(as_completed(futuros), total=len(pendentes)):
                resultados.append(fut.result())

    print(f"  cache '{cache_nome}': {len(resultados)} itens ({len(pendentes)} novos via API)")
    falhas = sum(1 for r in resultados if not r["ok"])
    if falhas:
        print(f"  ATENÇÃO: {falhas} itens caíram no fallback (ver coluna 'erro').")
    return resultados


def contar_tokens(prompt: str) -> int:
    """Conta tokens de entrada de um prompt (para estimar custo)."""
    resp = get_client().models.count_tokens(model=config.LLM_MODEL, contents=prompt)
    return resp.total_tokens
