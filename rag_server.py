#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sqlite3
import re
import threading
import time
import uuid
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import requests
from requests import exceptions as requests_exceptions
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from rag_db import (
    ChunkHit,
    ensure_db,
    fts_search,
    get_entity_ids,
    graph_chunks_for_entities,
    graph_expand_entities,
    init_conn,
    rerank_with_embeddings,
    setup_logger,
)

OLLAMA_GENERATE_URL = os.environ.get("OLLAMA_GENERATE_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embeddings")
OLLAMA_STATUS_URL = os.environ.get("OLLAMA_STATUS_URL", "http://127.0.0.1:11434/api/tags")
GEN_MODEL = os.environ.get("OLLAMA_GEN_MODEL", "phi4-mini:latest")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
RAG_DB = os.environ.get("PHI4MINI_RAG_DB", "/var/lib/phi4mini/rag.sqlite")
RAG_DB_ABS = os.path.abspath(RAG_DB)
INDEX_NAME = os.environ.get("PHI4MINI_INDEX_NAME", "rag.sqlite")
INDEX_VERSION = os.environ.get("PHI4MINI_INDEX_VERSION", "v1")
TOP_K = int(os.environ.get("PHI4MINI_TOP_K", "8"))
CANDIDATE_K = int(os.environ.get("PHI4MINI_CANDIDATE_K", "40"))
RELAXED_TOP_K = int(os.environ.get("PHI4MINI_RELAXED_TOP_K", "12"))
SCORE_THRESHOLD = float(os.environ.get("PHI4MINI_SCORE_THRESHOLD", "0.0"))
RETRIEVE_TIMEOUT_S = float(os.environ.get("PHI4MINI_RETRIEVE_TIMEOUT_S", "4.0"))
EMBED_TIMEOUT_S = float(os.environ.get("PHI4MINI_EMBED_TIMEOUT_S", "6.0"))
GEN_TIMEOUT_S = float(os.environ.get("PHI4MINI_GEN_TIMEOUT_S", "45.0"))
CONCURRENCY = int(os.environ.get("PHI4MINI_MAX_CONCURRENCY", "4"))
METRICS_MAX_SAMPLES = int(os.environ.get("PHI4MINI_METRICS_SAMPLES", "500"))
OLLAMA_CONNECT_TIMEOUT_S = float(os.environ.get("PHI4MINI_OLLAMA_CONNECT_TIMEOUT_S", "5.0"))
GRAPH_ENABLED = os.environ.get("PHI4MINI_GRAPH_ENABLED", "1") != "0"
GRAPH_ENTITY_LIMIT = int(os.environ.get("PHI4MINI_GRAPH_ENTITY_LIMIT", "6"))
GRAPH_EXPAND_LIMIT = int(os.environ.get("PHI4MINI_GRAPH_EXPAND_LIMIT", "16"))
GRAPH_CHUNK_LIMIT = int(os.environ.get("PHI4MINI_GRAPH_CHUNK_LIMIT", "12"))
GRAPH_QUERY_LLM = os.environ.get("PHI4MINI_GRAPH_QUERY_LLM", "1") != "0"
GRAPH_LLM_TIMEOUT_S = float(os.environ.get("PHI4MINI_GRAPH_LLM_TIMEOUT_S", "3.0"))
GRAPH_LLM_MODEL = os.environ.get("OLLAMA_ENTITY_MODEL", GEN_MODEL)
MAX_CTX_CHARS = int(os.environ.get("PHI4MINI_MAX_CTX_CHARS", "2200"))
CONTEXT_PER_CHUNK = int(os.environ.get("PHI4MINI_CONTEXT_PER_CHUNK", "900"))

log = setup_logger("phi4mini.rag", os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(title="Phi4Mini Global RAG", docs_url=None, redoc_url=None)
_sema = threading.BoundedSemaphore(CONCURRENCY)
_metrics_lock = threading.Lock()
_metrics: Dict[str, int] = {"requests_total": 0, "empty_retrieval": 0, "errors_total": 0}
_latency_samples: List[float] = []

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "why",
    "with",
    "you",
    "your",
}
_GENERIC_TERMS = {
    "advantages",
    "about",
    "approach",
    "approaches",
    "beyond",
    "benefits",
    "concerning",
    "guide",
    "introduction",
    "main",
    "news",
    "overview",
    "regarding",
    "related",
    "report",
    "summary",
    "themes",
    "theme",
    "topic",
    "topics",
    "update",
}
_WEAK_ANCHORS = {
    "approach",
    "approaches",
    "method",
    "methods",
    "system",
    "systems",
    "study",
    "studies",
    "survey",
    "paper",
    "papers",
}
_SHORT_TOKEN_ALLOWLIST = {"rag", "llm", "nlp", "ai"}
_ENTITY_MIN_LEN = 4
_ALIAS_MAP = {
    "graphrag": ["graphrag", "graph rag", "rag", "retrieval augmented", "retrieval-augmented"],
    "rag": ["rag", "retrieval augmented", "retrieval-augmented"],
    "retrieval-augmented": ["retrieval augmented", "retrieval-augmented", "rag"],
}

log.info(
    "rag_server_start db=%s index=%s version=%s top_k=%d candidate_k=%d relaxed_top_k=%d "
    "score_threshold=%.3f graph=%s graph_entity_limit=%d graph_expand_limit=%d "
    "graph_chunk_limit=%d graph_query_llm=%s graph_llm_timeout_s=%.1f",
    RAG_DB_ABS,
    INDEX_NAME,
    INDEX_VERSION,
    TOP_K,
    CANDIDATE_K,
    RELAXED_TOP_K,
    SCORE_THRESHOLD,
    GRAPH_ENABLED,
    GRAPH_ENTITY_LIMIT,
    GRAPH_EXPAND_LIMIT,
    GRAPH_CHUNK_LIMIT,
    GRAPH_QUERY_LLM,
    GRAPH_LLM_TIMEOUT_S,
)


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>{title}</title>
  <style>
    body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;max-width:980px;margin:30px auto;padding:0 16px;}}
    input,textarea,button{{font:inherit;}}
    textarea{{width:100%;min-height:120px;}}
    .card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin:16px 0;}}
    .muted{{color:#555;}}
    a{{color:#0b5; text-decoration:none;}}
    a:hover{{text-decoration:underline;}}
    code,pre{{background:#f6f6f6;border-radius:8px;padding:2px 6px;}}
    pre{{padding:12px;overflow:auto;}}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def _embed(text: str) -> Optional[List[float]]:
    try:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=(OLLAMA_CONNECT_TIMEOUT_S, EMBED_TIMEOUT_S),
        )
        if r.status_code != 200:
            return None
        data = r.json()
        emb = data.get("embedding")
        if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
            return [float(x) for x in emb]
        return None
    except Exception:
        return None


def _generate(prompt: str) -> str:
    payload = {
        "model": GEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": int(os.environ.get("NUM_CTX", "4096")),
            "temperature": float(os.environ.get("TEMP", "0.2")),
            "num_predict": int(os.environ.get("PHI4MINI_MAX_TOKENS", "512")),
        },
        "keep_alive": os.environ.get("KEEP_ALIVE", "10m"),
    }
    r = requests.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=(OLLAMA_CONNECT_TIMEOUT_S, GEN_TIMEOUT_S),
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _build_prompt(question: str, contexts: List[ChunkHit]) -> str:
    ctx_blocks = []
    used_chars = 0
    for i, h in enumerate(contexts, start=1):
        meta = f"{h.title or 'Untitled'}\n{h.source_url or ''}".strip()
        text = (h.text or "").strip()
        if CONTEXT_PER_CHUNK > 0 and len(text) > CONTEXT_PER_CHUNK:
            text = text[:CONTEXT_PER_CHUNK].rstrip()
        block = f"[Source {i}]\n{meta}\n---\n{text}\n"
        if MAX_CTX_CHARS > 0 and used_chars + len(block) > MAX_CTX_CHARS:
            break
        ctx_blocks.append(block)
        used_chars += len(block)

    joined = "\n".join(ctx_blocks)
    return f"""You are a helpful assistant. Answer the user's question using ONLY the provided sources.
If the sources are insufficient, say so and suggest what to look for.
Cite sources like [Source 1], [Source 2] inline.

Question:
{question}

Sources:
{joined}

Answer:"""


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "")]


def _signal_tokens(question: str) -> List[str]:
    tokens = [
        t
        for t in _tokenize(question)
        if t not in _STOPWORDS and (len(t) >= 4 or t in _SHORT_TOKEN_ALLOWLIST)
    ]
    return tokens


def _focus_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _GENERIC_TERMS]


def _anchor_phrases(question: str, focus_tokens: List[str], signal_tokens: List[str]) -> List[str]:
    raw = (question or "").lower()
    anchors: List[str] = []
    alias_matched = False
    for t in focus_tokens:
        if t in _WEAK_ANCHORS:
            continue
        anchors.append(t)
    for alias, expansions in _ALIAS_MAP.items():
        if alias in raw:
            anchors = list(expansions)
            alias_matched = True
            break
    if not alias_matched and not anchors and signal_tokens:
        anchors = [t for t in signal_tokens if t not in _WEAK_ANCHORS]
    if not anchors:
        return []
    # Keep phrases; only normalize single-word anchors.
    normalized: List[str] = []
    for a in anchors:
        if " " in a or "-" in a:
            normalized.append(a.strip().lower())
        else:
            normalized.extend(_normalize_entity_tokens([a]))
    return normalized


def _anchor_match(hay: str, anchor: str) -> bool:
    a = anchor.lower().strip()
    if not a:
        return False
    if " " in a or "-" in a:
        return a in hay
    if len(a) <= 4:
        return re.search(rf"\\b{re.escape(a)}\\b", hay) is not None
    return a in hay


def _hits_contain_any(hits: List[ChunkHit], anchors: List[str]) -> bool:
    if not hits or not anchors:
        return False
    token_set = {t.lower() for t in anchors}
    for h in hits:
        hay = f"{h.title} {h.text}".lower()
        if any(_anchor_match(hay, a) for a in token_set):
            return True
    return False


def _expand_tokens(question: str, tokens: List[str]) -> List[str]:
    q = (question or "").lower()
    extra: List[str] = []
    if "rag" in q or "retrieval augmented generation" in q or "graphrag" in q:
        extra.extend(["retrieval", "augmented", "generation"])
    if "graphrag" in q or "graph rag" in q:
        extra.append("graph")
    return [t for t in extra if t not in tokens]


def _normalize_entity_tokens(tokens: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for t in tokens:
        tl = (t or "").strip().lower()
        if not tl or tl in _STOPWORDS or tl in _GENERIC_TERMS:
            continue
        if len(tl) < _ENTITY_MIN_LEN and tl not in _SHORT_TOKEN_ALLOWLIST:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(tl)
    return out


def _llm_entity_extract(question: str, limit: int) -> List[str]:
    if not GRAPH_QUERY_LLM:
        return []
    prompt = (
        "Extract up to {n} key entity keywords (single words) for GraphRAG lookup. "
        "Return ONLY a JSON array of lower-case strings, no extra text.\n"
        "Question: {q}\nJSON:"
    ).format(n=limit, q=question)
    try:
        r = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": GRAPH_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 64},
            },
            timeout=(OLLAMA_CONNECT_TIMEOUT_S, GRAPH_LLM_TIMEOUT_S),
        )
        if r.status_code != 200:
            return []
        text = (r.json().get("response") or "").strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", text, re.S)
            if not m:
                return []
            data = json.loads(m.group(0))
        if isinstance(data, dict):
            data = data.get("entities", [])
        if not isinstance(data, list):
            return []
        tokens: List[str] = []
        for item in data:
            if isinstance(item, str):
                tokens.extend(_tokenize(item))
        return tokens
    except Exception:
        return []


def _extractive_answer(question: str, contexts: List[ChunkHit], focus_tokens: List[str]) -> str:
    tokens = focus_tokens or _signal_tokens(question)
    token_set = {t.lower() for t in tokens}
    snippets: List[str] = []
    seen = set()

    for idx, h in enumerate(contexts, start=1):
        text = (h.text or "").replace("\n", " ").strip()
        if not text:
            continue
        parts = re.split(r"(?<=[.!?])\\s+", text)
        for s in parts:
            s_clean = s.strip()
            if len(s_clean) < 40 or len(s_clean) > 240:
                continue
            if token_set and not any(t in s_clean.lower() for t in token_set):
                continue
            key = (s_clean[:120], idx)
            if key in seen:
                continue
            seen.add(key)
            snippets.append(f"- {s_clean} [Source {idx}]")
            if len(snippets) >= 3:
                break
        if len(snippets) >= 3:
            break

    if not snippets:
        for idx, h in enumerate(contexts[:2], start=1):
            text = (h.text or "").replace("\n", " ").strip()
            if not text:
                continue
            s_clean = text[:200].rstrip()
            snippets.append(f"- {s_clean} [Source {idx}]")

    if not snippets:
        return "I could not generate an answer from the available sources."

    return "Here are relevant points from the indexed sources:\n" + "\n".join(snippets)


def _retrieve(conn: sqlite3.Connection, question: str) -> Tuple[List[ChunkHit], Dict[str, object]]:
    signal_tokens = _signal_tokens(question)
    focus_tokens = _focus_tokens(signal_tokens)
    expanded_tokens = _expand_tokens(question, signal_tokens)
    anchor_phrases = _anchor_phrases(question, focus_tokens, signal_tokens)
    if focus_tokens:
        query_text = " ".join(focus_tokens)
    elif signal_tokens:
        query_text = " ".join(signal_tokens)
    else:
        query_text = question
    # Lexical candidates (strict AND match)
    hits = fts_search(conn, query=query_text, source_type="outbox", user_id=None, limit=CANDIDATE_K)
    num_candidates = len(hits)

    # Optional embedding rerank
    t_embed0 = time.monotonic()
    qemb = _embed(question)
    t_embed_ms = (time.monotonic() - t_embed0) * 1000.0
    t_rerank_ms = 0.0
    if qemb is not None:
        t_rerank0 = time.monotonic()
        hits = rerank_with_embeddings(qemb, hits, conn=conn, top_k=TOP_K)
        t_rerank_ms = (time.monotonic() - t_rerank0) * 1000.0
    else:
        hits = hits[:TOP_K]

    if SCORE_THRESHOLD > 0:
        hits = [h for h in hits if h.score >= SCORE_THRESHOLD]

    used_relaxed = False
    if not hits:
        relaxed_tokens = signal_tokens + expanded_tokens
        if relaxed_tokens:
            relaxed_query = " ".join(dict.fromkeys(relaxed_tokens))
        else:
            relaxed_query = query_text
        relaxed = fts_search(
            conn,
            query=relaxed_query,
            source_type="outbox",
            user_id=None,
            limit=RELAXED_TOP_K,
            match_any=True,
        )
        num_candidates += len(relaxed)
        filter_tokens = focus_tokens or signal_tokens or expanded_tokens
        if filter_tokens:
            token_set = set(filter_tokens)
            min_match = 2 if len(token_set) >= 3 else 1
            filtered = []
            for h in relaxed:
                hay = f"{h.title} {h.text}".lower()
                match_count = sum(1 for t in token_set if t in hay)
                if match_count >= min_match:
                    filtered.append((match_count, h))
            filtered.sort(key=lambda x: x[0], reverse=True)
            hits = [h for _, h in filtered[:RELAXED_TOP_K]]
        else:
            hits = relaxed[:RELAXED_TOP_K]
        used_relaxed = True

    graph_used = False
    graph_entity_ids: List[int] = []
    graph_chunk_count = 0
    if GRAPH_ENABLED:
        base_tokens = focus_tokens or signal_tokens
        entity_names = _normalize_entity_tokens(base_tokens + expanded_tokens)[:GRAPH_ENTITY_LIMIT]
        if not entity_names:
            llm_tokens = _llm_entity_extract(question, GRAPH_ENTITY_LIMIT * 2)
            entity_names = _normalize_entity_tokens(llm_tokens)[:GRAPH_ENTITY_LIMIT]
        if entity_names:
            graph_entity_ids = get_entity_ids(conn, entity_names)
        if graph_entity_ids:
            expanded = graph_expand_entities(conn, graph_entity_ids, limit=GRAPH_EXPAND_LIMIT)
            expanded_ids = [eid for eid, _ in expanded]
            all_ids: List[int] = []
            for eid in graph_entity_ids + expanded_ids:
                if eid not in all_ids:
                    all_ids.append(eid)
            graph_hits = graph_chunks_for_entities(
                conn,
                entity_ids=all_ids,
                limit=GRAPH_CHUNK_LIMIT,
                source_type="outbox",
                user_id=None,
            )
            graph_chunk_count = len(graph_hits)
            graph_used = True
            if not hits:
                hits = graph_hits
            elif graph_hits:
                seen = {h.chunk_id for h in hits}
                for h in graph_hits:
                    if h.chunk_id not in seen:
                        hits.append(h)
                        seen.add(h.chunk_id)
            if len(hits) > TOP_K:
                hits = hits[:TOP_K]

    anchor_matched = False
    if anchor_phrases:
        anchor_hits = [h for h in hits if _hits_contain_any([h], anchor_phrases)]
        if anchor_hits:
            hits = anchor_hits[:TOP_K]
            anchor_matched = True

    meta = {
        "num_candidates": num_candidates,
        "num_returned": len(hits),
        "t_embed_ms": round(t_embed_ms, 1),
        "t_rerank_ms": round(t_rerank_ms, 1),
        "used_relaxed": used_relaxed,
        "signal_tokens": signal_tokens,
        "focus_tokens": focus_tokens,
        "anchor_tokens": anchor_phrases,
        "anchor_matched": anchor_matched,
        "graph_used": graph_used,
        "graph_entity_ids": graph_entity_ids,
        "graph_chunk_count": graph_chunk_count,
    }
    return hits, meta


def _set_request_id(response: Response, request_id: str) -> None:
    response.headers["x-request-id"] = request_id


def _note_latency(t_total_ms: float) -> None:
    with _metrics_lock:
        _metrics["requests_total"] += 1
        _latency_samples.append(t_total_ms)
        if len(_latency_samples) > METRICS_MAX_SAMPLES:
            _latency_samples.pop(0)


def _record_empty() -> None:
    with _metrics_lock:
        _metrics["empty_retrieval"] += 1


def _record_error() -> None:
    with _metrics_lock:
        _metrics["errors_total"] += 1


def _p95(samples: List[float]) -> Optional[float]:
    if not samples:
        return None
    s = sorted(samples)
    idx = int(0.95 * (len(s) - 1))
    return s[idx]


def _make_request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex


def _acquire_slot(timeout_s: float = 2.0) -> bool:
    return _sema.acquire(timeout=timeout_s)


def _release_slot() -> None:
    try:
        _sema.release()
    except ValueError:
        pass


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    body = """
    <h1>Global RAG</h1>
    <p class='muted'>Answers questions using the data your summarizer collects (outbox.jsonl → rag.sqlite).</p>
    <div class='card'>
      <form method='post' action='ask'>
        <label><b>Question</b></label><br/>
        <textarea name='question' placeholder='Ask about RAG, chunking, evals, local inference, etc.'></textarea><br/>
        <button type='submit'>Ask</button>
      </form>
    </div>
    <p class='muted'>Tip: Keep the summarizer running. The indexer updates during the old 15-minute sleep window.</p>
    """
    return _html_page("Global RAG", body)


@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)) -> HTMLResponse:
    request_id = _make_request_id(request)
    question = (question or "").strip()
    if not question:
        resp = HTMLResponse(_html_page("Global RAG", "<p>Missing question.</p>"))
        _set_request_id(resp, request_id)
        return resp

    acquired = _acquire_slot()
    if not acquired:
        _record_error()
        resp = HTMLResponse("<p>Server busy. Please retry.</p>", status_code=429)
        _set_request_id(resp, request_id)
        return resp

    try:
        conn = init_conn(RAG_DB)
        ensure_db(conn)

        t0 = time.monotonic()
        t_retrieve = t_llm = 0.0
        ctx: List[ChunkHit] = []
        retrieval_meta: Dict[str, object] = {}
        try:
            t_r0 = time.monotonic()
            ctx, retrieval_meta = _retrieve(conn, question)
            t_retrieve = (time.monotonic() - t_r0) * 1000.0
            if t_retrieve / 1000.0 > RETRIEVE_TIMEOUT_S:
                log.warning(
                    "request_id=%s stage=retrieve timeout_s=%.2f elapsed_ms=%.1f",
                    request_id,
                    RETRIEVE_TIMEOUT_S,
                    t_retrieve,
                )
        except Exception as e:
            _record_error()
            log.exception("request_id=%s stage=retrieve error=%s", request_id, e)

        if not ctx or (
            retrieval_meta.get("anchor_tokens")
            and not bool(retrieval_meta.get("anchor_matched", False))
        ):
            _record_empty()
            body = "<h1>Global RAG</h1><p>No relevant sources found for this query. Try different keywords.</p>"
            resp = HTMLResponse(_html_page("Global RAG", body))
            _set_request_id(resp, request_id)
            _note_latency((time.monotonic() - t0) * 1000.0)
            return resp

        prompt = _build_prompt(question, ctx)
        try:
            t_llm0 = time.monotonic()
            answer = _generate(prompt)
            t_llm = (time.monotonic() - t_llm0) * 1000.0
            llm_used = True
        except requests_exceptions.RequestException as e:
            _record_error()
            answer = _extractive_answer(question, ctx, retrieval_meta.get("focus_tokens", []))
            llm_used = False
        except Exception as e:
            _record_error()
            answer = _extractive_answer(question, ctx, retrieval_meta.get("focus_tokens", []))
            llm_used = False

        sources_html = "".join(
            [
                f"<li><a href='{(h.source_url or '#')}' target='_blank' rel='noreferrer'>{(h.title or h.source_url or 'source')}</a></li>"
                for h in ctx
            ]
        )

        body = f"""
    <h1>Global RAG</h1>
    <div class='card'>
      <p><b>Question</b></p>
      <pre>{question}</pre>
    </div>
    <div class='card'>
      <p><b>Answer</b></p>
      <pre>{answer}</pre>
    </div>
        <div class='card'>
          <p><b>Sources</b></p>
          <ol>{sources_html}</ol>
        </div>
        <p><a href='/'>Ask another</a></p>
        """
        t_total = (time.monotonic() - t0) * 1000.0
        log.info(
            "request_id=%s db=%s index=%s version=%s top_k=%d score_threshold=%.3f candidate_k=%d relaxed_top_k=%d used_relaxed=%s t_total_ms=%.1f t_retrieve_ms=%.1f t_embed_ms=%.1f t_rerank_ms=%.1f t_llm_ms=%.1f num_candidates=%d num_returned=%d",
            request_id,
            RAG_DB_ABS,
            INDEX_NAME,
            INDEX_VERSION,
            TOP_K,
            SCORE_THRESHOLD,
            CANDIDATE_K,
            RELAXED_TOP_K,
            bool(retrieval_meta.get("used_relaxed", False)),
            t_total,
            t_retrieve,
            float(retrieval_meta.get("t_embed_ms", 0.0)),
            float(retrieval_meta.get("t_rerank_ms", 0.0)),
            t_llm,
            int(retrieval_meta.get("num_candidates", 0)),
            int(retrieval_meta.get("num_returned", 0)),
        )
        _note_latency(t_total)
        resp = HTMLResponse(_html_page("Global RAG", body))
        _set_request_id(resp, request_id)
        return resp
    finally:
        if acquired:
            _release_slot()


@app.get("/api/ask")
def api_ask(request: Request, q: str = Query(..., description="Question")) -> JSONResponse:
    request_id = _make_request_id(request)
    question = (q or "").strip()
    conn = init_conn(RAG_DB)
    ensure_db(conn)

    acquired = _acquire_slot()
    if not acquired:
        _record_error()
        resp = JSONResponse({"error": "server_busy"}, status_code=429)
        _set_request_id(resp, request_id)
        return resp

    try:
        t0 = time.monotonic()
        t_retrieve = t_llm = 0.0
        ctx: List[ChunkHit] = []
        retrieval_meta: Dict[str, object] = {}
        try:
            t_r0 = time.monotonic()
            ctx, retrieval_meta = _retrieve(conn, question)
            t_retrieve = (time.monotonic() - t_r0) * 1000.0
        except Exception as e:
            _record_error()
            log.exception("request_id=%s stage=retrieve error=%s", request_id, e)

        prompt = _build_prompt(question, ctx)
        answer = ""
        llm_used = False
        if ctx and (
            not retrieval_meta.get("anchor_tokens")
            or bool(retrieval_meta.get("anchor_matched", False))
        ):
            try:
                t_llm0 = time.monotonic()
                answer = _generate(prompt)
                t_llm = (time.monotonic() - t_llm0) * 1000.0
                llm_used = True
            except requests_exceptions.RequestException:
                _record_error()
                answer = _extractive_answer(question, ctx, retrieval_meta.get("focus_tokens", []))
                llm_used = False
            except Exception:
                _record_error()
                answer = _extractive_answer(question, ctx, retrieval_meta.get("focus_tokens", []))
                llm_used = False
        else:
            _record_empty()
            answer = "No relevant sources found for this query. Try different keywords."

        t_total = (time.monotonic() - t0) * 1000.0
        log.info(
            "request_id=%s db=%s index=%s version=%s top_k=%d score_threshold=%.3f candidate_k=%d relaxed_top_k=%d used_relaxed=%s t_total_ms=%.1f t_retrieve_ms=%.1f t_embed_ms=%.1f t_rerank_ms=%.1f t_llm_ms=%.1f num_candidates=%d num_returned=%d",
            request_id,
            RAG_DB_ABS,
            INDEX_NAME,
            INDEX_VERSION,
            TOP_K,
            SCORE_THRESHOLD,
            CANDIDATE_K,
            RELAXED_TOP_K,
            bool(retrieval_meta.get("used_relaxed", False)),
            t_total,
            t_retrieve,
            float(retrieval_meta.get("t_embed_ms", 0.0)),
            float(retrieval_meta.get("t_rerank_ms", 0.0)),
            t_llm,
            int(retrieval_meta.get("num_candidates", 0)),
            int(retrieval_meta.get("num_returned", 0)),
        )
        _note_latency(t_total)
        payload = {
            "request_id": request_id,
            "question": question,
            "answer": answer,
            "sources": [asdict(h) for h in ctx],
            "gen_model": GEN_MODEL,
            "embed_model": EMBED_MODEL,
            "db_path": RAG_DB_ABS,
            "index_name": INDEX_NAME,
            "index_version": INDEX_VERSION,
            "top_k": TOP_K,
            "score_threshold": SCORE_THRESHOLD,
            "num_candidates": int(retrieval_meta.get("num_candidates", 0)),
            "num_returned": int(retrieval_meta.get("num_returned", 0)),
            "llm_used": llm_used,
            "graph_used": bool(retrieval_meta.get("graph_used", False)),
            "graph_chunk_count": int(retrieval_meta.get("graph_chunk_count", 0)),
            "timings_ms": {
                "t_total": round(t_total, 1),
                "t_retrieve": round(t_retrieve, 1),
                "t_embed_query": retrieval_meta.get("t_embed_ms", 0.0),
                "t_rerank": retrieval_meta.get("t_rerank_ms", 0.0),
                "t_llm": round(t_llm, 1),
            },
        }
        resp = JSONResponse(payload)
        _set_request_id(resp, request_id)
        return resp
    finally:
        if acquired:
            _release_slot()


@app.get("/health")
def health() -> JSONResponse:
    status = "ok"
    details: Dict[str, object] = {
        "db_path": RAG_DB_ABS,
        "index_name": INDEX_NAME,
        "index_version": INDEX_VERSION,
        "gen_model": GEN_MODEL,
        "embed_model": EMBED_MODEL,
    }
    try:
        conn = init_conn(RAG_DB)
        ensure_db(conn)
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        chunk_count = int(row[0]) if row else 0
        details["chunk_count"] = chunk_count
        if chunk_count <= 0:
            status = "degraded"
    except Exception as e:
        status = "error"
        details["db_error"] = f"{type(e).__name__}: {e}"

    try:
        r = requests.get(OLLAMA_STATUS_URL, timeout=(OLLAMA_CONNECT_TIMEOUT_S, 5.0))
        details["ollama_ok"] = r.status_code == 200
        if r.status_code != 200:
            status = "degraded"
    except Exception as e:
        details["ollama_ok"] = False
        details["ollama_error"] = f"{type(e).__name__}: {e}"
        status = "degraded"

    return JSONResponse({"status": status, "details": details})


@app.get("/metrics")
def metrics() -> JSONResponse:
    with _metrics_lock:
        samples = list(_latency_samples)
        snapshot = dict(_metrics)
    return JSONResponse(
        {
            "counts": snapshot,
            "latency_ms": {
                "p95": _p95(samples),
                "samples": len(samples),
            },
        }
    )


@app.get("/debug/retrieval")
def debug_retrieval(request: Request, q: str = Query(..., description="Question")) -> JSONResponse:
    request_id = _make_request_id(request)
    question = (q or "").strip()
    conn = init_conn(RAG_DB)
    ensure_db(conn)

    ctx: List[ChunkHit] = []
    retrieval_meta: Dict[str, object] = {}
    try:
        ctx, retrieval_meta = _retrieve(conn, question)
    except Exception as e:
        _record_error()
        log.exception("request_id=%s stage=retrieve error=%s", request_id, e)

    top_chunks = []
    for h in ctx[:3]:
        snippet = (h.text or "").replace("\n", " ").strip()
        top_chunks.append(
            {
                "chunk_id": h.chunk_id,
                "title": h.title,
                "source_url": h.source_url,
                "score": h.score,
                "snippet": snippet[:200],
            }
        )

        payload = {
            "request_id": request_id,
            "db_path": RAG_DB_ABS,
            "index_name": INDEX_NAME,
            "index_version": INDEX_VERSION,
            "source_type": "outbox",
            "top_k": TOP_K,
            "score_threshold": SCORE_THRESHOLD,
            "candidate_k": CANDIDATE_K,
            "relaxed_top_k": RELAXED_TOP_K,
            "used_relaxed": bool(retrieval_meta.get("used_relaxed", False)),
            "signal_tokens": retrieval_meta.get("signal_tokens", []),
            "focus_tokens": retrieval_meta.get("focus_tokens", []),
            "anchor_tokens": retrieval_meta.get("anchor_tokens", []),
            "anchor_matched": bool(retrieval_meta.get("anchor_matched", False)),
            "graph_used": bool(retrieval_meta.get("graph_used", False)),
            "graph_entity_ids": retrieval_meta.get("graph_entity_ids", []),
            "graph_chunk_count": int(retrieval_meta.get("graph_chunk_count", 0)),
            "num_candidates": int(retrieval_meta.get("num_candidates", 0)),
            "num_returned": int(retrieval_meta.get("num_returned", 0)),
        "timings_ms": {
            "t_embed_query": retrieval_meta.get("t_embed_ms", 0.0),
            "t_rerank": retrieval_meta.get("t_rerank_ms", 0.0),
        },
        "top_chunks": top_chunks,
    }
    resp = JSONResponse(payload)
    _set_request_id(resp, request_id)
    return resp
