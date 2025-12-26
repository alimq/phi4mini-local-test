#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from rag_db import (
    ensure_db,
    init_conn,
    insert_chunk,
    fts_search,
    rerank_with_embeddings,
    setup_logger,
)

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None  # type: ignore

OLLAMA_GENERATE_URL = os.environ.get("OLLAMA_GENERATE_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embeddings")
GEN_MODEL = os.environ.get("OLLAMA_GEN_MODEL", "phi4-mini:latest")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
RAG_DB = os.environ.get("PHI4MINI_RAG_DB", "/var/lib/phi4mini/rag.sqlite")
UPLOAD_ROOT = Path(os.environ.get("PHI4MINI_UPLOAD_ROOT", "/var/lib/phi4mini/user_uploads"))

log = setup_logger("phi4mini.rag_user", os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI(title="Phi4Mini User-Docs RAG", docs_url=None, redoc_url=None)

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
_SHORT_TOKEN_ALLOWLIST = {"rag", "llm", "ai", "nlp", "ml"}


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
        r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
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
        "options": {"num_ctx": int(os.environ.get("NUM_CTX", "4096")), "temperature": float(os.environ.get("TEMP", "0.2"))},
        "keep_alive": os.environ.get("KEEP_ALIVE", "10m"),
    }
    r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=600)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_upload(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".csv", ".log"}:
        return file_path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        return file_path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf" and PdfReader is not None:
        reader = PdfReader(str(file_path))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    if suffix == ".docx" and Document is not None:
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)
    # Fallback: try decode bytes as text
    return file_path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in text.splitlines() if p.strip()]
    if not paras:
        paras = [text]

    chunks: List[str] = []
    cur = ""
    for p in paras:
        if not cur:
            cur = p
            continue
        if len(cur) + 1 + len(p) <= max_chars:
            cur = cur + "\n" + p
        else:
            chunks.append(cur)
            if overlap > 0 and len(cur) > overlap:
                cur = cur[-overlap:] + "\n" + p
            else:
                cur = p
    if cur:
        chunks.append(cur)

    out: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            out.append(c)
        else:
            for i in range(0, len(c), max_chars):
                out.append(c[i : i + max_chars])
    return out


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "")]


def _keyword_tags(text: str, max_tags: int = 12) -> List[str]:
    tokens = [
        t
        for t in _tokenize(text)
        if t not in _STOPWORDS and (len(t) >= 4 or t in _SHORT_TOKEN_ALLOWLIST)
    ]
    if not tokens:
        return []
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in ranked[:max_tags]]


def _build_prompt(question: str, contexts, user_id: str) -> str:
    blocks = []
    for i, h in enumerate(contexts, start=1):
        meta = f"{h.title or 'Untitled'}\n{h.source_url or ''}".strip()
        blocks.append(f"[Source {i}]\n{meta}\n---\n{h.text}\n")
    joined = "\n".join(blocks)
    return f"""You are a helpful assistant. Answer the user's question using ONLY the provided sources.
If the sources are insufficient, say so and ask the user to upload additional documents.
Cite sources like [Source 1], [Source 2] inline.

User: {user_id}

Question:
{question}

Sources:
{joined}

Answer:"""


def _retrieve(conn: sqlite3.Connection, question: str, user_id: str, top_k: int = 8):
    hits = fts_search(conn, query=question, source_type="user", user_id=user_id, limit=40)
    qemb = _embed(question)
    if qemb is not None:
        hits = rerank_with_embeddings(qemb, hits, conn=conn, top_k=top_k)
    else:
        hits = hits[:top_k]
    return hits


def _list_user_docs(conn: sqlite3.Connection, user_id: str) -> List[Tuple[str, str]]:
    rows = conn.execute(
        "SELECT DISTINCT doc_id, COALESCE(title,'' ) FROM chunks WHERE source_type='user' AND user_id=? ORDER BY created_at_utc DESC LIMIT 50",
        (user_id or "",),
    ).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows]


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    body = """
    <h1>User-docs RAG</h1>
    <p class='muted'>Upload documents and ask questions over them (per-user namespace).</p>

    <div class='card'>
      <h3>Upload</h3>
      <form method='post' action='/upload' enctype='multipart/form-data'>
        <label><b>User ID</b> (anything; used to separate your docs)</label><br/>
        <input name='user_id' placeholder='e.g. alice' required/>
        <br/><br/>
        <input type='file' name='file' required/>
        <button type='submit'>Upload & Index</button>
      </form>
      <p class='muted'>Supports .txt/.md/.json/.csv/.html and (optional) .pdf/.docx if deps are installed.</p>
    </div>

    <div class='card'>
      <h3>Ask</h3>
      <form method='post' action='ask'>
        <label><b>User ID</b></label><br/>
        <input name='user_id' placeholder='e.g. alice' required/>
        <br/><br/>
        <label><b>Question</b></label><br/>
        <textarea name='question' placeholder='Ask something that is answered by your uploaded docs.'></textarea><br/>
        <button type='submit'>Ask</button>
      </form>
    </div>
    """
    return _html_page("User-docs RAG", body)


@app.post("/upload", response_class=HTMLResponse)
async def upload(user_id: str = Form(...), file: UploadFile = File(...)) -> str:
    user_id = (user_id or "").strip()
    if not user_id:
        return _html_page("Upload", "<p>Missing user_id</p>")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    user_dir = UPLOAD_ROOT / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    raw = await file.read()
    digest = _sha256(raw)
    safe_name = (file.filename or "upload").replace("/", "_").replace("\\", "_")
    dest = user_dir / f"{digest[:12]}_{safe_name}"
    dest.write_bytes(raw)

    # Extract text, chunk, embed and write to DB
    conn = init_conn(RAG_DB)
    ensure_db(conn)

    try:
        text = _read_upload(dest)
    except Exception as e:
        return _html_page("Upload", f"<p>Failed to read file: {type(e).__name__}: {e}</p>")

    text = (text or "").strip()
    if not text:
        return _html_page("Upload", "<p>File contained no extractable text.</p>")

    chunks = _chunk_text(text)
    embedded = 0
    for idx, chunk in enumerate(chunks):
        tags = _keyword_tags(f"{safe_name}\n{chunk}")
        fts_text = chunk if not tags else f"{chunk}\n\nTAGS: {' '.join(tags)}"
        emb = _embed(chunk)
        if emb is not None:
            embedded += 1
        insert_chunk(
            conn,
            doc_id=digest,
            user_id=user_id,
            source_type="user",
            source_url=str(dest),
            title=safe_name,
            chunk_index=idx,
            text=chunk,
            fts_text=fts_text,
            embedding=emb,
            embedding_model=EMBED_MODEL if emb is not None else "",
        )

    docs = _list_user_docs(conn, user_id)
    docs_html = "".join([f"<li><code>{d}</code> — {t}</li>" for d, t in docs])

    body = f"""
    <h1>Uploaded</h1>
    <p class='muted'>User: <code>{user_id}</code></p>
    <div class='card'>
      <p><b>File</b>: {safe_name}</p>
      <p><b>Chunks</b>: {len(chunks)} (embedded: {embedded}, model: <code>{EMBED_MODEL}</code>)</p>
      <p><b>Stored at</b>: <code>{dest}</code></p>
    </div>
    <div class='card'>
      <p><b>Your docs</b></p>
      <ol>{docs_html}</ol>
    </div>
    <p><a href='/'>Back</a></p>
    """
    return _html_page("Uploaded", body)


@app.post("/ask", response_class=HTMLResponse)
def ask(user_id: str = Form(...), question: str = Form(...)) -> str:
    user_id = (user_id or "").strip()
    question = (question or "").strip()
    if not user_id or not question:
        return _html_page("Ask", "<p>Missing user_id or question</p>")

    conn = init_conn(RAG_DB)
    ensure_db(conn)

    ctx = _retrieve(conn, question, user_id=user_id)
    if not ctx:
        body = f"<h1>User-docs RAG</h1><p>No indexed chunks found for user <code>{user_id}</code>. Upload docs first.</p>"
        return _html_page("User-docs RAG", body)

    prompt = _build_prompt(question, ctx, user_id)
    try:
        answer = _generate(prompt)
    except Exception as e:
        answer = f"Error generating answer: {type(e).__name__}: {e}"

    sources_html = "".join(
        [
            f"<li><code>{h.title or 'doc'}</code><br/><span class='muted'>{h.source_url}</span></li>"
            for h in ctx
        ]
    )

    body = f"""
    <h1>User-docs RAG</h1>
    <p class='muted'>User: <code>{user_id}</code></p>
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
    <p><a href='/'>Back</a></p>
    """
    return _html_page("User-docs RAG", body)


@app.get("/api/ask")
def api_ask(user_id: str = Query(...), q: str = Query(...)) -> JSONResponse:
    user_id = (user_id or "").strip()
    question = (q or "").strip()
    conn = init_conn(RAG_DB)
    ensure_db(conn)
    ctx = _retrieve(conn, question, user_id=user_id)
    prompt = _build_prompt(question, ctx, user_id)
    answer = ""
    if ctx:
        try:
            answer = _generate(prompt)
        except Exception as e:
            answer = f"Error generating answer: {type(e).__name__}: {e}"

    return JSONResponse(
        {
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "sources": [h.__dict__ for h in ctx],
            "gen_model": GEN_MODEL,
            "embed_model": EMBED_MODEL,
        }
    )
